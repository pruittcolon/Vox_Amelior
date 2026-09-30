import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

class ProcessorConfig {
  const ProcessorConfig({
    this.sampleRate = 16000,
    this.minTextChars = 2,
    this.maxEmbedSeconds = 20,
    this.minDiarizeSeconds = 2.0,
  });

  final int sampleRate;

  /// Transcripts shorter than this are noise ("uh", empty results) and dropped.
  final int minTextChars;

  /// Only the first seconds of long utterances are used for the voiceprint.
  final double maxEmbedSeconds;

  /// Shorter stretches are too short to hold a change of speaker.
  final double minDiarizeSeconds;
}

class ProcessorStats {
  int processed = 0;
  int saved = 0;
  int dropped = 0;
  int errors = 0;

  /// Stretches of speech that were split because the speaker changed.
  int split = 0;
  Duration last = Duration.zero;
}

/// The expensive step: one speech chunk → who spoke when → text → speaker →
/// saved utterances.
///
/// With a [diarizer], a stretch where the speaker changes (quick
/// back-and-forth, interruptions) is split into one line per turn: each voice
/// is named from the audio where it talks alone, each turn is transcribed on
/// its own, and turns where two people talked at once are marked as overlap.
class SegmentProcessor {
  SegmentProcessor({
    required this._asr,
    required this._embedder,
    required this.identifier,
    required this._transcripts,
    required this._speakers,
    this.config = const ProcessorConfig(),
    this.diarizer,
    this.turns = const SpeakerTurns(),
    this.onSaved,
  });

  final AsrEngine _asr;
  final EmbeddingEngine _embedder;
  final SpeakerIdentifier identifier;
  final TranscriptRepository _transcripts;
  final SpeakerRepository _speakers;
  final ProcessorConfig config;
  final ProcessorStats stats = ProcessorStats();

  /// Splits lines at speaker changes when set (the model may arrive later).
  DiarizationEngine? diarizer;

  /// Settings switch for splitting.
  bool splitSpeakers = true;
  final SpeakerTurns turns;

  /// Called with each saved utterance and its audio (e.g. to keep a clip).
  /// Failures here never lose the transcript.
  final void Function(SegmentView segment, Float32List samples)? onSaved;

  int get _sr => config.sampleRate;

  /// Returns the saved utterances: usually one, several when the speaker
  /// changes, none when the audio held no usable words.
  List<SegmentView> process(Float32List samples, DateTime startedAt) {
    final watch = Stopwatch()..start();
    stats.processed++;
    try {
      final seconds = samples.length / _sr;
      final d = splitSpeakers ? diarizer : null;
      if (d == null || seconds < config.minDiarizeSeconds) return _whole(samples, startedAt);
      final SpeakerActivity activity;
      try {
        activity = d.analyze(samples, _sr);
      } on Object catch (e) {
        Log.w('processor', 'diarizer failed; keeping the line whole', e);
        return _whole(samples, startedAt);
      }
      var parts = turns.turns(activity, totalSeconds: seconds);
      if (parts.length == 1) return _whole(samples, startedAt, overlap: parts.single.overlap);

      // Name each voice once, from the audio where it talks alone.
      final named = <int, (SpeakerMatch, Float32List)>{};
      for (final slot in {for (final t in parts) t.slot}) {
        final solo = _soloAudio(samples, activity, slot);
        final soloSeconds = solo.length / _sr;
        if (soloSeconds < identifier.config.minEmbeddingSeconds) continue;
        final e = _embedder.embed(solo, _sr);
        final m = identifier.identify(e, seconds: soloSeconds);
        if (m.cluster != null) _speakers.saveCluster(m.cluster!);
        named[slot] = (m, e);
      }
      // Two diarizer slots that turn out to be the same person: one turn.
      String who(SpeakerTurn t) {
        final m = named[t.slot]?.$1;
        return m?.speakerId ?? m?.cluster?.id ?? 'slot ${t.slot}';
      }

      parts = SpeakerTurns.joinSame(parts, who);
      if (parts.length == 1) return _whole(samples, startedAt, overlap: parts.single.overlap);

      final saved = <SegmentView>[];
      for (final t in parts) {
        final from = (t.start * _sr).round().clamp(0, samples.length);
        final to = (t.end * _sr).round().clamp(from, samples.length);
        final slice = Float32List.sublistView(samples, from, to);
        final text = _asr.transcribe(slice, _sr).trim();
        if (!_isUsable(text)) {
          stats.dropped++;
          continue;
        }
        final match = named[t.slot];
        saved.add(_save(
          text: text,
          startedAt: startedAt.add(Duration(microseconds: (t.start * 1e6).round())),
          samples: slice,
          match: match?.$1 ?? const SpeakerMatch.none(),
          embedding: match?.$2,
          overlap: t.overlap,
        ));
      }
      if (saved.length > 1) stats.split++;
      return saved;
    } finally {
      stats.last = watch.elapsed;
    }
  }

  /// The whole stretch as one line (no change of speaker).
  List<SegmentView> _whole(Float32List samples, DateTime startedAt, {bool overlap = false}) {
    final text = _asr.transcribe(samples, _sr).trim();
    if (!_isUsable(text)) {
      stats.dropped++;
      return const [];
    }
    final seconds = samples.length / _sr;
    Float32List? embedding;
    var match = const SpeakerMatch.none();
    if (seconds >= identifier.config.minEmbeddingSeconds) {
      final cap = math.min(samples.length, (config.maxEmbedSeconds * _sr).round());
      embedding = _embedder.embed(Float32List.sublistView(samples, 0, cap), _sr);
      match = identifier.identify(embedding, seconds: seconds);
      if (match.cluster != null) _speakers.saveCluster(match.cluster!);
    }
    return [
      _save(text: text, startedAt: startedAt, samples: samples, match: match, embedding: embedding, overlap: overlap),
    ];
  }

  SegmentView _save({
    required String text,
    required DateTime startedAt,
    required Float32List samples,
    required SpeakerMatch match,
    required Float32List? embedding,
    required bool overlap,
  }) {
    stats.saved++;
    final saved = _transcripts.addSegment(
      text: text,
      startedAt: startedAt,
      duration: Duration(microseconds: (samples.length / _sr * 1e6).round()),
      speakerId: match.speakerId,
      clusterId: match.cluster?.id,
      score: match.score,
      embedding: embedding,
      overlap: overlap,
    );
    try {
      onSaved?.call(saved, samples);
    } on Object catch (e) {
      Log.w('processor', 'after-save hook failed', e);
    }
    return saved;
  }

  /// Audio where only [slot] talks, joined, up to [ProcessorConfig.maxEmbedSeconds].
  Float32List _soloAudio(Float32List samples, SpeakerActivity activity, int slot) {
    final cap = (config.maxEmbedSeconds * _sr).round();
    final pieces = <(int, int)>[];
    var total = 0;
    for (final (from, to) in SpeakerTurns.soloRanges(activity, slot, sampleRate: _sr)) {
      final a = from.clamp(0, samples.length);
      final e = math.min(to.clamp(a, samples.length), a + cap - total);
      if (e <= a) continue;
      pieces.add((a, e));
      total += e - a;
      if (total >= cap) break;
    }
    final out = Float32List(total);
    var at = 0;
    for (final (a, e) in pieces) {
      out.setRange(at, at + e - a, samples, a);
      at += e - a;
    }
    return out;
  }

  void dispose() {
    _asr.dispose();
    _embedder.dispose();
    diarizer?.dispose();
  }

  bool _isUsable(String text) {
    if (text.length < config.minTextChars) return false;
    return RegExp(r'[\p{L}\p{N}]', unicode: true).hasMatch(text);
  }
}
