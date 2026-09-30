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
    this.chunkSeconds = 30,
    this.chunkPause = const Duration(seconds: 3),
    this.minPartWords = 2,
  });

  final int sampleRate;

  /// Transcripts shorter than this are noise ("uh", empty results) and dropped.
  final int minTextChars;

  /// Only the first seconds of long utterances are used for the voiceprint.
  final double maxEmbedSeconds;

  /// Shorter stretches are too short to hold a change of speaker.
  final double minDiarizeSeconds;

  /// Speaker changes are found per chunk of recent lines: a chunk ends after
  /// this much speech, or at a pause in the talk of [chunkPause].
  final int chunkSeconds;
  final Duration chunkPause;

  /// A line is only cut when every part has at least this many words (and
  /// lasts at least [SpeakerTurns.minTurnSeconds]); otherwise it stays whole.
  final int minPartWords;
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

/// The expensive step, in two stages.
///
/// 1. [transcribe]: as soon as a sentence ends, the whole stretch is
///    transcribed once, matched to a voice, saved and shown.
/// 2. [refine]: per chunk of recent lines (up to [ProcessorConfig.chunkSeconds]
///    of speech, or until a pause), the [diarizer] finds who spoke when; a line
///    where the speaker changes is cut into one line per person, and people
///    talking at the same time are marked.
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
    this.onReplaced,
  });

  AsrEngine _asr;
  final EmbeddingEngine _embedder;
  final SpeakerIdentifier identifier;
  final TranscriptRepository _transcripts;
  final SpeakerRepository _speakers;
  /// Tunable while running (e.g. from settings).
  ProcessorConfig config;
  final ProcessorStats stats = ProcessorStats();

  /// Splits lines at speaker changes when set (the model may arrive later).
  DiarizationEngine? diarizer;

  /// Settings switch for splitting.
  bool splitSpeakers = true;
  SpeakerTurns turns;

  /// Called with each saved utterance and its audio (e.g. to keep a clip).
  /// Failures here never lose the transcript.
  final void Function(SegmentView segment, Float32List samples)? onSaved;

  /// Called before line [id] is cut into parts (e.g. to drop its old clip).
  final void Function(int id)? onReplaced;

  /// Switches to another speech model (e.g. int8 → fp16). Safe between
  /// transcriptions; the old model is released.
  void swapAsr(AsrEngine next) {
    final old = _asr;
    _asr = next;
    old.dispose();
  }

  int get _sr => config.sampleRate;

  final List<_Line> _open = [];
  final List<List<_Line>> _ready = [];
  DateTime _openEnd = DateTime.fromMillisecondsSinceEpoch(0);

  bool get _splitting => splitSpeakers && diarizer != null;

  /// A chunk that is complete and should be refined now.
  bool get hasReadyChunk => _ready.isNotEmpty;

  /// Lines saved by [transcribe] that [refine] has not finished yet.
  bool get hasPending => _open.isNotEmpty || _ready.isNotEmpty;

  /// Stage 1, as soon as a sentence ends: the whole stretch is transcribed
  /// once and given a first voice match, then saved and returned. The line
  /// is also kept for [refine].
  List<SegmentView> transcribe(Float32List samples, DateTime startedAt) {
    final watch = Stopwatch()..start();
    stats.processed++;
    try {
      final t = _asr.transcribeTimed(samples, _sr);
      final text = t.text.trim();
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
      final saved = _save(text: text, startedAt: startedAt, samples: samples, match: match, embedding: embedding, overlap: false);
      if (_open.isNotEmpty && startedAt.difference(_openEnd) >= config.chunkPause) closeChunk();
      _open.add(_Line(saved, samples, t.words));
      _openEnd = startedAt.add(saved.duration);
      // Without speaker splitting there is nothing to wait for.
      if (!_splitting || _open.fold<int>(0, (a, l) => a + l.samples.length) >= config.chunkSeconds * _sr) closeChunk();
      return [saved];
    } finally {
      stats.last = watch.elapsed;
    }
  }

  /// Ends the chunk being gathered (e.g. listening stopped), so [refine] takes it now.
  void closeChunk() {
    if (_open.isEmpty) return;
    _ready.add(List.of(_open));
    _open.clear();
  }

  /// Whether [refine] has work: a finished chunk, or a pause in the talk.
  bool refineDue(DateTime now) => _ready.isNotEmpty || _openDue(now);

  bool _openDue(DateTime now) => _open.isNotEmpty && now.difference(_openEnd) >= config.chunkPause;

  /// Stage 2, per chunk of recent lines: finds who spoke when across the
  /// whole chunk, and cuts a line where the speaker changes (text cut at word
  /// start times, each part named from that voice's solo audio in the
  /// chunk). Returns the finished lines, changed or not.
  List<SegmentView> refine(DateTime now, {bool readyOnly = false}) {
    if (!readyOnly && _openDue(now)) closeChunk();
    final out = <SegmentView>[];
    while (_ready.isNotEmpty) {
      final chunk = _ready.removeAt(0);
      try {
        out.addAll(_refineChunk(chunk));
      } on Object catch (e, st) {
        Log.e('processor', 'refining a chunk failed; lines kept as they are', e, st);
        out.addAll(_current(chunk));
      }
    }
    return out;
  }

  /// Both stages at once for one stretch (tests, one-off processing).
  List<SegmentView> process(Float32List samples, DateTime startedAt) {
    if (transcribe(samples, startedAt).isEmpty) return const [];
    closeChunk();
    return refine(DateTime.now());
  }

  /// Lines of [chunk] as they are now; lines deleted or re-labelled by hand meanwhile are left out.
  List<SegmentView> _current(List<_Line> chunk) => [
        for (final l in chunk)
          if (_transcripts.segment(l.fast.id) case final v?)
            if (v.speakerId == l.fast.speakerId && v.clusterId == l.fast.clusterId) v,
      ];

  List<SegmentView> _refineChunk(List<_Line> chunk) {
    final lines = [for (final l in chunk) if (_current([l]).isNotEmpty) l];
    final total = lines.fold<int>(0, (a, l) => a + l.samples.length);
    final d = splitSpeakers ? diarizer : null;
    if (d == null || total < config.minDiarizeSeconds * _sr) return _current(lines);
    final audio = Float32List(total);
    final offsets = <int>[];
    var at = 0;
    for (final l in lines) {
      offsets.add(at);
      audio.setRange(at, at + l.samples.length, l.samples);
      at += l.samples.length;
    }
    final SpeakerActivity activity;
    try {
      activity = d.analyze(audio, _sr);
    } on Object catch (e) {
      Log.w('processor', 'diarizer failed; keeping lines as they are', e);
      return _current(lines);
    }
    final all = turns.turns(activity, totalSeconds: total / _sr);

    // Each voice is named once per chunk, from the audio where it talks alone.
    final named = <int, (SpeakerMatch, Float32List)?>{};
    (SpeakerMatch, Float32List)? name(int slot) => named.putIfAbsent(slot, () {
          final solo = _soloAudio(audio, activity, slot);
          final soloSeconds = solo.length / _sr;
          if (soloSeconds < identifier.config.minEmbeddingSeconds) return null;
          final e = _embedder.embed(solo, _sr);
          final m = identifier.identify(e, seconds: soloSeconds);
          if (m.cluster != null) _speakers.saveCluster(m.cluster!);
          return (m, e);
        });
    // A voice too quiet or brief to name is not a different person: only
    // named people (or guests) can cut a line.
    bool isNamed(SpeakerTurn t) => name(t.slot) != null;
    String who(SpeakerTurn t) {
      final m = name(t.slot)!.$1;
      return m.speakerId ?? m.cluster?.id ?? 'slot ${t.slot}';
    }

    final out = <SegmentView>[];
    for (var i = 0; i < lines.length; i++) {
      final l = lines[i];
      final from = offsets[i] / _sr;
      var parts = _within(all, from, from + l.samples.length / _sr);
      if (parts.length > 1) parts = _absorbUnnamed(parts, isNamed);
      if (parts.length > 1) parts = SpeakerTurns.joinSame(parts, who);
      if (parts.length > 1) {
        final cut = _cut(l, parts, name);
        if (cut != null) {
          out.addAll(cut);
          continue;
        }
      } else if (parts.length == 1 && parts.single.overlap && !l.fast.overlap) {
        _transcripts.setOverlap(l.fast.id);
      }
      out.addAll(_current([l]));
    }
    return out;
  }

  /// Chunk turns inside [a, b] seconds, relative to [a]. A sliver at either
  /// edge (a turn that belongs to the neighbouring line) joins its neighbour.
  List<SpeakerTurn> _within(List<SpeakerTurn> all, double a, double b) {
    final out = <SpeakerTurn>[
      for (final t in all)
        if (math.min(t.end, b) > math.max(t.start, a))
          t.copyWith(start: math.max(t.start, a) - a, end: math.min(t.end, b) - a),
    ];
    while (out.length > 1 && out.first.end - out.first.start < turns.minTurnSeconds) {
      out[1] = out[1].copyWith(start: 0);
      out.removeAt(0);
    }
    while (out.length > 1 && out.last.end - out.last.start < turns.minTurnSeconds) {
      out[out.length - 2] = out[out.length - 2].copyWith(end: out.last.end);
      out.removeLast();
    }
    return out;
  }

  /// Merges every part whose voice could not be named into its longer
  /// neighbour, so it can never start a new line on its own.
  List<SpeakerTurn> _absorbUnnamed(List<SpeakerTurn> parts, bool Function(SpeakerTurn) named) {
    final list = List<SpeakerTurn>.of(parts);
    while (list.length > 1) {
      final i = list.indexWhere((t) => !named(t));
      if (i < 0) break;
      final left = i > 0 ? list[i - 1] : null;
      final right = i + 1 < list.length ? list[i + 1] : null;
      final intoLeft = right == null || (left != null && left.duration >= right.duration);
      if (intoLeft) {
        list[i - 1] = left!.copyWith(end: list[i].end, overlap: left.overlap || list[i].overlap);
      } else {
        list[i + 1] = right.copyWith(start: list[i].start, overlap: right.overlap || list[i].overlap);
      }
      list.removeAt(i);
    }
    return list;
  }

  /// Replaces line [l] by one line per part. Null (line kept exactly as it
  /// was saved) unless every part lasts at least [SpeakerTurns.minTurnSeconds]
  /// and has at least [ProcessorConfig.minPartWords] words.
  List<SegmentView>? _cut(_Line l, List<SpeakerTurn> parts, (SpeakerMatch, Float32List)? Function(int) name) {
    if (parts.any((t) => t.duration < turns.minTurnSeconds)) return null;
    final texts = _texts(l, parts);
    for (final t in texts) {
      if (!_isUsable(t) || t.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).length < config.minPartWords) return null;
    }
    try {
      onReplaced?.call(l.fast.id);
    } on Object catch (e) {
      Log.w('processor', 'replace hook failed', e);
    }
    final saved = <SegmentView>[];
    for (var i = 0; i < parts.length; i++) {
      final t = parts[i];
      final from = (t.start * _sr).round().clamp(0, l.samples.length);
      final to = (t.end * _sr).round().clamp(from, l.samples.length);
      final slice = Float32List.sublistView(l.samples, from, to);
      final m = name(t.slot);
      final match = m?.$1 ?? const SpeakerMatch.none();
      if (saved.isEmpty) {
        final v = _transcripts.updateSegment(
          l.fast.id,
          text: texts[i],
          duration: Duration(microseconds: (slice.length / _sr * 1e6).round()),
          speakerId: match.speakerId,
          clusterId: match.cluster?.id,
          score: match.score,
          embedding: m?.$2,
          overlap: t.overlap,
        );
        saved.add(v);
        _afterSave(v, slice);
      } else {
        saved.add(_save(
          text: texts[i],
          startedAt: l.fast.startedAt.add(Duration(microseconds: (t.start * 1e6).round())),
          samples: slice,
          match: match,
          embedding: m?.$2,
          overlap: t.overlap,
        ));
      }
    }
    stats.split++;
    return saved;
  }

  /// Text of each part: words by their start time, or (when the engine gave
  /// no times) each part transcribed on its own.
  List<String> _texts(_Line l, List<SpeakerTurn> parts) {
    if (l.words.isEmpty) {
      return [
        for (final t in parts)
          _asr
              .transcribe(
                Float32List.sublistView(
                  l.samples,
                  (t.start * _sr).round().clamp(0, l.samples.length),
                  (t.end * _sr).round().clamp(0, l.samples.length),
                ),
                _sr,
              )
              .trim(),
      ];
    }
    final buckets = [for (final _ in parts) <String>[]];
    for (final w in l.words) {
      var i = parts.indexWhere((t) => w.start < t.end);
      if (i < 0) i = parts.length - 1;
      buckets[i].add(w.text);
    }
    return [for (final b in buckets) b.join(' ')];
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
    _afterSave(saved, samples);
    return saved;
  }

  void _afterSave(SegmentView saved, Float32List samples) {
    try {
      onSaved?.call(saved, samples);
    } on Object catch (e) {
      Log.w('processor', 'after-save hook failed', e);
    }
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

class _Line {
  _Line(this.fast, this.samples, this.words);

  /// The line as saved by stage 1.
  final SegmentView fast;
  final Float32List samples;
  final List<TimedWord> words;
}
