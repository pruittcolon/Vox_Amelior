import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

class ProcessorConfig {
  const ProcessorConfig({this.sampleRate = 16000, this.minTextChars = 2, this.maxEmbedSeconds = 20});

  final int sampleRate;

  /// Transcripts shorter than this are noise ("uh", empty results) and dropped.
  final int minTextChars;

  /// Only the first seconds of long utterances are used for the voiceprint.
  final double maxEmbedSeconds;
}

class ProcessorStats {
  int processed = 0;
  int saved = 0;
  int dropped = 0;
  int errors = 0;
  Duration last = Duration.zero;
}

/// The expensive step: one speech chunk → text → speaker → saved utterance.
class SegmentProcessor {
  SegmentProcessor({
    required this._asr,
    required this._embedder,
    required this.identifier,
    required this._transcripts,
    required this._speakers,
    this.config = const ProcessorConfig(),
    this.onSaved,
  });

  final AsrEngine _asr;
  final EmbeddingEngine _embedder;
  final SpeakerIdentifier identifier;
  final TranscriptRepository _transcripts;
  final SpeakerRepository _speakers;
  final ProcessorConfig config;
  final ProcessorStats stats = ProcessorStats();

  /// Called with each saved utterance and its audio (e.g. to keep a clip).
  /// Failures here never lose the transcript.
  final void Function(SegmentView segment, Float32List samples)? onSaved;

  /// Returns the saved utterance, or null if the audio held no usable words.
  SegmentView? process(Float32List samples, DateTime startedAt) {
    final watch = Stopwatch()..start();
    stats.processed++;
    try {
      final text = _asr.transcribe(samples, config.sampleRate).trim();
      if (!_isUsable(text)) {
        stats.dropped++;
        return null;
      }
      final seconds = samples.length / config.sampleRate;
      Float32List? embedding;
      var match = const SpeakerMatch.none();
      if (seconds >= identifier.config.minEmbeddingSeconds) {
        final cap = math.min(samples.length, (config.maxEmbedSeconds * config.sampleRate).round());
        embedding = _embedder.embed(Float32List.sublistView(samples, 0, cap), config.sampleRate);
        match = identifier.identify(embedding, seconds: seconds);
        if (match.cluster != null) _speakers.saveCluster(match.cluster!);
      }
      stats.saved++;
      final saved = _transcripts.addSegment(
        text: text,
        startedAt: startedAt,
        duration: Duration(microseconds: (seconds * 1e6).round()),
        speakerId: match.speakerId,
        clusterId: match.cluster?.id,
        score: match.score,
        embedding: embedding,
      );
      try {
        onSaved?.call(saved, samples);
      } on Object catch (e) {
        Log.w('processor', 'after-save hook failed', e);
      }
      return saved;
    } finally {
      stats.last = watch.elapsed;
    }
  }

  void dispose() {
    _asr.dispose();
    _embedder.dispose();
  }

  bool _isUsable(String text) {
    if (text.length < config.minTextChars) return false;
    return RegExp(r'[\p{L}\p{N}]', unicode: true).hasMatch(text);
  }
}
