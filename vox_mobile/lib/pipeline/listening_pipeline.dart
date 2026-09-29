import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

class PipelineConfig {
  const PipelineConfig({
    this.sampleRate = 16000,
    this.minTextChars = 2,
    this.maxEmbedSeconds = 20,
  });

  final int sampleRate;

  /// Transcripts shorter than this are noise ("uh", empty results) and dropped.
  final int minTextChars;

  /// Only the first seconds of long utterances are used for the voiceprint.
  final double maxEmbedSeconds;
}

class PipelineStats {
  int chunksSeen = 0;
  int segmentsSaved = 0;
  int dropped = 0;
  int errors = 0;
  Duration lastProcessing = Duration.zero;
}

/// Audio in, speaker-labelled transcript out:
/// microphone PCM → speech detection → Parakeet ASR → voiceprint →
/// speaker match → database.
///
/// Fully synchronous around native calls and free of Flutter dependencies, so
/// it runs unchanged inside the background service isolate and in unit tests.
class ListeningPipeline {
  ListeningPipeline({
    required this._vad,
    required this._asr,
    required this._embedder,
    required this._identifier,
    required this._transcripts,
    required this._speakers,
    this.clock = systemClock,
    this.config = const PipelineConfig(),
    this.onSegment,
  });

  final VadEngine _vad;
  final AsrEngine _asr;
  final EmbeddingEngine _embedder;
  final SpeakerIdentifier _identifier;
  final TranscriptRepository _transcripts;
  final SpeakerRepository _speakers;
  final Clock clock;
  final PipelineConfig config;

  /// Called for every saved utterance (used to trigger automations).
  final void Function(SegmentView segment)? onSegment;

  final PipelineStats stats = PipelineStats();

  DateTime? _origin;
  int _pendingByte = -1;

  SpeakerIdentifier get identifier => _identifier;

  /// Feeds raw little-endian 16-bit PCM (what the recorder produces).
  List<SegmentView> addPcm16(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    final total = bytes.length + (_pendingByte >= 0 ? 1 : 0);
    final sampleCount = total ~/ 2;
    final out = Float32List(sampleCount);
    var byteIndex = 0;
    for (var i = 0; i < sampleCount; i++) {
      int lo;
      if (i == 0 && _pendingByte >= 0) {
        lo = _pendingByte;
        _pendingByte = -1;
      } else {
        lo = data.getUint8(byteIndex++);
      }
      final hi = data.getInt8(byteIndex++);
      out[i] = ((hi << 8) | lo) / 32768.0;
    }
    if (byteIndex < bytes.length) _pendingByte = data.getUint8(byteIndex);
    return addSamples(out);
  }

  List<SegmentView> addSamples(Float32List samples) {
    if (samples.isEmpty) return const [];
    _origin ??= clock().subtract(_secondsToDuration(samples.length / config.sampleRate));
    _vad.accept(samples);
    return _process(_vad.takeSegments());
  }

  /// Finishes the current stream (e.g. listening paused) and starts fresh.
  List<SegmentView> flush() {
    final result = _process(_vad.flush());
    reset();
    return result;
  }

  void reset() {
    _vad.reset();
    _origin = null;
    _pendingByte = -1;
  }

  void dispose() {
    _vad.dispose();
    _asr.dispose();
    _embedder.dispose();
  }

  List<SegmentView> _process(List<SpeechChunk> chunks) {
    final saved = <SegmentView>[];
    for (final chunk in chunks) {
      stats.chunksSeen++;
      final watch = Stopwatch()..start();
      try {
        final segment = _handleChunk(chunk);
        if (segment != null) {
          saved.add(segment);
          stats.segmentsSaved++;
          onSegment?.call(segment);
        } else {
          stats.dropped++;
        }
      } catch (e, st) {
        stats.errors++;
        Log.e('pipeline', 'failed to process speech chunk', e, st);
      }
      stats.lastProcessing = watch.elapsed;
    }
    return saved;
  }

  SegmentView? _handleChunk(SpeechChunk chunk) {
    final text = _asr.transcribe(chunk.samples, config.sampleRate).trim();
    if (!_isUsable(text)) return null;

    final seconds = chunk.duration(config.sampleRate);
    Float32List? embedding;
    var match = const SpeakerMatch.none();
    if (seconds >= _identifier.config.minEmbeddingSeconds) {
      final cap = math.min(chunk.samples.length, (config.maxEmbedSeconds * config.sampleRate).round());
      embedding = _embedder.embed(Float32List.sublistView(chunk.samples, 0, cap), config.sampleRate);
      match = _identifier.identify(embedding, seconds: seconds);
      if (match.cluster != null) _speakers.saveCluster(match.cluster!);
    }

    return _transcripts.addSegment(
      text: text,
      startedAt: (_origin ?? clock()).add(_secondsToDuration(chunk.startSeconds)),
      duration: _secondsToDuration(seconds),
      speakerId: match.speakerId,
      clusterId: match.cluster?.id,
      score: match.score,
      embedding: embedding,
    );
  }

  bool _isUsable(String text) {
    if (text.length < config.minTextChars) return false;
    // Pure punctuation / filler artefacts.
    return RegExp(r'[\p{L}\p{N}]', unicode: true).hasMatch(text);
  }

  Duration _secondsToDuration(double s) => Duration(microseconds: (s * 1e6).round());
}
