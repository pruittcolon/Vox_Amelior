import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';

/// A finished stretch of speech with its wall-clock start time.
class CapturedSpeech {
  const CapturedSpeech(this.samples, this.startedAt);

  final Float32List samples;
  final DateTime startedAt;
}

/// Microphone PCM → speech detection → timestamped speech chunks.
///
/// Cheap enough to run continuously; the expensive work (transcription)
/// happens later from a queue.
class SpeechCapture {
  SpeechCapture(this._vad, {this.clock = systemClock, this.sampleRate = 16000, this.maxDrift = const Duration(seconds: 2)});

  final VadEngine _vad;
  final Clock clock;
  final int sampleRate;

  /// Re-anchor timestamps when the audio clock and wall clock disagree by
  /// more than this (e.g. after the microphone dropped audio).
  final Duration maxDrift;

  DateTime? _origin;
  int _received = 0;
  int _pendingByte = -1;

  /// Feeds raw little-endian 16-bit PCM (what the recorder produces).
  List<CapturedSpeech> addPcm16(Uint8List bytes) {
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

  List<CapturedSpeech> addSamples(Float32List samples) {
    if (samples.isEmpty) return const [];
    final now = clock();
    _origin ??= now.subtract(_seconds(samples.length / sampleRate));
    _received += samples.length;
    final expectedNow = _origin!.add(_seconds(_received / sampleRate));
    final drift = now.difference(expectedNow);
    if (drift.abs() > maxDrift) _origin = _origin!.add(drift);
    _vad.accept(samples);
    return _stamp(_vad.takeSegments());
  }

  /// Ends the current stream (e.g. listening paused) and starts fresh.
  List<CapturedSpeech> flush() {
    final result = _stamp(_vad.flush());
    reset();
    return result;
  }

  void reset() {
    _vad.reset();
    _origin = null;
    _received = 0;
    _pendingByte = -1;
  }

  void dispose() => _vad.dispose();

  List<CapturedSpeech> _stamp(List<SpeechChunk> chunks) => [
        for (final c in chunks) CapturedSpeech(c.samples, (_origin ?? clock()).add(_seconds(c.startSeconds))),
      ];

  static Duration _seconds(double s) => Duration(microseconds: (s * 1e6).round());
}
