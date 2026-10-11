import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';

/// A finished stretch of speech with its wall-clock start time.
class CapturedSpeech {
  const CapturedSpeech(this.samples, this.startedAt);

  final Float32List samples;
  final DateTime startedAt;
}

/// How loud a stretch of audio is (0–1 samples, as the speech detector hears it).
class AudioLevel {
  const AudioLevel(this.rms, this.peak);

  static const AudioLevel silent = AudioLevel(0, 0);

  final double rms;
  final double peak;

  static AudioLevel of(Float32List samples) {
    if (samples.isEmpty) return silent;
    var sum = 0.0;
    var peak = 0.0;
    for (final v in samples) {
      final a = v.abs();
      sum += v * v;
      if (a > peak) peak = a;
    }
    return AudioLevel(math.sqrt(sum / samples.length), peak);
  }

  /// Loudness on a 0–1 meter: -60 dBFS is empty, 0 dBFS is full.
  static double meter(double amplitude) {
    if (amplitude <= 0) return 0;
    final db = 20 * math.log(amplitude) / math.ln10;
    return ((db + 60) / 60).clamp(0.0, 1.0);
  }

  /// Full-scale audio: the boost (or the phone) is clipping.
  bool get clipping => peak >= 0.999;
}

/// Microphone PCM → speech detection → timestamped speech chunks.
///
/// Cheap enough to run continuously; the expensive work (transcription)
/// happens later from a queue.
class SpeechCapture {
  SpeechCapture(
    this._vad, {
    this.clock = systemClock,
    this.sampleRate = 16000,
    this.maxDrift = const Duration(seconds: 2),
    this.gain = 1.0,
  });

  VadEngine _vad;
  final Clock clock;
  final int sampleRate;

  /// Re-anchor timestamps when the audio clock and wall clock disagree by
  /// more than this (e.g. after the microphone dropped audio).
  final Duration maxDrift;

  /// Boost applied to what the speech detector (and level meter) hears
  /// (1.0 = as recorded). Changes apply to the next audio.
  ///
  /// Transcription and voice matching always get the audio as recorded: both
  /// models normalise loudness themselves, so a boost only adds clipping
  /// distortion there (which made transcripts noticeably worse).
  double gain;

  /// How much recorded audio is kept to hand out unboosted speech. A chunk
  /// ends at most maxSpeech (20 s) plus the closing pause after it began.
  static const double keepSeconds = 40;
  late final Float32List _raw = Float32List((keepSeconds * sampleRate).round());

  /// Stream position just after the last boosted sample (0 = none boosted).
  int _boostedEnd = 0;

  /// Measure loudness (for a level meter). Off by default: costs a pass over every sample.
  bool measureLevel = false;
  AudioLevel? _loudest;

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
    _keep(samples);
    final heard = gain == 1.0 ? samples : _boost(samples, gain);
    if (gain != 1.0) _boostedEnd = _received;
    if (measureLevel) {
      final l = AudioLevel.of(heard);
      final m = _loudest;
      _loudest = m == null ? l : AudioLevel(math.max(m.rms, l.rms), math.max(m.peak, l.peak));
    }
    _vad.accept(heard);
    return _stamp(_vad.takeSegments());
  }

  /// Remembers the recorded audio; position p lives at p % _raw.length.
  void _keep(Float32List samples) {
    final n = _raw.length;
    final from = samples.length > n ? samples.length - n : 0;
    var at = (_received - samples.length + from) % n;
    for (var i = from; i < samples.length; i++) {
      _raw[at] = samples[i];
      if (++at == n) at = 0;
    }
  }

  /// [c]'s audio as recorded, without the boost. Falls back to what the
  /// detector returned if that audio is no longer (or was never) kept.
  Float32List _unboosted(SpeechChunk c) {
    final start = (c.startSeconds * sampleRate).round();
    final end = start + c.samples.length;
    if (start >= _boostedEnd) return c.samples; // none of it was boosted
    if (start < 0 || end > _received || _received - start > _raw.length) return c.samples;
    final out = Float32List(c.samples.length);
    final n = _raw.length;
    var at = start % n;
    for (var i = 0; i < out.length; i++) {
      out[i] = _raw[at];
      if (++at == n) at = 0;
    }
    return out;
  }

  static Float32List _boost(Float32List samples, double gain) {
    final out = Float32List(samples.length);
    for (var i = 0; i < samples.length; i++) {
      out[i] = (samples[i] * gain).clamp(-1.0, 1.0);
    }
    return out;
  }

  /// The loudest audio since the last call (so a short clip between two
  /// readings is not missed), or silence when nothing was measured.
  AudioLevel takeLevel() {
    final l = _loudest ?? AudioLevel.silent;
    _loudest = null;
    return l;
  }

  /// Puts a new speech detector in place (its settings changed). Speech
  /// heard so far is returned first, then detection starts fresh.
  List<CapturedSpeech> swapVad(VadEngine next) {
    final result = _stamp(_vad.flush());
    _vad.dispose();
    _vad = next;
    // The new detector counts from zero, so timestamps re-anchor too. A
    // half-received sample byte belongs to the microphone stream and is kept,
    // or every later sample would be misaligned.
    _origin = null;
    _received = 0;
    _boostedEnd = 0;
    return result;
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
    _boostedEnd = 0;
    _pendingByte = -1;
  }

  void dispose() => _vad.dispose();

  List<CapturedSpeech> _stamp(List<SpeechChunk> chunks) => [
        for (final c in chunks) CapturedSpeech(_unboosted(c), (_origin ?? clock()).add(_seconds(c.startSeconds))),
      ];

  static Duration _seconds(double s) => Duration(microseconds: (s * 1e6).round());
}
