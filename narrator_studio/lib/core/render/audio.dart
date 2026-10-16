/// Small audio toolbox: resampling, trimming, loudness, fades, int16 bytes.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Everything is mixed and exported at this rate (Kokoro and Pocket TTS speak
/// at 24 kHz natively; Supertonic's 44.1 kHz is resampled).
const outputRate = 24000;

class Pcm {
  const Pcm(this.samples, this.sampleRate);
  final Float32List samples;
  final int sampleRate;

  double get seconds => samples.length / sampleRate;
}

/// Band-limited resampling with a windowed-sinc kernel (alias-free when going
/// down from 44.1 kHz). Uses a precomputed polyphase table, so it is cheap
/// enough to run on every sentence.
Float32List resample(Float32List x, int from, int to) {
  if (from == to || x.isEmpty) return x;
  final g = _gcd(from, to);
  final up = to ~/ g; // output steps per cycle
  final down = from ~/ g; // input steps per cycle
  final outLen = (x.length * to / from).floor();
  final out = Float32List(outLen);
  final cutoff = math.min(1.0, to / from) * 0.95;
  const halfWidth = 12;
  final reach = (halfWidth / cutoff).ceil();
  final taps = 2 * reach + 1;
  final table = _kernels.putIfAbsent('$from>$to', () {
    final t = Float32List(up * taps);
    for (var phase = 0; phase < up; phase++) {
      // Output sample (cycle*up + phase) sits at input position ... + phase*down/up.
      final frac = (phase * down) / up - ((phase * down) ~/ up);
      var norm = 0.0;
      for (var k = 0; k < taps; k++) {
        final d = (k - reach - frac) * cutoff;
        final sinc = d == 0 ? 1.0 : math.sin(math.pi * d) / (math.pi * d);
        final w = d.abs() >= halfWidth ? 0.0 : 0.5 + 0.5 * math.cos(math.pi * d / halfWidth);
        t[phase * taps + k] = sinc * w;
        norm += sinc * w;
      }
      for (var k = 0; k < taps; k++) {
        t[phase * taps + k] /= norm;
      }
    }
    return t;
  });
  for (var i = 0; i < outLen; i++) {
    final phase = i % up;
    final base = (i ~/ up) * down + (phase * down) ~/ up - reach;
    final o = phase * taps;
    var acc = 0.0;
    if (base >= 0 && base + taps <= x.length) {
      for (var k = 0; k < taps; k++) {
        acc += x[base + k] * table[o + k];
      }
    } else {
      for (var k = 0; k < taps; k++) {
        final j = base + k;
        if (j >= 0 && j < x.length) acc += x[j] * table[o + k];
      }
    }
    out[i] = acc;
  }
  return out;
}

final _kernels = <String, Float32List>{};

int _gcd(int a, int b) => b == 0 ? a : _gcd(b, a % b);

/// Removes leading/trailing near-silence but keeps a short natural margin.
Float32List trimSilence(Float32List x, int sampleRate, {double threshold = 0.012, int keepMs = 40}) {
  if (x.isEmpty) return x;
  final win = math.max(1, sampleRate ~/ 100); // 10 ms
  int firstLoud = -1, lastLoud = -1;
  for (var i = 0; i < x.length; i += win) {
    final end = math.min(x.length, i + win);
    var peak = 0.0;
    for (var j = i; j < end; j++) {
      final a = x[j].abs();
      if (a > peak) peak = a;
    }
    if (peak > threshold) {
      if (firstLoud < 0) firstLoud = i;
      lastLoud = end;
    }
  }
  if (firstLoud < 0) return Float32List(0);
  final keep = sampleRate * keepMs ~/ 1000;
  final start = math.max(0, firstLoud - keep);
  final stop = math.min(x.length, lastLoud + keep);
  return Float32List.sublistView(x, start, stop);
}

/// RMS of the louder parts (ignores pauses), for loudness matching.
double speechRms(Float32List x, int sampleRate) {
  final win = math.max(1, sampleRate ~/ 50); // 20 ms
  final energies = <double>[];
  for (var i = 0; i + win <= x.length; i += win) {
    var e = 0.0;
    for (var j = i; j < i + win; j++) {
      e += x[j] * x[j];
    }
    energies.add(e / win);
  }
  if (energies.isEmpty) return 0;
  energies.sort();
  // Average of the top 60% of frames: voiced speech.
  final from = (energies.length * 0.4).floor();
  var sum = 0.0;
  for (var i = from; i < energies.length; i++) {
    sum += energies[i];
  }
  return math.sqrt(sum / (energies.length - from));
}

/// Brings speech to a steady loudness so voices and engines match.
/// The default target is about -20 dBFS RMS.
Float32List normalizeSpeech(Float32List x, int sampleRate, {double target = 0.1, double maxGain = 6}) {
  final rms = speechRms(x, sampleRate);
  if (rms < 1e-5) return x;
  var gain = (target / rms).clamp(0.2, maxGain).toDouble();
  var peak = 0.0;
  for (final s in x) {
    final a = s.abs();
    if (a > peak) peak = a;
  }
  if (peak * gain > 0.97) gain = 0.97 / peak;
  final out = Float32List(x.length);
  for (var i = 0; i < x.length; i++) {
    out[i] = x[i] * gain;
  }
  return out;
}

/// Short fade in/out so clips never click.
void fadeEdges(Float32List x, int sampleRate, {int ms = 8}) {
  final n = math.min(x.length ~/ 2, sampleRate * ms ~/ 1000);
  for (var i = 0; i < n; i++) {
    final g = i / n;
    x[i] *= g;
    x[x.length - 1 - i] *= g;
  }
}

Float32List silence(int ms, {int sampleRate = outputRate}) => Float32List(sampleRate * ms ~/ 1000);

/// 16-bit little-endian PCM, clipped.
Uint8List toInt16Bytes(Float32List x) {
  final out = Int16List(x.length);
  for (var i = 0; i < x.length; i++) {
    final v = (x[i] * 32767).round();
    out[i] = v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
  }
  return out.buffer.asUint8List();
}

Float32List fromInt16Bytes(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  final n = bytes.length ~/ 2;
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

Float32List concat(List<Float32List> parts) {
  final total = parts.fold<int>(0, (a, b) => a + b.length);
  final out = Float32List(total);
  var o = 0;
  for (final p in parts) {
    out.setRange(o, o + p.length, p);
    o += p.length;
  }
  return out;
}
