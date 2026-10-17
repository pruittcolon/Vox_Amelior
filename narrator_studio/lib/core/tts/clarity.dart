/// Clarity: rebuilds the top octave of 24 kHz voices, so Kokoro and Pocket TTS
/// sound as crisp as 48 kHz audio.
///
/// The model is LavaSR (Apache-2.0), a Vocos-style bandwidth extension: from
/// the log-mel spectrum of the voice it predicts the full 48 kHz spectrum. We
/// keep the real voice up to where its own spectrum ends and take the
/// predicted spectrum above it, so the words themselves are never touched;
/// only the missing "air" is added. tool/lab/voicelab.py holds the same
/// pipeline in numpy (enhance_v2), checked against the reference implementation.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import '../render/spectral.dart';
import 'ort.dart';

class Clarity {
  Clarity(String dir, {int threads = 2})
      : _backbone = Ort.instance.open(p.join(dir, backboneFile), threads: threads),
        _head = Ort.instance.open(p.join(dir, headFile), threads: threads) {
    final outs = _head.outputs;
    // The head returns the real and imaginary parts; find which is which.
    _imagFirst = outs.length == 2 && outs[0].toLowerCase().contains('imag');
  }

  static const backboneFile = 'enhancer_backbone.onnx';
  static const headFile = 'enhancer_spec_head.onnx';
  static const files = [backboneFile, '$backboneFile.data', headFile, '$headFile.data'];
  static const rate = 48000;

  static bool installedIn(String dir) => dir.isNotEmpty && files.every((f) => File(p.join(dir, f)).existsSync());

  final OrtSession _backbone;
  final OrtSession _head;
  late final bool _imagFirst;
  final _stft = Stft(nFft: 2048, hop: 512);
  // The model's mel features are built for 44.1 kHz and stop at 8 kHz
  // (as it was trained), even though the audio is 48 kHz.
  final _mel = MelFilters(sampleRate: 44100, nFft: 2048, bands: 80, fMax: 8000);

  /// [x] is 48 kHz audio from a 24 kHz voice. Only a voice whose spectrum
  /// stops short is touched (Pocket TTS falls off a cliff at about 7.8 kHz;
  /// Kokoro reaches the 12 kHz limit and is left alone). The crossover ends at
  /// the cliff; in the crossfade, magnitudes blend under the voice's own
  /// phase, so the two never cancel into a notch, and the added band is
  /// level-matched, frame by frame, to the real voice just below it.
  Float32List enhance(Float32List x, {double width = 800, double skipAbove = 10500}) {
    if (x.length < _stft.nFft) return x;
    final spec = _stft.forward(x);
    final edge = voiceEdge(spec);
    if (edge >= skipAbove) return x; // the voice has its top end already
    final frames = spec.frames;
    final bins = spec.bins;
    final binHz = rate / _stft.nFft;
    final lo = edge - width;
    final features = _mel.logMel(spec);
    final input = _backbone.tensor(features, [1, _mel.bands, frames]);
    List<OrtTensor> hidden = const [];
    List<OrtTensor> parts = const [];
    try {
      hidden = _backbone.run([input]);
      parts = _head.run([hidden.first]);
      final re = parts[_imagFirst ? 1 : 0];
      final im = parts[_imagFirst ? 0 : 1];
      // Prediction is [1, bins, frames'] with frames' >= frames.
      final pBins = re.shape[re.shape.length - 2];
      final pFrames = re.shape.last;
      if (pBins != bins || pFrames < frames) {
        throw StateError('Clarity: unexpected prediction shape ${re.shape} for $frames frames');
      }
      final pr = re.data, pi = im.data;
      // Level match: the real voice vs the prediction just below the crossover.
      final raw = Float64List(frames);
      for (var t = 0; t < frames; t++) {
        var es = 0.0, ep = 0.0;
        for (var k = 0; k < bins; k++) {
          final f = k * binHz;
          if (f < lo - 2000 || f >= lo) continue;
          final i = k * frames + t, j = k * pFrames + t;
          es += spec.re[i] * spec.re[i] + spec.im[i] * spec.im[i];
          ep += pr[j] * pr[j] + pi[j] * pi[j];
        }
        raw[t] = ep > 1e-12 ? math.sqrt(es / ep).clamp(0.5, 2.0) : 1.0;
      }
      final gain = Float64List(frames); // ~50 ms moving average
      for (var t = 0; t < frames; t++) {
        var acc = 0.0;
        for (var d = -2; d <= 2; d++) {
          if (t + d >= 0 && t + d < frames) acc += raw[t + d];
        }
        gain[t] = acc / 5;
      }
      for (var k = 0; k < bins; k++) {
        final tt = ((k * binHz - lo) / width).clamp(0.0, 1.0);
        final m = tt * tt * (3 - 2 * tt);
        if (m <= 0) continue;
        for (var t = 0; t < frames; t++) {
          final i = k * frames + t, j = k * pFrames + t;
          final g = gain[t];
          if (m >= 1) {
            spec.re[i] = pr[j] * g;
            spec.im[i] = pi[j] * g;
            continue;
          }
          final sr = spec.re[i], si = spec.im[i];
          final ms = math.sqrt(sr * sr + si * si);
          final mp = math.sqrt(pr[j] * pr[j] + pi[j] * pi[j]);
          final mag = (1 - m) * ms + m * g * mp;
          if (ms > 1e-9) {
            spec.re[i] = mag * sr / ms;
            spec.im[i] = mag * si / ms;
          } else if (mp > 1e-9) {
            spec.re[i] = mag * pr[j] / mp;
            spec.im[i] = mag * pi[j] / mp;
          } else {
            spec.re[i] = 0;
            spec.im[i] = 0;
          }
        }
      }
    } finally {
      input.free();
      for (final t in hidden) {
        t.free();
      }
      for (final t in parts) {
        t.free();
      }
    }
    final y = _stft.inverse(spec, x.length);
    // The added top octave can push a peak past full scale; keep headroom.
    var peak = 0.0;
    for (final v in y) {
      final a = v.abs();
      if (a > peak) peak = a;
    }
    if (peak > 0.99) {
      final g = 0.99 / peak;
      for (var i = 0; i < y.length; i++) {
        y[i] *= g;
      }
    }
    return y;
  }

  void free() {
    _backbone.free();
    _head.free();
  }
}

/// Where the voice's own spectrum falls off a cliff (Hz): the largest drop in
/// its average log spectrum over 1 kHz, searched between 5 and 11.8 kHz. A
/// voice that only falls away at the 12 kHz limit of 24 kHz audio returns
/// 12000 (it has its top end; nothing to rebuild).
double voiceEdge(Spectrogram s, {double minDropDb = 15}) {
  final binHz = Clarity.rate / ((s.bins - 1) * 2);
  final power = Float64List(s.bins);
  for (var k = 0; k < s.bins; k++) {
    var acc = 0.0;
    for (var t = 0; t < s.frames; t++) {
      final i = k * s.frames + t;
      acc += s.re[i] * s.re[i] + s.im[i] * s.im[i];
    }
    power[k] = acc / s.frames;
  }
  final level = Float64List(s.bins);
  for (var k = 0; k < s.bins; k++) {
    var acc = 0.0;
    for (var d = -2; d <= 2; d++) {
      if (k + d >= 0 && k + d < s.bins) acc += power[k + d];
    }
    level[k] = 10 * math.log(acc / 5 + 1e-20) / math.ln10;
  }
  double mean(int from, int to) {
    var acc = 0.0;
    for (var k = from; k <= to; k++) {
      acc += level[k];
    }
    return acc / (to - from + 1);
  }

  final kLo = (5000 / binHz).round(), kHi = (11800 / binHz).round();
  final w0 = (100 / binHz).round(), w1 = (600 / binHz).round();
  var best = 0.0, edge = 12000.0;
  for (var k = kLo; k <= kHi && k + w1 < s.bins; k++) {
    final drop = mean(k - w1, k - w0) - mean(k + w0, k + w1);
    if (drop > best) {
      best = drop;
      edge = k * binHz;
    }
  }
  return best >= minDropDb ? edge : 12000;
}

/// Energy above [hz] relative to below it, in dB (a quick "how crisp" number).
double airDb(Float32List x, int sampleRate, {double hz = 8000}) {
  final stft = Stft(nFft: 1024, hop: 256);
  if (x.length < stft.nFft) return -120;
  final s = stft.forward(x);
  var lo = 0.0, hi = 0.0;
  for (var k = 0; k < s.bins; k++) {
    final f = k * sampleRate / stft.nFft;
    for (var t = 0; t < s.frames; t++) {
      final i = k * s.frames + t;
      final e = s.re[i] * s.re[i] + s.im[i] * s.im[i];
      if (f < hz) {
        lo += e;
      } else {
        hi += e;
      }
    }
  }
  return 10 * math.log(math.max(hi, 1e-20) / math.max(lo, 1e-20)) / math.ln10;
}
