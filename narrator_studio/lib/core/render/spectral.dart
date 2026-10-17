/// Short-time Fourier transforms and mel features, for the Clarity upscaler.
///
/// Matches the conventions of Vocos (and so LavaSR): a periodic Hann window,
/// "same" padding (reflect by (n_fft - hop) / 2 on both sides), numpy's
/// rfft/irfft scaling, and Slaney-style mel filters.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// In-place radix-2 complex FFT of a fixed power-of-two size.
class Fft {
  Fft(this.n) : assert(n > 1 && (n & (n - 1)) == 0) {
    var bits = 0;
    while ((1 << bits) < n) {
      bits++;
    }
    _rev = Int32List(n);
    for (var i = 0; i < n; i++) {
      var r = 0;
      for (var b = 0; b < bits; b++) {
        r |= ((i >> b) & 1) << (bits - 1 - b);
      }
      _rev[i] = r;
    }
    _cos = Float64List(n ~/ 2);
    _sin = Float64List(n ~/ 2);
    for (var i = 0; i < n ~/ 2; i++) {
      _cos[i] = math.cos(2 * math.pi * i / n);
      _sin[i] = math.sin(2 * math.pi * i / n);
    }
  }

  final int n;
  late final Int32List _rev;
  late final Float64List _cos;
  late final Float64List _sin;

  /// Forward: X[k] = sum x[j] e^(-2 pi i jk/n). Inverse divides by n.
  void transform(Float64List re, Float64List im, {bool inverse = false}) {
    for (var i = 0; i < n; i++) {
      final j = _rev[i];
      if (j > i) {
        final tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        final ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    final sign = inverse ? 1.0 : -1.0;
    for (var size = 2; size <= n; size <<= 1) {
      final half = size >> 1;
      final step = n ~/ size;
      for (var start = 0; start < n; start += size) {
        for (var k = 0; k < half; k++) {
          final wr = _cos[k * step];
          final wi = sign * _sin[k * step];
          final a = start + k;
          final b = a + half;
          final tr = re[b] * wr - im[b] * wi;
          final ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
    if (inverse) {
      final s = 1.0 / n;
      for (var i = 0; i < n; i++) {
        re[i] *= s;
        im[i] *= s;
      }
    }
  }
}

Float64List hannPeriodic(int n) {
  final w = Float64List(n);
  for (var i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / n);
  }
  return w;
}

/// A complex spectrogram, frequency-major: value (f, t) is at [f * frames + t].
class Spectrogram {
  Spectrogram(this.bins, this.frames)
      : re = Float32List(bins * frames),
        im = Float32List(bins * frames);
  Spectrogram.of(this.bins, this.frames, this.re, this.im);

  final int bins;
  final int frames;
  final Float32List re;
  final Float32List im;
}

/// STFT with Vocos "same" padding. Frames are transformed two at a time
/// (one as the real part, one as the imaginary part of a single complex FFT).
class Stft {
  Stft({this.nFft = 2048, this.hop = 512})
      : _fft = Fft(nFft),
        window = hannPeriodic(nFft);

  final int nFft;
  final int hop;
  final Float64List window;
  final Fft _fft;

  int get bins => nFft ~/ 2 + 1;
  int get pad => (nFft - hop) ~/ 2;

  /// Signals shorter than this can't be reflect-padded.
  int get minLength => pad + 1;

  int framesFor(int length) => (length + 2 * pad - nFft) ~/ hop + 1;

  Spectrogram forward(Float32List x) {
    final n = x.length;
    if (n < minLength) throw ArgumentError('signal too short for the STFT');
    final padded = Float32List(n + 2 * pad);
    for (var j = 0; j < pad; j++) {
      padded[j] = x[pad - j];
      padded[pad + n + j] = x[n - 2 - j];
    }
    padded.setAll(pad, x);
    final frames = framesFor(n);
    final out = Spectrogram(bins, frames);
    final re = Float64List(nFft);
    final im = Float64List(nFft);
    for (var t = 0; t < frames; t += 2) {
      final two = t + 1 < frames;
      final a = t * hop;
      final b = (t + 1) * hop;
      for (var j = 0; j < nFft; j++) {
        re[j] = padded[a + j] * window[j];
        im[j] = two ? padded[b + j] * window[j] : 0.0;
      }
      _fft.transform(re, im);
      // Unpack: X[k] = (Z[k] + conj Z[n-k]) / 2, Y[k] = (Z[k] - conj Z[n-k]) / 2i.
      for (var k = 0; k < bins; k++) {
        final m = k == 0 ? 0 : nFft - k;
        final zr = re[k], zi = im[k], cr = re[m], ci = -im[m];
        out.re[k * frames + t] = 0.5 * (zr + cr);
        out.im[k * frames + t] = 0.5 * (zi + ci);
        if (two) {
          out.re[k * frames + t + 1] = 0.5 * (zi - ci);
          out.im[k * frames + t + 1] = -0.5 * (zr - cr);
        }
      }
    }
    return out;
  }

  /// Inverse STFT (overlap-add, divided by the summed squared window), cut to [length].
  Float32List inverse(Spectrogram s, int length) {
    final frames = s.frames;
    final size = (frames - 1) * hop + nFft;
    final y = Float64List(size);
    final env = Float64List(size);
    final re = Float64List(nFft);
    final im = Float64List(nFft);
    for (var t = 0; t < frames; t += 2) {
      final two = t + 1 < frames;
      // Z = X + iY, both Hermitian, so ifft(Z) = x + iy.
      for (var k = 0; k < bins; k++) {
        var xr = s.re[k * frames + t], xi = s.im[k * frames + t];
        if (k == 0 || k == bins - 1) xi = 0; // irfft ignores these imaginary parts
        var yr = 0.0, yi = 0.0;
        if (two) {
          yr = s.re[k * frames + t + 1];
          yi = (k == 0 || k == bins - 1) ? 0.0 : s.im[k * frames + t + 1];
        }
        re[k] = xr - yi;
        im[k] = xi + yr;
        if (k > 0 && k < bins - 1) {
          re[nFft - k] = xr + yi;
          im[nFft - k] = -xi + yr;
        }
      }
      _fft.transform(re, im, inverse: true);
      final a = t * hop;
      for (var j = 0; j < nFft; j++) {
        final w = window[j];
        y[a + j] += re[j] * w;
        env[a + j] += w * w;
        if (two) {
          y[a + hop + j] += im[j] * w;
          env[a + hop + j] += w * w;
        }
      }
    }
    final out = Float32List(length);
    for (var i = 0; i < length; i++) {
      final j = i + pad;
      if (j >= size) break;
      out[i] = env[j] > 1e-8 ? y[j] / env[j] : 0.0;
    }
    return out;
  }
}

double _hzToMel(double f) => f < 1000 ? f / (200 / 3) : 15 + math.log(f / 1000) / (math.log(6.4) / 27);
double _melToHz(double m) => m < 15 ? m * (200 / 3) : 1000 * math.exp((math.log(6.4) / 27) * (m - 15));

/// Slaney mel filters (area-normalized), stored sparsely.
class MelFilters {
  MelFilters({required int sampleRate, required int nFft, required this.bands, double fMin = 0, required double fMax}) {
    final nBins = nFft ~/ 2 + 1;
    final lo = _hzToMel(fMin), hi = _hzToMel(fMax);
    final pts = [for (var i = 0; i < bands + 2; i++) _melToHz(lo + (hi - lo) * i / (bands + 1))];
    for (var m = 0; m < bands; m++) {
      final norm = 2 / (pts[m + 2] - pts[m]);
      final w = <double>[];
      var first = -1;
      for (var k = 0; k < nBins; k++) {
        final f = sampleRate / 2 * k / (nBins - 1);
        final up = (f - pts[m]) / math.max(pts[m + 1] - pts[m], 1e-12);
        final down = (pts[m + 2] - f) / math.max(pts[m + 2] - pts[m + 1], 1e-12);
        final v = math.max(0.0, math.min(up, down));
        if (v > 0) {
          if (first < 0) first = k;
          // Keep zeros that fall between non-zero weights.
          while (first + w.length < k) {
            w.add(0);
          }
          w.add(v * norm);
        }
      }
      starts.add(first < 0 ? 0 : first);
      weights.add(Float64List.fromList(w));
    }
  }

  final int bands;
  final starts = <int>[];
  final weights = <Float64List>[];

  /// log(max(1e-5, mel)) of the magnitude spectrogram, band-major [bands * frames].
  Float32List logMel(Spectrogram s) {
    final frames = s.frames;
    final out = Float32List(bands * frames);
    for (var m = 0; m < bands; m++) {
      final w = weights[m];
      final k0 = starts[m];
      for (var t = 0; t < frames; t++) {
        var acc = 0.0;
        for (var i = 0; i < w.length; i++) {
          final idx = (k0 + i) * frames + t;
          final r = s.re[idx], q = s.im[idx];
          acc += w[i] * math.sqrt(r * r + q * q);
        }
        out[m * frames + t] = math.log(math.max(acc, 1e-5));
      }
    }
    return out;
  }
}
