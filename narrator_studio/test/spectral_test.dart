import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/render/spectral.dart';

Float32List _signal(int n, {int seed = 1}) {
  final r = math.Random(seed);
  return Float32List.fromList([
    for (var i = 0; i < n; i++)
      (0.4 * math.sin(2 * math.pi * 440 * i / 48000) + 0.2 * math.sin(2 * math.pi * 5300 * i / 48000) + 0.1 * (r.nextDouble() - 0.5))
          .toDouble(),
  ]);
}

void main() {
  test('FFT matches the direct DFT, forward and inverse', () {
    const n = 32;
    final r = math.Random(3);
    final xr = Float64List.fromList([for (var i = 0; i < n; i++) r.nextDouble() - 0.5]);
    final xi = Float64List.fromList([for (var i = 0; i < n; i++) r.nextDouble() - 0.5]);
    final re = Float64List.fromList(xr), im = Float64List.fromList(xi);
    Fft(n).transform(re, im);
    for (var k = 0; k < n; k++) {
      var sr = 0.0, si = 0.0;
      for (var j = 0; j < n; j++) {
        final a = -2 * math.pi * j * k / n;
        sr += xr[j] * math.cos(a) - xi[j] * math.sin(a);
        si += xr[j] * math.sin(a) + xi[j] * math.cos(a);
      }
      expect(re[k], closeTo(sr, 1e-9));
      expect(im[k], closeTo(si, 1e-9));
    }
    Fft(n).transform(re, im, inverse: true);
    for (var j = 0; j < n; j++) {
      expect(re[j], closeTo(xr[j], 1e-12));
      expect(im[j], closeTo(xi[j], 1e-12));
    }
  });

  test('STFT frames (two per FFT) equal frame-by-frame transforms', () {
    final stft = Stft(nFft: 64, hop: 16);
    final x = _signal(400);
    final s = stft.forward(x);
    expect(s.frames, stft.framesFor(400));
    // Rebuild the padded signal and check every frame, odd and even.
    final pad = stft.pad;
    final padded = [for (var j = pad; j > 0; j--) x[j], ...x, for (var j = 0; j < pad; j++) x[x.length - 2 - j]];
    for (var t = 0; t < s.frames; t++) {
      final re = Float64List(64), im = Float64List(64);
      for (var j = 0; j < 64; j++) {
        re[j] = padded[t * 16 + j] * stft.window[j];
      }
      Fft(64).transform(re, im);
      for (var k = 0; k < s.bins; k++) {
        expect(s.re[k * s.frames + t], closeTo(re[k], 1e-4), reason: 'frame $t bin $k');
        expect(s.im[k * s.frames + t], closeTo(im[k], 1e-4), reason: 'frame $t bin $k');
      }
    }
  });

  test('inverse STFT gives the signal back (odd and even frame counts)', () {
    for (final n in [48000, 48000 + 512, 9000]) {
      final stft = Stft();
      final x = _signal(n, seed: n);
      final y = stft.inverse(stft.forward(x), n);
      var err = 0.0;
      for (var i = 0; i < n; i++) {
        err = math.max(err, (x[i] - y[i]).abs());
      }
      expect(err, lessThan(1e-4), reason: 'length $n');
    }
  });

  test('mel filters: 80 bands up to 8 kHz, Slaney-normalized', () {
    final mel = MelFilters(sampleRate: 44100, nFft: 2048, bands: 80, fMax: 8000);
    expect(mel.weights, hasLength(80));
    const binHz = 44100 / 2048;
    var lastPeak = -1;
    for (var m = 0; m < 80; m++) {
      final w = mel.weights[m];
      expect(w, isNotEmpty, reason: 'band $m');
      final peak = mel.starts[m] + w.indexOf(w.reduce(math.max));
      expect(peak, greaterThanOrEqualTo(lastPeak));
      lastPeak = peak;
      // Area-normalized: weights times the bin spacing integrate to about 1.
      final area = w.reduce((a, b) => a + b) * binHz;
      // (The narrow low bands span only a few bins, so the sum is rougher there.)
      expect(area, closeTo(1.0, m < 30 ? 0.4 : 0.1), reason: 'band $m');
    }
    expect((mel.starts.last + mel.weights.last.length) * binHz, lessThan(8100));
    final silence = mel.logMel(Stft().forward(Float32List(4096)));
    expect(silence.every((v) => (v - math.log(1e-5)).abs() < 1e-6), isTrue);
  });
}
