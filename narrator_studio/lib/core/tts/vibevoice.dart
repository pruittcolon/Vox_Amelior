/// VibeVoice-Realtime-0.5B (Microsoft, MIT) on ONNX Runtime: the most
/// lifelike voices in the app.
///
/// A port of tool/vibevoice/vv_pack.py, which follows Microsoft's
/// generate(): text goes in through the 4-layer text LM five tokens at a
/// time; after each window the 20-layer TTS LM produces six speech frames
/// (7.5 per second). Each frame is a 64-number latent drawn by a diffusion
/// head in five DPM-Solver++ steps with classifier-free guidance (a second,
/// voice-only TTS LM branch is the "unconditioned" side); the latent is fed
/// back through the acoustic connector, and a small classifier says when the
/// speech is over. The acoustic decoder then turns all frames into 24 kHz
/// audio at once.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import 'bpe.dart';
import 'ort.dart';

const _hidden = 896;
const _latent = 64;
const _kvHeads = 2;
const _headDim = 64;
const _textWindow = 5;
const _speechWindow = 6;
const _maxPositions = 8192;

/// A voice: the prompt caches Microsoft ships for it (text LM, TTS LM and the
/// TTS LM's negative branch), plus the negative branch's last hidden state.
class VibeVoicePrompt {
  VibeVoicePrompt._(this.lm, this.tts, this.neg, this.negLast);

  /// Per layer (k, v), each [1, 2, length, 64].
  final List<(Float32List, Float32List)> lm;
  final List<(Float32List, Float32List)> tts;
  final List<(Float32List, Float32List)> neg;
  final Float32List negLast;

  int length(List<(Float32List, Float32List)> kv) => kv.first.$1.length ~/ (_kvHeads * _headDim);

  static VibeVoicePrompt read(String path) {
    final raw = File(path).readAsBytesSync();
    final b = ByteData.sublistView(raw);
    if (String.fromCharCodes(raw.sublist(0, 4)) != 'VVP1') throw StateError('Not a VibeVoice voice: $path');
    var at = 28;
    // Little-endian float32 throughout (as are Android's CPUs).
    Float32List floats(int n) {
      final out = Float32List.fromList(raw.buffer.asFloat32List(raw.offsetInBytes + at, n));
      at += 4 * n;
      return out;
    }

    final parts = <List<(Float32List, Float32List)>>[];
    for (var i = 0; i < 3; i++) {
      final layers = b.getInt32(4 + 8 * i, Endian.little);
      final length = b.getInt32(8 + 8 * i, Endian.little);
      final n = _kvHeads * length * _headDim;
      parts.add([for (var l = 0; l < layers; l++) (floats(n), floats(n))]);
    }
    return VibeVoicePrompt._(parts[0], parts[1], parts[2], floats(_hidden));
  }
}

class VibeVoice {
  VibeVoice(this.dir, {int threads = 4})
      : _textLm = Ort.instance.open(p.join(dir, 'text_lm.onnx'), threads: threads),
        _ttsLm = Ort.instance.open(p.join(dir, 'tts_lm.onnx'), threads: threads),
        _head = Ort.instance.open(p.join(dir, 'head.onnx'), threads: threads),
        _decoder = Ort.instance.open(p.join(dir, 'decoder.onnx'), threads: threads),
        _bpe = Bpe.fromFile(p.join(dir, 'tokenizer.json')),
        _embed = File(p.join(dir, 'embed.f16')).openSync() {
    final cfg = jsonDecode(File(p.join(dir, 'vibevoice.json')).readAsStringSync()) as Map<String, Object?>;
    _scaling = (cfg['scaling']! as num).toDouble();
    _bias = (cfg['bias']! as num).toDouble();
    _cfgScale = (cfg['cfg_scale']! as num).toDouble();
    final s = cfg['scheduler']! as Map<String, Object?>;
    List<double> list(String k) => [for (final v in s[k]! as List<Object?>) (v! as num).toDouble()];
    _timesteps = list('timesteps');
    _alpha = list('alpha');
    _sigma = list('sigma');
    _cs = list('cs');
    _c0 = list('c0');
    _c1 = list('c1');
    _readSmall(p.join(dir, 'small.f32'));
  }

  static const files = [
    'text_lm.onnx', 'text_lm.onnx.data', 'tts_lm.onnx', 'tts_lm.onnx.data', 'head.onnx', 'head.onnx.data',
    'decoder.onnx', 'decoder.onnx.data', 'embed.f16', 'small.f32', 'tokenizer.json', 'vibevoice.json', 'voices.json',
  ];
  static const sampleRate = 24000;

  static bool installedIn(String dir) => dir.isNotEmpty && File(p.join(dir, 'vibevoice.json')).existsSync();

  final String dir;
  final OrtSession _textLm;
  final OrtSession _ttsLm;
  final OrtSession _head;
  final OrtSession _decoder;
  final Bpe _bpe;
  final RandomAccessFile _embed;
  final _voices = <String, VibeVoicePrompt>{};

  late final double _scaling, _bias, _cfgScale;
  late final List<double> _timesteps, _alpha, _sigma, _cs, _c0, _c1;
  late final Float32List _types, _connFc1W, _connFc1B, _connNormW, _connFc2W, _connFc2B;
  late final Float32List _eosFc1W, _eosFc1B, _eosFc2W, _eosFc2B;

  void _readSmall(String path) {
    final raw = File(path).readAsBytesSync();
    final all = raw.buffer.asFloat32List(raw.offsetInBytes, raw.lengthInBytes ~/ 4);
    var at = 0;
    Float32List take(int n) {
      final v = Float32List.fromList(all.sublist(at, at + n));
      at += n;
      return v;
    }

    _types = take(2 * _hidden);
    _connFc1W = take(_hidden * _latent);
    _connFc1B = take(_hidden);
    _connNormW = take(_hidden);
    _connFc2W = take(_hidden * _hidden);
    _connFc2B = take(_hidden);
    _eosFc1W = take(_hidden * _hidden);
    _eosFc1B = take(_hidden);
    _eosFc2W = take(_hidden);
    _eosFc2B = take(1);
    if (at != all.length) throw StateError('small.f32 has an unexpected size');
  }

  VibeVoicePrompt voice(String name) =>
      _voices[name] ??= VibeVoicePrompt.read(p.join(dir, 'voices', '$name.bin'));

  List<int> tokens(String text) => _bpe.encode('${text.trim()}\n');

  /// Speaks [text] in [voiceName]; 24 kHz audio. [noise] gives the i-th
  /// Gaussian vector (tests pass Microsoft's); otherwise a seeded generator.
  Float32List speak(String voiceName, String text, {int seed = 7, Float32List Function(int frame)? noise}) {
    final latents = generate(voiceName, text, seed: seed, noise: noise);
    return decode(latents);
  }

  /// The speech frames, [frames][64].
  List<Float32List> generate(String voiceName, String text, {int seed = 7, Float32List Function(int frame)? noise}) {
    final v = voice(voiceName);
    final rng = _Gauss(seed);
    noise ??= (_) => rng.vector(_latent);
    final ids = tokens(text);
    var lmKv = [for (final (k, vv) in v.lm) _kv(k, vv, v.length(v.lm))];
    var ttsKv = [for (final (k, vv) in v.tts) _kv(k, vv, v.length(v.tts))];
    var negKv = [for (final (k, vv) in v.neg) _kv(k, vv, v.length(v.neg))];
    var lmPos = v.length(v.lm), ttsPos = v.length(v.tts), negPos = v.length(v.neg);
    var negH = Float32List.fromList(v.negLast);
    Float32List? ttsH;
    final latents = <Float32List>[];
    try {
      var w = 0;
      var done = false;
      while (!done) {
        final from = w * _textWindow;
        final win = from < ids.length ? ids.sublist(from, math.min(ids.length, from + _textWindow)) : const <int>[];
        w++;
        if (win.isNotEmpty) {
          final n = win.length;
          final (h, kv) = _lm(_textLm, _embedRows(win), n, lmPos, lmKv);
          lmKv = kv;
          lmPos += n;
          final x = Float32List.fromList(h);
          for (var t = 0; t < n; t++) {
            for (var i = 0; i < _hidden; i++) {
              x[t * _hidden + i] += _types[_hidden + i]; // "text" type
            }
          }
          final (th, tkv) = _lm(_ttsLm, x, n, ttsPos, ttsKv);
          ttsKv = tkv;
          ttsPos += n;
          ttsH = Float32List.sublistView(th, (n - 1) * _hidden);
        } else if (ttsH == null) {
          break;
        }
        for (var s = 0; s < _speechWindow; s++) {
          final z = _sample(ttsH!, negH, noise(latents.length));
          latents.add(z);
          final e = _connector(z);
          for (var i = 0; i < _hidden; i++) {
            e[i] += _types[i]; // "speech" type
          }
          if (ttsPos >= _maxPositions) {
            done = true;
            break;
          }
          final (th, tkv) = _lm(_ttsLm, e, 1, ttsPos, ttsKv);
          ttsKv = tkv;
          ttsPos++;
          ttsH = th;
          final (nh, nkv) = _lm(_ttsLm, e, 1, negPos, negKv);
          negKv = nkv;
          negPos++;
          negH = nh;
          if (_eos(th)) {
            done = true;
            break;
          }
        }
      }
    } finally {
      for (final kv in [...lmKv, ...ttsKv, ...negKv]) {
        kv.free();
      }
    }
    return latents;
  }

  /// All frames to 24 kHz audio at once.
  Float32List decode(List<Float32List> latents) {
    final t = latents.length;
    if (t == 0) return Float32List(0);
    final x = Float32List(_latent * t); // [1, 64, T]
    for (var f = 0; f < t; f++) {
      for (var c = 0; c < _latent; c++) {
        x[c * t + f] = latents[f][c] / _scaling - _bias;
      }
    }
    final input = _decoder.tensor(x, [1, _latent, t]);
    try {
      final out = _decoder.run([input]);
      try {
        return Float32List.fromList(out.first.data);
      } finally {
        for (final o in out) {
          o.free();
        }
      }
    } finally {
      input.free();
    }
  }

  _Kv _kv(Float32List k, Float32List v, int length) => _Kv(
        _ttsLm.tensor(k, [1, _kvHeads, length, _headDim]),
        _ttsLm.tensor(v, [1, _kvHeads, length, _headDim]),
      );

  /// One pass of a language model over [n] new positions; returns the hidden
  /// states [n][896] (a copy) and the grown cache (the old one is freed).
  (Float32List, List<_Kv>) _lm(OrtSession s, Float32List x, int n, int pos, List<_Kv> past) {
    final xs = s.tensor(x, [1, n, _hidden]);
    final ps = s.tensor(Float32List.fromList([for (var i = 0; i < n; i++) (pos + i).toDouble()]), [1, n]);
    List<OrtTensor> out;
    try {
      out = s.run([xs, ps, for (final kv in past) ...[kv.k, kv.v]]);
    } finally {
      xs.free();
      ps.free();
    }
    final h = Float32List.fromList(out.first.data);
    out.first.free();
    for (final kv in past) {
      kv.free();
    }
    return (h, [for (var i = 0; i < past.length; i++) _Kv(out[1 + 2 * i], out[2 + 2 * i])]);
  }

  Float32List _embedRows(List<int> ids) {
    final out = Float32List(ids.length * _hidden);
    final row = Uint8List(_hidden * 2);
    for (var t = 0; t < ids.length; t++) {
      _embed.setPositionSync(ids[t] * _hidden * 2);
      _embed.readIntoSync(row);
      final halves = row.buffer.asUint16List();
      for (var i = 0; i < _hidden; i++) {
        out[t * _hidden + i] = _halves[halves[i]];
      }
    }
    return out;
  }

  /// One frame: five DPM-Solver++ steps, guided by the negative branch.
  Float32List _sample(Float32List cond, Float32List neg, Float32List noise) {
    var x = Float32List.fromList(noise);
    var prev = Float32List(_latent);
    final c = Float32List(2 * _hidden)
      ..setAll(0, cond)
      ..setAll(_hidden, neg);
    final condT = _head.tensor(c, [2, _hidden]);
    try {
      for (var k = 0; k < _timesteps.length; k++) {
        final noisy = Float32List(2 * _latent)
          ..setAll(0, x)
          ..setAll(_latent, x);
        final nt = _head.tensor(noisy, [2, _latent]);
        final tt = _head.tensor(Float32List.fromList([_timesteps[k], _timesteps[k]]), [2]);
        List<OrtTensor> out;
        try {
          out = _head.run([nt, tt, condT]);
        } finally {
          nt.free();
          tt.free();
        }
        final v = out.first.data;
        final next = Float32List(_latent);
        final x0 = Float32List(_latent);
        for (var i = 0; i < _latent; i++) {
          final g = v[_latent + i] + _cfgScale * (v[i] - v[_latent + i]);
          x0[i] = _alpha[k] * x[i] - _sigma[k] * g;
          next[i] = _cs[k] * x[i] + _c0[k] * x0[i] + _c1[k] * prev[i];
        }
        for (final o in out) {
          o.free();
        }
        x = next;
        prev = x0;
      }
    } finally {
      condT.free();
    }
    return x;
  }

  Float32List _connector(Float32List z) {
    final a = _linear(_connFc1W, _connFc1B, z, _hidden, _latent);
    var ss = 0.0;
    for (final v in a) {
      ss += v * v;
    }
    final r = 1 / math.sqrt(ss / _hidden + 1e-6);
    for (var i = 0; i < _hidden; i++) {
      a[i] = a[i] * r * _connNormW[i];
    }
    return _linear(_connFc2W, _connFc2B, a, _hidden, _hidden);
  }

  bool _eos(Float32List h) {
    final a = _linear(_eosFc1W, _eosFc1B, h, _hidden, _hidden);
    var y = _eosFc2B[0].toDouble();
    for (var i = 0; i < _hidden; i++) {
      if (a[i] > 0) y += _eosFc2W[i] * a[i];
    }
    return y > 0; // sigmoid(y) > 0.5
  }

  /// w [out][in] times x, plus b.
  static Float32List _linear(Float32List w, Float32List b, Float32List x, int out, int inn) {
    final y = Float32List(out);
    for (var o = 0; o < out; o++) {
      var acc = b[o].toDouble();
      final row = o * inn;
      for (var i = 0; i < inn; i++) {
        acc += w[row + i] * x[i];
      }
      y[o] = acc;
    }
    return y;
  }

  void free() {
    _textLm.free();
    _ttsLm.free();
    _head.free();
    _decoder.free();
    _embed.closeSync();
  }
}

class _Kv {
  _Kv(this.k, this.v);
  final OrtTensor k;
  final OrtTensor v;

  void free() {
    k.free();
    v.free();
  }
}

/// Gaussian numbers from a seeded generator (Box-Muller).
class _Gauss {
  _Gauss(int seed) : _r = math.Random(seed);
  final math.Random _r;

  Float32List vector(int n) {
    final out = Float32List(n);
    for (var i = 0; i < n; i += 2) {
      final u = math.max(_r.nextDouble(), 1e-12), v = _r.nextDouble();
      final m = math.sqrt(-2 * math.log(u));
      out[i] = m * math.cos(2 * math.pi * v);
      if (i + 1 < n) out[i + 1] = m * math.sin(2 * math.pi * v);
    }
    return out;
  }
}

/// Every float16 value, by its bits.
final Float32List _halves = Float32List.fromList([for (var h = 0; h < 65536; h++) halfToDouble(h)]);

/// IEEE half-precision bits to a double.
double halfToDouble(int h) {
  final sign = (h >> 15) & 1;
  final exp = (h >> 10) & 0x1F;
  final frac = h & 0x3FF;
  double v;
  if (exp == 0) {
    v = frac * math.pow(2, -24).toDouble();
  } else if (exp == 31) {
    v = frac == 0 ? double.infinity : double.nan;
  } else {
    v = (1 + frac / 1024) * math.pow(2, exp - 15).toDouble();
  }
  return sign == 1 ? -v : v;
}
