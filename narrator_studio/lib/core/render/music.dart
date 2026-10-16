/// Original intro/outro music and sound cues, synthesized on the phone.
///
/// No samples or downloads: every theme is a little generative arrangement
/// (pad + plucked arpeggio + bass, some with soft drums), so it is free of any
/// license and sounds the same every time.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio.dart';

class MusicTheme {
  const MusicTheme({
    required this.id,
    required this.name,
    required this.mood,
    required this.bpm,
    required this.chords,
    required this.arp,
    this.barsPerChord = 1,
    this.padCutoff = 1400,
    this.pluckDecay = 0.28,
    this.drums = false,
    this.shaker = false,
    this.bell = false,
    this.echo = 0.3,
  });

  final String id;
  final String name;
  final String mood;
  final double bpm;

  /// MIDI notes per chord (root first).
  final List<List<int>> chords;

  /// Arpeggio pattern: indexes into the chord, -1 = rest.
  final List<int> arp;
  final int barsPerChord;
  final double padCutoff;
  final double pluckDecay;
  final bool drums;
  final bool shaker;
  final bool bell;
  final double echo;
}

const musicThemes = <MusicTheme>[
  MusicTheme(
    id: 'curious',
    name: 'Curious Minds',
    mood: 'Bright and friendly. The classic explainer-podcast intro.',
    bpm: 100,
    chords: [
      [60, 64, 67, 72], // C
      [55, 62, 67, 71], // G
      [57, 60, 64, 69], // Am
      [53, 60, 65, 69], // F
    ],
    arp: [0, 2, 1, 3, 2, 1, 3, 2],
    shaker: true,
  ),
  MusicTheme(
    id: 'cosmic',
    name: 'Deep Space',
    mood: 'Slow and wondrous. Science, the universe, bedtime stories.',
    bpm: 68,
    chords: [
      [57, 60, 64, 71], // Am(add9)
      [53, 57, 60, 67], // Fmaj(add9)
      [48, 55, 60, 64], // C
      [55, 59, 62, 69], // G(add9)
    ],
    arp: [0, -1, 2, -1, 3, -1, 1, -1],
    barsPerChord: 1,
    padCutoff: 1800,
    pluckDecay: 0.9,
    bell: true,
    echo: 0.45,
  ),
  MusicTheme(
    id: 'fireside',
    name: 'Fireside',
    mood: 'Warm and acoustic. Interviews and history.',
    bpm: 84,
    chords: [
      [50, 57, 62, 66], // D
      [47, 54, 59, 62], // Bm
      [43, 55, 59, 62], // G
      [45, 52, 57, 61], // A
    ],
    arp: [0, 2, 3, 2, 1, 2, 3, 2],
    padCutoff: 900,
    pluckDecay: 0.45,
  ),
  MusicTheme(
    id: 'newsroom',
    name: 'Newsroom',
    mood: 'Driving pulse. Briefings and top-10 lists.',
    bpm: 118,
    chords: [
      [52, 59, 64, 67], // Em
      [48, 55, 60, 64], // C
      [55, 59, 62, 67], // G
      [50, 57, 62, 66], // D
    ],
    arp: [0, 0, 2, 0, 3, 0, 2, 1],
    padCutoff: 1100,
    pluckDecay: 0.12,
    drums: true,
    echo: 0.18,
  ),
];

MusicTheme? themeById(String? id) {
  for (final t in musicThemes) {
    if (t.id == id) return t;
  }
  return null;
}

double _hz(int midi) => 440.0 * math.pow(2, (midi - 69) / 12);

/// Band-limited sawtooth, one cycle.
final Float32List _saw = () {
  const n = 2048;
  final t = Float32List(n);
  for (var i = 0; i < n; i++) {
    var v = 0.0;
    for (var h = 1; h <= 10; h++) {
      v += math.sin(2 * math.pi * h * i / n) / h;
    }
    t[i] = v * 0.55;
  }
  return t;
}();

double _wave(Float32List table, double phase) {
  final x = (phase - phase.floorToDouble()) * table.length;
  final i = x.floor();
  final f = x - i;
  final a = table[i % table.length];
  final b = table[(i + 1) % table.length];
  return a + (b - a) * f;
}

/// Plays a theme forever; call [render] for the next n samples.
class MusicBed {
  MusicBed(this.theme, {this.sampleRate = outputRate})
      : _beat = sampleRate * 60 / theme.bpm,
        _delay = Float32List((sampleRate * 60 / theme.bpm * 0.75).round());

  final MusicTheme theme;
  final int sampleRate;
  final double _beat;
  final Float32List _delay;
  int _t = 0;
  int _delayAt = 0;
  double _padLow = 0;
  double _hatPrev = 0;
  int _noise = 0x1234567;

  double _rand() {
    _noise = (_noise * 1103515245 + 12345) & 0x7fffffff;
    return _noise / 0x3fffffff - 1;
  }

  Float32List render(int n) {
    final out = Float32List(n);
    final sr = sampleRate.toDouble();
    final seg = _beat * 4 * theme.barsPerChord;
    final step = _beat / 2; // eighth notes
    final padA = 1 - math.exp(-2 * math.pi * theme.padCutoff / sr);
    for (var k = 0; k < n; k++, _t++) {
      final t = _t.toDouble();
      final segIndex = (t / seg).floor();
      final segPos = (t - segIndex * seg) / sr;
      final chord = theme.chords[segIndex % theme.chords.length];
      final prev = theme.chords[(segIndex - 1) % theme.chords.length];

      // Pad: two detuned saws per note, crossfading between chords.
      var pad = 0.0;
      final inGain = math.min(1.0, segPos / 0.9);
      final outGain = segIndex == 0 ? 0.0 : math.max(0.0, 1 - segPos / 1.2);
      for (var i = 1; i < chord.length; i++) {
        final f = _hz(chord[i]);
        pad += inGain * (_wave(_saw, f * 0.997 * t / sr) + _wave(_saw, f * 1.003 * t / sr));
      }
      if (outGain > 0) {
        for (var i = 1; i < prev.length; i++) {
          final f = _hz(prev[i]);
          pad += outGain * (_wave(_saw, f * 0.997 * t / sr) + _wave(_saw, f * 1.003 * t / sr));
        }
      }
      final lfo = 0.85 + 0.15 * math.sin(2 * math.pi * 0.07 * t / sr);
      _padLow += padA * lfo * (pad - _padLow);
      var mix = _padLow * 0.05;

      // Plucked arpeggio (last three notes ring out).
      final stepIndex = (t / step).floor();
      var pluck = 0.0;
      for (var back = 0; back < 3; back++) {
        final si = stepIndex - back;
        if (si < 0) break;
        final idx = theme.arp[si % theme.arp.length];
        if (idx < 0) continue;
        final noteChord = theme.chords[((si * step) / seg).floor() % theme.chords.length];
        final f = _hz(noteChord[idx % noteChord.length] + 12);
        final pos = (t - si * step) / sr;
        final env = math.min(1.0, pos / 0.004) * math.exp(-pos / theme.pluckDecay);
        if (env < 1e-4) continue;
        final ph = 2 * math.pi * f * pos;
        pluck += env * (math.sin(ph) + 0.35 * math.sin(2 * ph) + 0.12 * math.sin(3 * ph));
      }

      // Bell sparkle on the first beat of each chord (cosmic).
      if (theme.bell) {
        final f = _hz(chord.last + 12);
        final env = math.exp(-segPos / 1.6) * math.min(1.0, segPos / 0.003);
        final ph = 2 * math.pi * f * segPos;
        pluck += 0.6 * env * math.sin(ph + 1.8 * math.exp(-segPos / 0.4) * math.sin(3.5 * ph));
      }

      // Echo on the plucks.
      final delayed = _delay[_delayAt];
      _delay[_delayAt] = pluck + delayed * theme.echo;
      _delayAt = (_delayAt + 1) % _delay.length;
      mix += 0.16 * (pluck + delayed * 0.6);

      // Bass: root, two octaves down, on every bar.
      final bar = (t / (_beat * 4)).floor();
      final barPos = (t - bar * _beat * 4) / sr;
      final bassEnv = math.min(1.0, barPos / 0.02) * (0.55 + 0.45 * math.exp(-barPos / 0.5));
      mix += 0.13 * bassEnv * math.sin(2 * math.pi * _hz(chord[0] - 12) * t / sr);

      // Drums.
      final beat = (t / _beat).floor();
      final beatPos = (t - beat * _beat) / sr;
      if (theme.drums) {
        if (beat % 2 == 0) {
          final ph = 2 * math.pi * (48 * beatPos + 110 * 0.035 * (1 - math.exp(-beatPos / 0.035)));
          mix += 0.32 * math.exp(-beatPos / 0.16) * math.sin(ph);
        }
        final half = (t / step).floor();
        final halfPos = (t - half * step) / sr;
        if (half.isOdd) {
          final n0 = _rand();
          final hp = n0 - _hatPrev;
          _hatPrev = n0;
          mix += 0.05 * math.exp(-halfPos / 0.025) * hp;
        }
      }
      if (theme.shaker) {
        final sixteenth = _beat / 4;
        final s = (t / sixteenth).floor();
        final sPos = (t - s * sixteenth) / sr;
        final accent = s % 2 == 1 ? 1.0 : 0.5;
        final n0 = _rand();
        final hp = n0 - _hatPrev;
        _hatPrev = n0;
        mix += 0.02 * accent * math.exp(-sPos / 0.03) * hp;
      }

      out[k] = _softClip(mix * 1.6) * 0.55;
    }
    return out;
  }
}

double _softClip(double x) {
  if (x > 3) return 1;
  if (x < -3) return -1;
  final x2 = x * x;
  return x * (27 + x2) / (27 + 9 * x2); // tanh approximation
}

/// Short sound cues: chime, ding, whoosh, pop.
const soundNames = ['chime', 'ding', 'whoosh', 'pop'];

Float32List soundEffect(String name, {int sampleRate = outputRate}) {
  final sr = sampleRate.toDouble();
  Float32List bell(List<(double, double, double)> notes, double seconds) {
    final out = Float32List((seconds * sr).round());
    for (final (freq, start, amp) in notes) {
      final s0 = (start * sr).round();
      for (var i = s0; i < out.length; i++) {
        final t = (i - s0) / sr;
        final env = math.min(1.0, t / 0.002) * math.exp(-t / 0.55);
        final ph = 2 * math.pi * freq * t;
        out[i] += amp * env * math.sin(ph + 1.3 * math.exp(-t / 0.25) * math.sin(3.5 * ph));
      }
    }
    return out;
  }

  switch (name) {
    case 'chime' || 'bell' || 'transition-chime':
      return bell([(1318.5, 0, 0.32), (987.8, 0.16, 0.3)], 1.6);
    case 'ding':
      return bell([(1174.7, 0, 0.35)], 1.2);
    case 'pop':
      final out = Float32List((0.12 * sr).round());
      for (var i = 0; i < out.length; i++) {
        final t = i / sr;
        out[i] = 0.4 * math.exp(-t / 0.03) * math.sin(2 * math.pi * (600 + 900 * math.exp(-t / 0.02)) * t);
      }
      return out;
    case 'whoosh' || 'transition' || 'swoosh':
    default:
      final out = Float32List((0.9 * sr).round());
      var seed = 99;
      double low = 0, band = 0;
      for (var i = 0; i < out.length; i++) {
        final x = i / out.length;
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        final noise = seed / 0x3fffffff - 1;
        final fc = 300 + 3500 * math.sin(math.pi * x);
        final f = 2 * math.sin(math.pi * fc / sr);
        low += f * band;
        final high = noise - low - 0.5 * band;
        band += f * high;
        out[i] = 0.8 * _softClip(0.3 * band * math.sin(math.pi * x));
      }
      return out;
  }
}
