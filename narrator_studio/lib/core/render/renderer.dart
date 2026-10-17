/// Produces the finished audio of a [Plan], one piece at a time: speech from
/// the synth, gaps, sound cues, and the music bed mixed (and ducked) beneath.
///
/// The same renderer feeds both live listening and export, so an exported
/// podcast sounds exactly like what you heard.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tts/synth.dart';
import 'audio.dart';
import 'music.dart';
import 'plan.dart';

class RenderChunk {
  const RenderChunk(this.samples, {this.utterance});
  final Float32List samples;

  /// The utterance whose speech starts at the beginning of this chunk.
  final int? utterance;
}

/// Length of the intro (sting and music) and of the outro tail.
const introSeconds = 2.2;
const outroSeconds = 5.0;

/// Music levels (linear gain on the bed).
const _solo = 0.62;
const _duck = 0.13;

class Renderer {
  Renderer({
    required this.plan,
    required this.synth,
    required this.voiceFor,
    this.theme,
    int startUtterance = 0,
    this.sampleRate = outputRate,
    this.onError,
  }) : _step = startUtterance <= 0 || plan.utterances.isEmpty
            ? 0
            : plan.stepOfUtterance[math.min(startUtterance, plan.utterances.length - 1)];

  final Plan plan;
  final Synth synth;
  final VoiceSpec Function(String? speaker) voiceFor;

  /// Theme for music cues (null: Curious Minds when a script asks for music).
  final MusicTheme? theme;
  final int sampleRate;
  final void Function(int utterance, Object error)? onError;

  int _step;
  bool _tailDone = false;

  // Music bed state.
  MusicBed? _bed;
  double _gain = 0;
  double _target = 0;
  double _slew = 0; // gain change per sample
  bool _introBed = false;
  int _introSamples = 0; // speech-time since the intro started
  bool _outro = false;

  int get step => _step;
  bool get done => _step >= plan.steps.length && _tailDone;

  /// Next piece of audio, or null when the plan is finished.
  Future<RenderChunk?> next() async {
    while (_step < plan.steps.length) {
      final s = plan.steps[_step++];
      switch (s) {
        case SpeakStep(:final utterance):
          final u = plan.utterances[utterance];
          final spec = voiceFor(u.speaker);
          Float32List speech;
          try {
            speech = await synth.speak(spec.voice, u.speech, speed: spec.speed);
          } catch (e) {
            onError?.call(utterance, e);
            speech = silence(300, sampleRate: sampleRate);
          }
          speech = trimSilence(speech, sampleRate);
          speech = normalizeSpeech(speech, sampleRate);
          fadeEdges(speech, sampleRate, ms: 6);
          return RenderChunk(_withBed(speech), utterance: utterance);
        case GapStep(:final ms):
          return RenderChunk(_withBed(silence(ms, sampleRate: sampleRate)));
        case CueStep(:final cue):
          final chunk = _cue(cue);
          if (chunk != null) return chunk;
      }
    }
    if (!_tailDone) {
      _tailDone = true;
      if (_bed != null && _outro) return RenderChunk(_outroTail());
      if (_bed != null && _gain > 0.001) {
        _fadeTo(0, 1.5);
        return RenderChunk(_withBed(silence(1600, sampleRate: sampleRate)));
      }
    }
    return null;
  }

  RenderChunk? _cue(String cue) {
    switch (cue) {
      case 'music:intro':
        _bed = MusicBed(theme ?? musicThemes.first, sampleRate: sampleRate);
        _introBed = true;
        _introSamples = 0;
        _gain = 0;
        // A short science sting over the music: in fast, 1.5 s solo, then
        // duck under the first words (2.2 s in all).
        final n = (introSeconds * sampleRate).round();
        final out = Float32List(n);
        final bed = _bed!.render(n);
        final sting = soundEffect('science', sampleRate: sampleRate);
        for (var i = 0; i < n; i++) {
          final t = i / sampleRate;
          final g = t < 0.15
              ? _solo * t / 0.15
              : t < 1.6
                  ? _solo
                  : _solo + (_duck - _solo) * ((t - 1.6) / 0.6);
          out[i] = bed[i] * g + (i < sting.length ? sting[i] * 0.5 : 0);
        }
        _gain = _duck;
        _target = _duck;
        return RenderChunk(out);
      case 'music:outro':
        _bed ??= MusicBed(theme ?? musicThemes.first, sampleRate: sampleRate);
        _introBed = false;
        _outro = true;
        _fadeTo(_duck, 2.0);
        return null;
      case 'music:stop':
      case 'section':
        if (_bed != null && _introBed) {
          _introBed = false;
          _fadeTo(0, 2.5);
        }
        return null;
      default:
        if (cue.startsWith('sound:')) {
          return RenderChunk(_withBed(soundEffect(cue.substring(6), sampleRate: sampleRate)));
        }
        return null;
    }
  }

  void _fadeTo(double target, double seconds) {
    _target = target;
    _slew = (target - _gain).abs() / (seconds * sampleRate);
    if (_slew == 0) _slew = 1;
  }

  /// Mixes the bed (if any) under [x], following the gain envelope.
  Float32List _withBed(Float32List x) {
    final bed = _bed;
    if (bed == null) return x;
    if (_introBed) {
      _introSamples += x.length;
      // A long intro bed gets in the way; fade it after ~8 s of speech.
      if (_introSamples > 8 * sampleRate) {
        _introBed = false;
        _fadeTo(0, 3.0);
      }
    }
    final music = bed.render(x.length);
    final out = Float32List(x.length);
    for (var i = 0; i < x.length; i++) {
      if (_gain != _target) {
        _gain = _gain < _target ? math.min(_target, _gain + _slew) : math.max(_target, _gain - _slew);
      }
      out[i] = x[i] + music[i] * _gain;
    }
    if (_gain == 0 && _target == 0 && !_outro) _bed = null;
    return out;
  }

  /// After the last words: music swells, plays a moment, fades out.
  Float32List _outroTail() {
    final n = (outroSeconds * sampleRate).round();
    final music = _bed!.render(n);
    final out = Float32List(n);
    final start = _gain;
    for (var i = 0; i < n; i++) {
      final t = i / sampleRate;
      final g = t < 1.0
          ? start + (_solo - start) * t
          : t < 3.0
              ? _solo
              : _solo * (1 - (t - 3.0) / 2.0);
      out[i] = music[i] * math.max(0, g);
    }
    _bed = null;
    _gain = 0;
    return out;
  }
}
