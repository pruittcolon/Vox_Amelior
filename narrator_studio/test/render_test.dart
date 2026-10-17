import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/render/audio.dart';
import 'package:narrator_studio/core/render/music.dart';
import 'package:narrator_studio/core/render/plan.dart';
import 'package:narrator_studio/core/render/renderer.dart';
import 'package:narrator_studio/core/render/styles.dart';
import 'package:narrator_studio/core/render/wav.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/script/script_parser.dart';
import 'package:narrator_studio/core/tts/synth.dart';
import 'package:narrator_studio/core/tts/voices.dart';

import 'helpers/fakes.dart';

double rms(Float32List x, [int from = 0, int? to]) {
  final end = to ?? x.length;
  var s = 0.0;
  for (var i = from; i < end; i++) {
    s += x[i] * x[i];
  }
  return math.sqrt(s / math.max(1, end - from));
}

double peak(Float32List x) => x.fold(0.0, (a, b) => math.max(a, b.abs()));

void main() {
  final voice = kokoroVoices.first;
  VoiceSpec anyVoice(String? _) => VoiceSpec(voice, 1.0);

  group('plan', () {
    test('audiobook reads headings and spaces paragraphs', () {
      final plan = buildPlan(parseScript('# Title\n\n## One\n\nFirst. Second.\n\nThird.'), styleById('audiobook'));
      expect(plan.utterances.map((u) => u.speech), ['Title.', 'One.', 'First.', 'Second.', 'Third.']);
      expect(plan.utterances.where((u) => u.heading).length, 2);
      final gaps = plan.steps.whereType<GapStep>().map((g) => g.ms).toList();
      expect(gaps, contains(260)); // between sentences
      expect(gaps, contains(700)); // between paragraphs
      expect(plan.steps.whereType<CueStep>().where((c) => c.cue.startsWith('music')), isEmpty);
    });

    test('podcast: intro + outro music, section sound, speaker gaps', () {
      final plan = buildPlan(parseScript(samplePodcast), styleById('podcast'));
      final cues = plan.steps.whereType<CueStep>().map((c) => c.cue).toList();
      expect(cues.first, 'music:intro');
      expect(cues, contains('music:outro'));
      expect(cues, contains('sound:sweep'));
      expect(plan.utterances.any((u) => u.heading), isFalse, reason: 'podcast headings are not read');
      expect(plan.steps.whereType<GapStep>().map((g) => g.ms), contains(260));
      for (var i = 0; i < plan.utterances.length; i++) {
        expect((plan.steps[plan.stepOfUtterance[i]] as SpeakStep).utterance, i);
      }
    });

    test('music off removes every cue', () {
      final plan = buildPlan(parseScript(samplePodcast), styleById('podcast'), music: false);
      expect(plan.steps.whereType<CueStep>().where((c) => c.cue != 'section'), isEmpty);
    });

    test('[pause] lengthens the gap instead of adding to it', () {
      final plan = buildPlan(parseScript('One.\n[pause 3]\nTwo.'), styleById('audiobook'));
      expect(plan.steps.whereType<GapStep>().map((g) => g.ms).toList(), [3000]);
    });

    test('estimate is roughly words / 155 wpm', () {
      final plan = buildPlan(parseScript(List.filled(155, 'word').join(' ')), styleById('audiobook'));
      expect(plan.estimate(1.0).inSeconds, inInclusiveRange(55, 65));
    });
  });

  group('renderer', () {
    test('speech chunks carry their utterance; the output adds up', () async {
      final synth = FakeSynth();
      final plan = buildPlan(parseScript('One two three. Four five.\n\nSix.'), styleById('audiobook'));
      final r = Renderer(plan: plan, synth: synth, voiceFor: anyVoice);
      final utterances = <int>[];
      var total = 0;
      for (var c = await r.next(); c != null; c = await r.next()) {
        if (c.utterance != null) utterances.add(c.utterance!);
        total += c.samples.length;
      }
      expect(utterances, [0, 1, 2]);
      expect(synth.calls.map((c) => c.$2), ['One two three.', 'Four five.', 'Six.']);
      expect(total / outputRate, greaterThan(2));
    });

    test('starting mid-way skips earlier sentences and the intro', () async {
      final synth = FakeSynth();
      final plan = buildPlan(parseScript(samplePodcast), styleById('podcast'));
      final r = Renderer(plan: plan, synth: synth, voiceFor: anyVoice, theme: musicThemes.first, startUtterance: 5);
      final first = await r.next();
      expect(first!.utterance, 5);
      expect(synth.calls.single.$2, plan.utterances[5].speech);
    });

    test('intro music plays solo, then ducks under speech; outro tails off', () async {
      final plan = buildPlan(parseScript(samplePodcast), styleById('podcast'));
      final r = Renderer(plan: plan, synth: FakeSynth(), voiceFor: anyVoice, theme: musicThemes.first);
      final intro = (await r.next())!;
      expect(intro.utterance, isNull);
      expect(intro.samples.length / outputRate, closeTo(introSeconds, 0.01));
      final soloLevel = rms(intro.samples, outputRate ~/ 2, outputRate * 3 ~/ 2);
      expect(soloLevel, greaterThan(0.02));
      // The first gap after a sentence has only (ducked) music in it.
      RenderChunk? gap;
      for (var c = await r.next(); c != null; c = await r.next()) {
        if (c.utterance == null && c.samples.length < outputRate) {
          gap = c;
          break;
        }
      }
      expect(rms(gap!.samples), lessThan(soloLevel * 0.4));
      expect(rms(gap.samples), greaterThan(0.0005));
      RenderChunk? last;
      for (var c = await r.next(); c != null; c = await r.next()) {
        last = c;
      }
      expect(last!.utterance, isNull);
      expect(last.samples.length / outputRate, closeTo(outroSeconds, 0.01));
      final tail = Float32List.sublistView(last.samples, last.samples.length - outputRate ~/ 10);
      expect(rms(tail), lessThan(0.01), reason: 'outro fades to silence');
    });

    test('a failing sentence becomes a short silence instead of stopping', () async {
      final synth = FakeSynth()..failWith = StateError('boom');
      final errors = <int>[];
      final plan = buildPlan(parseScript('One. Two.'), styleById('audiobook'));
      final r = Renderer(plan: plan, synth: synth, voiceFor: anyVoice, onError: (u, _) => errors.add(u));
      var chunks = 0;
      while (await r.next() != null) {
        chunks++;
      }
      expect(errors, [0, 1]);
      expect(chunks, greaterThan(1));
    });
  });

  group('audio', () {
    test('resampling 44.1k -> 24k keeps a 1 kHz tone and its level', () {
      const n = 44100;
      final x = Float32List(n);
      for (var i = 0; i < n; i++) {
        x[i] = 0.5 * math.sin(2 * math.pi * 1000 * i / 44100);
      }
      final y = resample(x, 44100, 24000);
      expect(y.length, closeTo(24000, 1));
      expect(rms(y, 1000, 23000), closeTo(0.5 / math.sqrt2, 0.01));
      // Zero crossings ~ 2000 per second.
      var crossings = 0;
      for (var i = 1001; i < 23000; i++) {
        if ((y[i - 1] < 0) != (y[i] < 0)) crossings++;
      }
      expect(crossings / (22000 / 24000), closeTo(2000, 20));
    });

    test('resampling removes tones above the new Nyquist', () {
      const n = 44100;
      final x = Float32List(n);
      for (var i = 0; i < n; i++) {
        x[i] = 0.5 * math.sin(2 * math.pi * 15000 * i / 44100);
      }
      expect(rms(resample(x, 44100, 24000), 1000, 23000), lessThan(0.01));
    });

    test('trim, normalize, int16 round trip', () {
      final x = Float32List(24000);
      for (var i = 6000; i < 18000; i++) {
        x[i] = 0.05 * math.sin(i / 5);
      }
      final t = trimSilence(x, 24000);
      expect(t.length, inInclusiveRange(12000, 12000 + 2 * 960 + 240));
      final n = normalizeSpeech(t, 24000);
      expect(speechRms(n, 24000), closeTo(0.1, 0.02));
      expect(peak(n), lessThanOrEqualTo(0.97));
      final back = fromInt16Bytes(toInt16Bytes(n));
      expect(back[6000], closeTo(n[6000], 1 / 16000));
    });

    test('WAV write and read', () async {
      final dir = await Directory.systemTemp.createTemp('wav');
      final w = await WavWriter.create('${dir.path}/a.wav', 24000);
      await w.add(Float32List.fromList(List.generate(2400, (i) => math.sin(i / 10) * 0.5)));
      await w.add(Float32List(2400));
      await w.close();
      final pcm = decodeWav(await File('${dir.path}/a.wav').readAsBytes());
      expect(pcm.sampleRate, 24000);
      expect(pcm.samples.length, 4800);
      expect(pcm.samples[10], closeTo(math.sin(1) * 0.5, 1e-3));
      expect(decodeWav(encodeWav(pcm.samples, 24000)).samples.length, 4800);
    });
  });

  group('music', () {
    for (final theme in musicThemes) {
      test('${theme.id} is audible, not clipping, and repeatable', () {
        final a = MusicBed(theme).render(outputRate * 6);
        final b = MusicBed(theme).render(outputRate * 6);
        expect(rms(a), inInclusiveRange(0.03, 0.35));
        expect(peak(a), lessThan(0.99));
        expect(a, b);
      });
    }

    test('sound effects', () {
      for (final name in soundNames) {
        final s = soundEffect(name);
        expect(s.length, greaterThan(outputRate ~/ 20), reason: name);
        expect(peak(s), inInclusiveRange(0.05, 0.99), reason: name);
      }
    });
  });
}
