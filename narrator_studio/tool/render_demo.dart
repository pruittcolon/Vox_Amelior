// Renders the sample podcast with the real voice models, using the app's own
// parser, planner, renderer and speech worker (no Flutter needed).
//
//   dart run tool/render_demo.dart --kokoro DIR [--pocket DIR] --out demo.wav
//
// CI runs this to prove the whole pipeline speaks, and publishes the result.
import 'dart:io';

import 'package:narrator_studio/core/render/audio.dart';
import 'package:narrator_studio/core/render/music.dart';
import 'package:narrator_studio/core/render/plan.dart';
import 'package:narrator_studio/core/render/renderer.dart';
import 'package:narrator_studio/core/render/styles.dart';
import 'package:narrator_studio/core/render/wav.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/script/script_parser.dart';
import 'package:narrator_studio/core/tts/sherpa_synth.dart';
import 'package:narrator_studio/core/tts/synth.dart';
import 'package:narrator_studio/core/tts/voices.dart';

Future<void> main(List<String> args) async {
  String? arg(String name) {
    final i = args.indexOf('--$name');
    return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
  }

  final kokoro = arg('kokoro')!;
  final pocket = arg('pocket') ?? '';
  final out = arg('out') ?? 'demo.wav';
  final which = arg('script') ?? 'podcast';
  final threads = int.tryParse(arg('threads') ?? '') ?? 4;

  final synth = SherpaSynth(() => EngineSetup(
        kokoroDir: kokoro,
        supertonicDir: arg('supertonic') ?? '',
        pocketDir: pocket,
        kokoroVoicesFile: '$kokoro/voices.bin',
        threads: threads,
      ));

  Voice k(String name) => kokoroVoices.firstWhere((v) => v.id == 'kokoro:$name');
  final script = parseScript(which == 'podcast' ? samplePodcast : sampleAudiobook);
  final style = styleById(script.style ?? 'audiobook');
  final cast = <String?, VoiceSpec>{
    null: VoiceSpec(k('af_heart'), 0.97),
    'HOST': VoiceSpec(k('am_michael'), 1.02),
    'CO-HOST': VoiceSpec(k('af_bella'), 1.02),
    'GUEST': VoiceSpec(k('bm_george'), 0.92),
  };
  if (pocket.isNotEmpty && which == 'podcast') {
    // The guest speaks in a cloned voice (Pocket TTS, reference clip from the pack).
    cast['GUEST'] = VoiceSpec(
      Voice(id: 'demo:bria', name: 'Bria', engine: Engine.pocket, blurb: '', refPath: '$pocket/test_wavs/bria.wav'),
      1.0,
    );
  }
  final plan = buildPlan(script, style);
  final renderer = Renderer(
    plan: plan,
    synth: synth,
    voiceFor: (s) => cast[s] ?? cast[null]!,
    theme: themeById(script.music ?? style.music) ?? musicThemes.first,
    onError: (u, e) => stderr.writeln('sentence $u failed: $e'),
  );
  final wav = await WavWriter.create(out, outputRate);
  final watch = Stopwatch()..start();
  final transcript = StringBuffer();
  for (var c = await renderer.next(); c != null; c = await renderer.next()) {
    await wav.add(c.samples);
    if (c.utterance != null) {
      final u = plan.utterances[c.utterance!];
      transcript.writeln(u.speech);
      stdout.writeln('${c.utterance! + 1}/${plan.utterances.length} ${u.speaker ?? 'NARRATOR'}: ${u.speech}');
    }
  }
  await wav.close();
  final secs = watch.elapsedMilliseconds / 1000;
  await File('$out.txt').writeAsString(transcript.toString());
  stdout.writeln('Rendered ${wav.seconds.toStringAsFixed(1)} s of audio in ${secs.toStringAsFixed(1)} s '
      '(${(wav.seconds / secs).toStringAsFixed(2)}x real time) -> $out');
  await synth.dispose();
  exit(0);
}
