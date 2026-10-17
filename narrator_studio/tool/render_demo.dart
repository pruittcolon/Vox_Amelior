// Renders the sample podcast with the real voice models, using the app's own
// parser, planner, renderer and speech worker (no Flutter needed).
//
//   dart run tool/render_demo.dart --kokoro DIR [--supertonic DIR] [--pocket DIR] [--clarity DIR] --out demo.wav
//
// CI runs this to prove the whole pipeline speaks, and publishes the result.
import 'dart:convert';
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
        clarityDir: arg('clarity') ?? '',
        threads: threads,
      ));

  final supertonic = arg('supertonic') ?? '';
  final all = [...kokoroVoices, ...supertonicVoices];
  // Cast exactly as the app does: the character's voice, or its Kokoro stand-in
  // when the Supertonic studio voices aren't there.
  VoiceSpec character(String id) {
    final c = builtInCharacters.firstWhere((c) => c.id == id);
    final want = c.voiceId.startsWith('supertonic:') && supertonic.isEmpty ? c.fallbackVoiceId! : c.voiceId;
    return VoiceSpec(all.firstWhere((v) => v.id == want), c.speed);
  }

  final script = parseScript(which == 'podcast' ? samplePodcast : sampleAudiobook);
  final style = styleById(script.style ?? 'audiobook');
  final cast = <String?, VoiceSpec>{
    null: character('narrator'),
    'HOST': character('host'),
    'CO-HOST': character('cohost'),
    'GUEST': character('professor'),
  };
  // Also prove the other paths speak: British Kokoro (espeak "en") and a cloned voice.
  final extras = <String, Voice>{
    'British Kokoro': all.firstWhere((v) => v.id == 'kokoro:bf_isabella'),
    if (pocket.isNotEmpty)
      'Pocket clone': Voice(id: 'demo:bria', name: 'Bria', engine: Engine.pocket, blurb: '', refPath: '$pocket/test_wavs/bria.wav'),
  };
  for (final e in extras.entries) {
    final pcm = await synth.speak(e.value, 'The tomato schedule was rather good, and the light was perfectly clear.');
    stdout.writeln('${e.key}: ${(pcm.length / outputRate).toStringAsFixed(2)} s');
    if (pcm.length < outputRate) {
      stderr.writeln('${e.key} produced no speech');
      exit(1);
    }
  }
  final plan = buildPlan(script, style);
  var failures = 0;
  final renderer = Renderer(
    plan: plan,
    synth: synth,
    voiceFor: (s) => cast[s] ?? cast[null]!,
    theme: themeById(script.music ?? style.music) ?? musicThemes.first,
    onError: (u, e) {
      stderr.writeln('sentence $u failed: $e');
      failures++;
    },
  );
  final wav = await WavWriter.create(out, outputRate);
  final watch = Stopwatch()..start();
  final transcript = StringBuffer();
  // Chapter starts, exactly as the app's exporter records them.
  final starts = {for (final c in plan.chapters) c.first: c};
  final chapters = <Map<String, Object>>[];
  var written = 0;
  for (var c = await renderer.next(); c != null; c = await renderer.next()) {
    final chapter = c.utterance == null ? null : starts[c.utterance];
    if (chapter != null) chapters.add({'ms': chapters.isEmpty ? 0 : written * 1000 ~/ outputRate, 'title': chapter.title});
    written += c.samples.length;
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
  await File('$out.chapters.json').writeAsString(jsonEncode(chapters));
  stdout.writeln('Chapters: ${chapters.map((c) => '${c['ms']} ms ${c['title']}').join(' | ')}');
  stdout.writeln('Rendered ${wav.seconds.toStringAsFixed(1)} s of audio in ${secs.toStringAsFixed(1)} s '
      '(${(wav.seconds / secs).toStringAsFixed(2)}x real time) -> $out');
  await synth.dispose();
  if (failures > 0) {
    stderr.writeln('$failures sentence(s) failed');
    exit(1);
  }
  exit(0);
}
