// Runs the app's VibeVoice port (lib/core/tts/vibevoice.dart) on the exported
// pack and compares it with the Python reference loop (tool/vibevoice):
// the tokenizer on test strings, then the speech frames and audio given the
// same diffusion noise. Also reports how fast it is.
//
//   dart run tool/vibevoice_check.dart --pack DIR --check DIR [--voice carter] [--threads 4]
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:narrator_studio/core/render/wav.dart';
import 'package:narrator_studio/core/tts/vibevoice.dart';

Float32List readF32(String path) {
  final b = File(path).readAsBytesSync();
  return Float32List.fromList(b.buffer.asFloat32List(b.offsetInBytes, b.lengthInBytes ~/ 4));
}

double snrDb(List<double> ref, List<double> got) {
  var s = 0.0, e = 0.0;
  for (var i = 0; i < math.min(ref.length, got.length); i++) {
    s += ref[i] * ref[i];
    e += (ref[i] - got[i]) * (ref[i] - got[i]);
  }
  return 10 * math.log(s / math.max(e, 1e-20)) / math.ln10;
}

Future<void> main(List<String> args) async {
  String? arg(String name) {
    final i = args.indexOf('--$name');
    return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
  }

  final pack = arg('pack')!;
  final check = arg('check')!;
  final threads = int.tryParse(arg('threads') ?? '') ?? 4;
  var ok = true;

  final watch = Stopwatch()..start();
  final vv = VibeVoice(pack, threads: threads);
  stdout.writeln('loaded in ${watch.elapsedMilliseconds} ms');

  // 1. Tokenizer: the same ids as Python's tokenizers library.
  final cases = (jsonDecode(File('$check/tokenizer_cases.json').readAsStringSync()) as List).cast<Map<String, Object?>>();
  for (final c in cases) {
    final want = (c['ids']! as List).cast<int>();
    final got = vv.tokens(c['text']! as String);
    final same = want.length == got.length && [for (var i = 0; i < want.length; i++) want[i] == got[i]].every((x) => x);
    stdout.writeln('${same ? 'same' : 'DIFFERENT'} tokens for ${jsonEncode(c['text'])}');
    if (!same) {
      stdout.writeln('  python: $want\n  dart:   $got');
      ok = false;
    }
  }

  // 2. The loop, with Microsoft's noise: the frames must follow the reference.
  const text = "Here's the key. Short blue waves are scattered far more than long red waves, roughly five times more.";
  for (final voice in (arg('voice') ?? 'carter,emma').split(',')) {
    final noise = readF32('$check/$voice-noise.f32');
    final refLatents = readF32('$check/$voice-latents.f32');
    watch.reset();
    final latents = vv.generate(voice, text, noise: (i) => Float32List.sublistView(noise, i * 64, i * 64 + 64));
    final genMs = watch.elapsedMilliseconds;
    watch.reset();
    final audio = vv.decode(latents);
    final decMs = watch.elapsedMilliseconds;
    final seconds = audio.length / VibeVoice.sampleRate;
    final flat = [for (final l in latents) ...l];
    final refFrames = refLatents.length ~/ 64;
    final latSnr = snrDb(refLatents, flat);
    stdout.writeln('$voice: ${latents.length} frames (reference $refFrames), latents SNR ${latSnr.toStringAsFixed(1)} dB; '
        '${seconds.toStringAsFixed(2)} s of audio: generate $genMs ms + decode $decMs ms '
        '= ${((genMs + decMs) / 1000 / seconds).toStringAsFixed(2)} s of work per second of speech');
    final ref = decodeWav(File('$check/$voice-onnx.wav').readAsBytesSync()).samples;
    stdout.writeln('  audio vs the Python ONNX loop: SNR ${snrDb(ref, audio).toStringAsFixed(1)} dB');
    final out = await WavWriter.create('$check/$voice-dart.wav', VibeVoice.sampleRate);
    await out.add(audio);
    await out.close();
    if ((latents.length - refFrames).abs() > 2 || latSnr < 20) ok = false;
  }

  // 3. Free-running (the app's own noise): speaks, and stops by itself.
  watch.reset();
  final free = vv.speak('emma', 'Welcome to Curious Minds, the show where we chase one everyday question all the way down.');
  final secs = free.length / VibeVoice.sampleRate;
  stdout.writeln('emma, own noise: ${secs.toStringAsFixed(2)} s of audio in ${watch.elapsedMilliseconds} ms');
  final wav = await WavWriter.create('$check/emma-free-dart.wav', VibeVoice.sampleRate);
  await wav.add(free);
  await wav.close();
  if (secs < 2 || secs > 15) ok = false;
  vv.free();
  stdout.writeln(ok ? 'VibeVoice port OK' : 'VibeVoice port DIFFERS');
  exit(ok ? 0 : 1);
}
