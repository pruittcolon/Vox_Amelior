// Runs the app's Clarity upscaler (Dart + ONNX Runtime through FFI) on a
// 48 kHz WAV, so CI can compare it with the numpy reference pipeline in
// tool/lab/voicelab.py.
//
//   dart run tool/clarity_check.dart --clarity DIR --in voice48k.wav --out clear.wav
import 'dart:io';

import 'package:narrator_studio/core/render/wav.dart';
import 'package:narrator_studio/core/tts/clarity.dart';
import 'package:narrator_studio/core/tts/ort.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as so;

Future<void> main(List<String> args) async {
  String arg(String name) {
    final i = args.indexOf('--$name');
    if (i < 0 || i + 1 >= args.length) throw ArgumentError('missing --$name');
    return args[i + 1];
  }

  so.initBindings();
  final input = decodeWav(File(arg('in')).readAsBytesSync());
  if (input.sampleRate != Clarity.rate) throw StateError('input must be 48 kHz, got ${input.sampleRate}');
  final clarity = Clarity(arg('clarity'), threads: 4);
  stdout.writeln('ONNX Runtime ${Ort.instance.version}');
  final watch = Stopwatch()..start();
  final out = clarity.enhance(input.samples);
  final secs = watch.elapsedMilliseconds / 1000;
  File(arg('out')).writeAsBytesSync(encodeWav(out, Clarity.rate));
  stdout.writeln('Clarity: ${input.seconds.toStringAsFixed(1)} s of audio in ${secs.toStringAsFixed(2)} s; '
      'air (energy above 8 kHz) ${airDb(input.samples, Clarity.rate).toStringAsFixed(1)} dB -> '
      '${airDb(out, Clarity.rate).toStringAsFixed(1)} dB');
  clarity.free();
  exit(0);
}
