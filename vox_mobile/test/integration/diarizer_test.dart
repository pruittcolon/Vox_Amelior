// Runs the converted NVIDIA Nemotron 3 Diarization model through Vox's own
// ONNX Runtime bindings (the runtime that ships inside sherpa-onnx).
// Skipped unless VOX_DIARIZER points at diarizer.int8.onnx and SHERPA_LIB_DIR
// at the folder holding libonnxruntime.so. VOX_MULTI_WAV may point at a
// 16 kHz recording with several speakers.
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/native/sortformer_diarizer.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';

void main() {
  final model = Platform.environment['VOX_DIARIZER'];
  final skip = model == null || Platform.environment['SHERPA_LIB_DIR'] == null
      ? 'Set VOX_DIARIZER and SHERPA_LIB_DIR to run the real diarizer'
      : null;

  Set<int> speakersIn(SpeakerActivity a) => {
        for (var t = 0; t < a.frames; t++)
          for (var s = 0; s < a.speakers; s++)
            if (a.at(t, s) > 0.5) s,
      };

  test('one voice is one speaker, with timing that matches the audio', () {
    final d = SortformerDiarizer(model!);
    final speech = readWavAs16k('test/fixtures/speech.wav');
    final watch = Stopwatch()..start();
    final a = d.analyze(speech, 16000);
    final seconds = speech.length / 16000;
    // ignore: avoid_print
    print('one speaker: ${a.frames} frames × ${a.speakers} slots (${a.frameSeconds}s/frame) in ${watch.elapsedMilliseconds} ms');
    expect(a.frameSeconds, closeTo(seconds / a.frames, 1e-9));
    expect(speakersIn(a), hasLength(1));
    expect(const SpeakerTurns().turns(a, totalSeconds: seconds), hasLength(1));
    d.dispose();
  }, skip: skip);

  test('two different people back to back are split where the voice changes', () {
    final multi = Platform.environment['VOX_MULTI_WAV'];
    if (multi == null) return;
    final d = SortformerDiarizer(model!);
    final other = readWavAs16k(multi);
    final speech = readWavAs16k('test/fixtures/speech.wav');
    // 6 s of the other recording, then our speaker.
    final audio = Float32List.fromList([...Float32List.sublistView(other, 0, 16000 * 6), ...speech]);
    final a = d.analyze(audio, 16000);
    final turns = const SpeakerTurns().turns(a, totalSeconds: audio.length / 16000);
    // ignore: avoid_print
    print('turns: $turns');
    expect(speakersIn(a).length, greaterThanOrEqualTo(2));
    expect(turns.length, greaterThanOrEqualTo(2));
    expect(turns.any((t) => (t.start - 6).abs() < 1.0), isTrue, reason: 'a turn should start near 6 s');
    d.dispose();
  }, skip: skip);
}
