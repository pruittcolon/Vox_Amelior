// Runs the real speech models on this machine's CPU.
// Skipped unless VOX_MODELS_DIR points at a folder with:
//   encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt
//   silero_vad.onnx titanet.onnx test.wav
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/pipeline/listening_pipeline.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

void main() {
  final dir = Platform.environment['VOX_MODELS_DIR'];
  final skip = dir == null ? 'Set VOX_MODELS_DIR to run real-model tests' : null;
  String f(String name) => p.join(dir ?? '', name);

  SpeechModelPaths paths() => SpeechModelPaths(
        encoder: f('encoder.int8.onnx'),
        decoder: f('decoder.int8.onnx'),
        joiner: f('joiner.int8.onnx'),
        tokens: f('tokens.txt'),
        vad: f('silero_vad.onnx'),
        speaker: f('titanet.onnx'),
      );

  test('Parakeet transcribes real speech', () {
    final audio = readWavAs16k(f('test.wav'));
    final asr = SherpaParakeetAsr(paths());
    final watch = Stopwatch()..start();
    final text = asr.transcribe(audio, 16000);
    // ignore: avoid_print
    print('ASR (${(audio.length / 16000).toStringAsFixed(1)}s audio, ${watch.elapsedMilliseconds}ms): $text');
    expect(text.split(' ').length, greaterThan(3));
    asr.dispose();
  }, skip: skip);

  test('Silero VAD finds speech and ignores silence', () {
    final audio = readWavAs16k(f('test.wav'));
    final vad = SherpaVad(modelPath: f('silero_vad.onnx'));
    for (var i = 0; i < audio.length; i += 512) {
      vad.accept(Float32List.sublistView(audio, i, i + 512 > audio.length ? audio.length : i + 512));
    }
    final speech = [...vad.takeSegments(), ...vad.flush()];
    expect(speech, isNotEmpty);
    vad.reset();
    vad.accept(Float32List(16000 * 3));
    expect([...vad.takeSegments(), ...vad.flush()], isEmpty);
    vad.dispose();
  }, skip: skip);

  test('TitaNet voiceprints: same speaker similar, noise dissimilar', () {
    final audio = readWavAs16k(f('test.wav'));
    final emb = SherpaSpeakerEmbedder(f('titanet.onnx'));
    final half = audio.length ~/ 2;
    final a = emb.embed(Float32List.sublistView(audio, 0, half), 16000);
    final b = emb.embed(Float32List.sublistView(audio, half), 16000);
    final noise = Float32List.fromList(List.generate(half, (i) => ((i * 7919) % 200 - 100) / 1000));
    final n = emb.embed(noise, 16000);
    final same = cosine(a, b);
    final diff = cosine(a, n);
    // ignore: avoid_print
    print('dim=${emb.dimension} same-speaker=$same noise=$diff');
    expect(same, greaterThan(0.5));
    expect(same, greaterThan(diff + 0.2));
    emb.dispose();
  }, skip: skip);

  test('Full pipeline with real models stores a named, transcribed segment', () {
    final db = AppDatabase.inMemory();
    final speakers = SpeakerRepository(db);
    final transcripts = TranscriptRepository(db);
    final audio = readWavAs16k(f('test.wav'));
    final embedder = SherpaSpeakerEmbedder(f('titanet.onnx'));
    speakers.create(name: 'Reader', embeddingModel: embedder.modelId, samples: [embedder.embed(audio, 16000)]);

    final pipeline = ListeningPipeline(
      vad: SherpaVad(modelPath: f('silero_vad.onnx')),
      asr: SherpaParakeetAsr(paths()),
      embedder: embedder,
      identifier: SpeakerIdentifier(
        profiles: speakers.profiles(),
        newClusterId: SpeakerRepository.newId,
        nextGuestLabel: speakers.nextGuestLabel,
      ),
      transcripts: transcripts,
      speakers: speakers,
    );
    // Feed as 16-bit PCM in 100 ms chunks, like the microphone does.
    final pcm = ByteData(audio.length * 2);
    for (var i = 0; i < audio.length; i++) {
      pcm.setInt16(i * 2, (audio[i] * 32767).round().clamp(-32768, 32767), Endian.little);
    }
    final bytes = pcm.buffer.asUint8List();
    for (var i = 0; i < bytes.length; i += 3200) {
      pipeline.addPcm16(Uint8List.sublistView(bytes, i, i + 3200 > bytes.length ? bytes.length : i + 3200));
    }
    pipeline.addSamples(Float32List(16000)); // trailing silence ends the utterance
    pipeline.flush();
    final saved = transcripts.recent();
    // ignore: avoid_print
    print(saved.map((s) => '${s.speakerLabel} (${s.score?.toStringAsFixed(2)}): ${s.text}').join('\n'));
    expect(saved, isNotEmpty);
    expect(saved.where((s) => s.speakerName == 'Reader'), isNotEmpty);
    pipeline.dispose();
    db.close();
  }, skip: skip);
}
