// Runs the real EmbeddingGemma model through Vox's own ONNX Runtime bindings
// and checks the Dart pipeline against Hugging Face's reference
// (tool/embedding_reference.py): the same token ids and the same vectors.
// Skipped unless VOX_EMBEDDER_DIR (the graph, its .onnx_data and
// tokenizer.model), VOX_EMBEDDER_REFERENCE (the reference JSON) and
// SHERPA_LIB_DIR (the folder holding libonnxruntime.so) are set.
// VOX_EMBEDDER_GRAPH picks the size (model_quantized.onnx by default,
// model_q4.onnx or model.onnx). CI runs every size in
// .github/workflows/verify-embeddings.yml.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/native/embedding_gemma.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

void main() {
  final env = Platform.environment;
  final dir = env['VOX_EMBEDDER_DIR'];
  final reference = env['VOX_EMBEDDER_REFERENCE'];
  final skip = dir == null || reference == null || env['SHERPA_LIB_DIR'] == null
      ? 'Set VOX_EMBEDDER_DIR, VOX_EMBEDDER_REFERENCE and SHERPA_LIB_DIR to run the real embedder'
      : null;

  late EmbeddingGemmaOnnx embedder;
  late List<({EmbedTask task, String text, List<int> ids, Float32List vec})> cases;

  setUpAll(() {
    if (skip != null) return;
    embedder = EmbeddingGemmaOnnx(
      modelPath: '$dir/${env['VOX_EMBEDDER_GRAPH'] ?? 'model_quantized.onnx'}',
      tokenizerPath: '$dir/tokenizer.model',
      libraryDir: env['SHERPA_LIB_DIR'],
    );
    final json = jsonDecode(File(reference!).readAsStringSync()) as Map<String, Object?>;
    cases = [
      for (final c in (json['cases']! as List).cast<Map<String, Object?>>())
        (
          task: c['task'] == 'query' ? EmbedTask.query : EmbedTask.document,
          text: c['text']! as String,
          ids: (c['ids']! as List).cast<int>(),
          vec: Float32List.fromList([for (final x in c['vec']! as List) (x as num).toDouble()]),
        ),
    ];
  });
  tearDownAll(() {
    if (skip == null) embedder.dispose();
  });

  test('token ids are exactly what the Hugging Face tokenizer gives', () {
    for (final c in cases) {
      expect(embedder.tokenize(c.text, c.task), c.ids, reason: '${c.task.name}: ${c.text}');
    }
  }, skip: skip);

  test('vectors match the reference', () {
    for (final c in cases) {
      final v = embedder.embed([c.text], task: c.task).single;
      expect(v, hasLength(768));
      final sim = cosine(v, c.vec);
      // ignore: avoid_print
      print('reference match ${sim.toStringAsFixed(5)}  ${c.text.replaceAll('\n', ' ')}');
      expect(sim, greaterThan(0.999), reason: c.text);
    }
  }, skip: skip);

  test('related lines score higher than unrelated ones, also after compact storage', () {
    double sim(String q, String d) {
      final qv = EmbeddingCodec.shorten(embedder.embed([q], task: EmbedTask.query).single);
      final stored = EmbeddingCodec.quantize(EmbeddingCodec.shorten(embedder.embed([d], task: EmbedTask.document).single));
      return EmbeddingCodec.score(qv, stored.bytes, 0, stored.scale);
    }

    const pairs = [
      ('money worries', 'We cannot pay the electric bill this month', 'The puppy chewed my shoe again'),
      ('when is the dentist?', 'The dentist appointment is on Tuesday at 9:30.', 'Rent is due on Friday too'),
      ('pet causing trouble', 'The puppy chewed my shoe again', 'We cannot pay the electric bill this month'),
    ];
    for (final (q, related, unrelated) in pairs) {
      final a = sim(q, related), b = sim(q, unrelated);
      // ignore: avoid_print
      print('"$q": related ${a.toStringAsFixed(3)}, unrelated ${b.toStringAsFixed(3)}');
      expect(a, greaterThan(b + 0.05), reason: q);
    }
  }, skip: skip);

  test('speed on this machine', () {
    final watch = Stopwatch()..start();
    for (var i = 0; i < 20; i++) {
      embedder.embed(['Line number $i about the garden, the bills and dinner plans'], task: EmbedTask.document);
    }
    // ignore: avoid_print
    print('${(watch.elapsedMilliseconds / 20).toStringAsFixed(1)} ms per line');
  }, skip: skip);
}
