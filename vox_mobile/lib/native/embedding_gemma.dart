import 'dart:ffi';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:vox_amelior_mobile/native/onnx_runtime.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

/// Google EmbeddingGemma 300M (the onnx-community int8 export) on the ONNX
/// Runtime that ships inside sherpa-onnx.
///
/// Tokens are `[BOS=2, ...SentencePiece(prompt + text), EOS=1]`, as in
/// EmbeddingGemma's own pipeline; getting this wrong silently changes every
/// vector, which is why CI compares these ids and vectors with the Hugging
/// Face reference implementation. The graph's `sentence_embedding` output is
/// already mean-pooled and projected; it is normalised again here.
class EmbeddingGemmaOnnx implements TextEmbedder {
  EmbeddingGemmaOnnx({
    required String modelPath,
    required String tokenizerPath,
    int threads = 2,
    String? libraryDir,
    this.maxTokens = 512,
  }) : _ort = OnnxRuntime.load(libraryDir: libraryDir) {
    _session = _ort.createSession(modelPath, threads: threads);
    _tokenizer = SentencePieceTokenizer.fromModelFileSync(tokenizerPath, config: const SentencePieceConfig());
  }

  static const int bos = 2;
  static const int eos = 1;

  /// Longer texts are cut (lines are short; a whole speech rarely is).
  final int maxTokens;

  final OnnxRuntime _ort;
  late final Pointer<Void> _session;
  late final SentencePieceTokenizer _tokenizer;
  bool _disposed = false;

  /// The ids the model sees for [text] as [task].
  List<int> tokenize(String text, EmbedTask task) {
    final ids = _tokenizer.encode(EmbeddingPrompts.of(task) + text).ids;
    final body = ids.length > maxTokens - 2 ? ids.sublist(0, maxTokens - 2) : ids;
    return [bos, ...body, eos];
  }

  @override
  List<Float32List> embed(List<String> texts, {required EmbedTask task}) {
    if (_disposed) throw StateError('Embedder was disposed');
    // One text per run: no padding to get wrong, and lines are short.
    return [for (final t in texts) _one(tokenize(t, task))];
  }

  Float32List _one(List<int> ids) {
    final n = ids.length;
    final out = _ort.run(
      _session,
      inputs: {
        'input_ids': (data: Int64List.fromList(ids), shape: [1, n]),
        'attention_mask': (data: Int64List(n)..fillRange(0, n, 1), shape: [1, n]),
      },
      output: 'sentence_embedding',
    );
    return l2Normalize(out.data);
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _ort.releaseSession(_session);
  }
}
