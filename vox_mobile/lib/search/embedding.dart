import 'dart:math' as math;
import 'dart:typed_data';

/// What a text is embedded as. EmbeddingGemma was trained with a different
/// prompt for questions and for the texts they should find.
enum EmbedTask { query, document }

/// EmbeddingGemma's retrieval prompts (from its model card).
abstract final class EmbeddingPrompts {
  static const String query = 'task: search result | query: ';
  static const String document = 'title: none | text: ';

  static String of(EmbedTask task) => task == EmbedTask.query ? query : document;
}

/// Turns texts into unit-length vectors (implemented with EmbeddingGemma).
/// Synchronous and expensive: run it off the UI isolate.
abstract interface class TextEmbedder {
  List<Float32List> embed(List<String> texts, {required EmbedTask task});

  void dispose();
}

/// A [TextEmbedder] that runs elsewhere (a worker isolate on the phone).
abstract interface class AsyncEmbedder {
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task});

  Future<void> close();
}

/// Runs a [TextEmbedder] in place (tests, or code already off the UI isolate).
class InlineEmbedder implements AsyncEmbedder {
  InlineEmbedder(this._embedder);

  final TextEmbedder _embedder;

  @override
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task}) async =>
      _embedder.embed(texts, task: task);

  @override
  Future<void> close() async => _embedder.dispose();
}

/// How vectors are stored: EmbeddingGemma's 768 dimensions cut to the first
/// [dims] (it is trained so a prefix still works, "Matryoshka" style) and
/// re-normalised, then stored as signed bytes with one scale per vector.
/// 260 bytes a line instead of 3 KB; scores stay within ~1% of full precision.
abstract final class EmbeddingCodec {
  static const int dims = 256;

  /// The first [dims] values of [full], at unit length.
  static Float32List shorten(Float32List full) {
    final n = math.min(dims, full.length);
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      sum += full[i] * full[i];
    }
    final norm = math.sqrt(sum);
    final out = Float32List(dims);
    if (norm == 0) return out;
    for (var i = 0; i < n; i++) {
      out[i] = full[i] / norm;
    }
    return out;
  }

  /// [v] (already [shorten]ed) as bytes plus the scale that turns them back.
  static ({Int8List bytes, double scale}) quantize(Float32List v) {
    var peak = 0.0;
    for (final x in v) {
      peak = math.max(peak, x.abs());
    }
    final out = Int8List(v.length);
    if (peak == 0) return (bytes: out, scale: 0);
    final scale = peak / 127;
    for (var i = 0; i < v.length; i++) {
      out[i] = (v[i] / scale).round().clamp(-127, 127);
    }
    return (bytes: out, scale: scale);
  }

  /// Cosine similarity between a shortened [query] and the stored vector at
  /// [offset] in [data] (both unit length, so the dot product).
  static double score(Float32List query, Int8List data, int offset, double scale) {
    var dot = 0.0;
    for (var i = 0; i < dims; i++) {
      dot += query[i] * data[offset + i];
    }
    return dot * scale;
  }
}
