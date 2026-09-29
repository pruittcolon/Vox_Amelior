import 'dart:math' as math;
import 'dart:typed_data';

/// Returns a unit-length copy of [v] (or a zero vector if [v] is all zeros).
Float32List l2Normalize(Float32List v) {
  var sum = 0.0;
  for (final x in v) {
    sum += x * x;
  }
  final norm = math.sqrt(sum);
  final out = Float32List(v.length);
  if (norm == 0) return out;
  for (var i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/// Cosine similarity in [-1, 1]. Returns 0 for empty or mismatched vectors.
double cosine(Float32List a, Float32List b) {
  if (a.length != b.length || a.isEmpty) return 0;
  var dot = 0.0;
  var na = 0.0;
  var nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na == 0 || nb == 0) return 0;
  return dot / (math.sqrt(na) * math.sqrt(nb));
}

/// Mean of unit-normalised [vectors], re-normalised. Throws on empty input.
Float32List meanEmbedding(List<Float32List> vectors) {
  if (vectors.isEmpty) throw ArgumentError('No vectors to average');
  final dim = vectors.first.length;
  final acc = Float32List(dim);
  for (final raw in vectors) {
    if (raw.length != dim) {
      throw ArgumentError('Embedding size mismatch: ${raw.length} vs $dim');
    }
    final v = l2Normalize(raw);
    for (var i = 0; i < dim; i++) {
      acc[i] += v[i];
    }
  }
  return l2Normalize(acc);
}

/// Serialises floats for a SQLite BLOB (little-endian on every Android ABI).
Uint8List floatsToBlob(Float32List v) =>
    Uint8List.fromList(v.buffer.asUint8List(v.offsetInBytes, v.lengthInBytes));

Float32List blobToFloats(Uint8List blob) {
  final copy = Uint8List.fromList(blob);
  return copy.buffer.asFloat32List(0, copy.lengthInBytes ~/ 4);
}
