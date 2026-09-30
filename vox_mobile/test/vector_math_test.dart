import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

void main() {
  test('cosine of identical vectors is 1, opposite is -1, orthogonal is 0', () {
    final a = Float32List.fromList([1, 2, 3]);
    expect(cosine(a, a), closeTo(1, 1e-6));
    expect(cosine(a, Float32List.fromList([-1, -2, -3])), closeTo(-1, 1e-6));
    expect(cosine(Float32List.fromList([1, 0]), Float32List.fromList([0, 1])), closeTo(0, 1e-9));
  });

  test('cosine is 0 for mismatched or zero vectors', () {
    expect(cosine(Float32List(2), Float32List(3)), 0);
    expect(cosine(Float32List(3), Float32List.fromList([1, 2, 3])), 0);
  });

  test('l2Normalize gives unit length and handles zero', () {
    final n = l2Normalize(Float32List.fromList([3, 4]));
    expect(n[0], closeTo(0.6, 1e-6));
    expect(n[1], closeTo(0.8, 1e-6));
    expect(l2Normalize(Float32List(3)), everyElement(0));
  });

  test('meanEmbedding averages normalised vectors regardless of magnitude', () {
    final m = meanEmbedding([Float32List.fromList([10, 0]), Float32List.fromList([0, 0.1])]);
    expect(m[0], closeTo(m[1], 1e-6));
    expect(() => meanEmbedding([]), throwsArgumentError);
    expect(() => meanEmbedding([Float32List(2), Float32List(3)]), throwsArgumentError);
  });

  test('blob round trip preserves values', () {
    final v = Float32List.fromList([0.5, -1.25, 3]);
    expect(blobToFloats(floatsToBlob(v)), v);
  });
}
