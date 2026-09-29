import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/speakers/vector_math.dart';

/// Splits one person's voice samples into a few groups of similar-sounding
/// samples ("close to the phone", "across the room", "with a cold") and
/// returns each group's average voice.
///
/// Deterministic, rebuilt from scratch whenever the samples change, so the
/// patterns can never drift. Every group needs at least [minGroup] samples;
/// with too few samples there are no patterns and recognition uses only the
/// person's overall average, exactly as before patterns existed.
List<Float32List> voicePatterns(
  List<Float32List> samples, {
  int maxPatterns = 5,
  int minGroup = 10,
  int iterations = 12,
}) {
  final kMax = math.min(maxPatterns, samples.length ~/ minGroup);
  if (kMax < 2) return const [];
  final xs = [for (final s in samples) l2Normalize(s)];
  // The most groups for which every group is big enough to trust.
  for (var k = kMax; k >= 2; k--) {
    final (centers, sizes) = _kMeans(xs, k, iterations);
    if (sizes.every((n) => n >= minGroup)) return centers;
  }
  return const [];
}

(List<Float32List>, List<int>) _kMeans(List<Float32List> xs, int k, int iterations) {
  // Start from the most typical sample, then repeatedly add the sample least
  // like any chosen start (spreads the groups over the different sounds).
  final mean = meanEmbedding(xs);
  var first = 0;
  var best = -2.0;
  for (var i = 0; i < xs.length; i++) {
    final s = _dot(xs[i], mean);
    if (s > best) {
      best = s;
      first = i;
    }
  }
  final centers = <Float32List>[Float32List.fromList(xs[first])];
  while (centers.length < k) {
    var pick = -1;
    var lowest = 2.0;
    for (var i = 0; i < xs.length; i++) {
      var nearest = -2.0;
      for (final c in centers) {
        nearest = math.max(nearest, _dot(xs[i], c));
      }
      if (nearest < lowest) {
        lowest = nearest;
        pick = i;
      }
    }
    if (pick < 0) break;
    centers.add(Float32List.fromList(xs[pick]));
  }

  // Spherical k-means.
  var assignment = List<int>.filled(xs.length, 0);
  for (var it = 0; it < iterations; it++) {
    final next = List<int>.filled(xs.length, 0);
    for (var i = 0; i < xs.length; i++) {
      var bestC = 0;
      var bestS = -2.0;
      for (var c = 0; c < centers.length; c++) {
        final s = _dot(xs[i], centers[c]);
        if (s > bestS) {
          bestS = s;
          bestC = c;
        }
      }
      next[i] = bestC;
    }
    for (var c = 0; c < centers.length; c++) {
      final members = [for (var i = 0; i < xs.length; i++) if (next[i] == c) xs[i]];
      if (members.isNotEmpty) centers[c] = meanEmbedding(members);
    }
    final settled = _same(assignment, next);
    assignment = next;
    if (settled && it > 0) break;
  }
  final sizes = List<int>.filled(centers.length, 0);
  for (final a in assignment) {
    sizes[a]++;
  }
  return (centers, sizes);
}

/// Packs patterns into one BLOB (they all have the same length).
Uint8List? patternsToBlob(List<Float32List> patterns) {
  if (patterns.isEmpty) return null;
  final dim = patterns.first.length;
  final all = Float32List(dim * patterns.length);
  for (var i = 0; i < patterns.length; i++) {
    all.setAll(i * dim, patterns[i]);
  }
  return floatsToBlob(all);
}

List<Float32List> patternsFromBlob(Uint8List? blob, int dim) {
  if (blob == null || blob.isEmpty || dim <= 0) return const [];
  final all = blobToFloats(blob);
  if (all.length % dim != 0) return const [];
  return [for (var i = 0; i < all.length; i += dim) Float32List.sublistView(all, i, i + dim)];
}

double _dot(Float32List a, Float32List b) {
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

bool _same(List<int> a, List<int> b) {
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
