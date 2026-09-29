import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

/// Tunable recognition thresholds (cosine similarity).
class IdentifierConfig {
  const IdentifierConfig({
    this.threshold = 0.55,
    this.margin = 0.04,
    this.clusterThreshold = 0.6,
    this.minEmbeddingSeconds = 1.0,
    this.maxClusters = 12,
    this.maxClusterWeight = 50,
  });

  /// Minimum similarity to an enrolled person to accept the match.
  final double threshold;

  /// The best person must beat the runner-up by this much; otherwise the
  /// voice is treated as ambiguous and falls through to "guest" handling.
  final double margin;

  /// Similarity needed to join an existing guest cluster.
  final double clusterThreshold;

  /// Shorter utterances give unreliable voiceprints and stay unattributed.
  final double minEmbeddingSeconds;

  final int maxClusters;

  /// Guest centroids stop moving after this many samples (keeps them stable).
  final int maxClusterWeight;

  IdentifierConfig copyWith({
    double? threshold,
    double? margin,
    double? clusterThreshold,
    double? minEmbeddingSeconds,
  }) =>
      IdentifierConfig(
        threshold: threshold ?? this.threshold,
        margin: margin ?? this.margin,
        clusterThreshold: clusterThreshold ?? this.clusterThreshold,
        minEmbeddingSeconds: minEmbeddingSeconds ?? this.minEmbeddingSeconds,
        maxClusters: maxClusters,
        maxClusterWeight: maxClusterWeight,
      );
}

/// Outcome of matching one utterance's voiceprint.
class SpeakerMatch {
  const SpeakerMatch.person(String id, this.score)
      : speakerId = id,
        cluster = null,
        createdCluster = false;

  const SpeakerMatch.guest(UnknownCluster this.cluster, this.score, {this.createdCluster = false})
      : speakerId = null;

  const SpeakerMatch.none()
      : speakerId = null,
        cluster = null,
        score = null,
        createdCluster = false;

  final String? speakerId;
  final UnknownCluster? cluster;
  final double? score;
  final bool createdCluster;

  bool get isKnown => speakerId != null;
}

/// Decides who is speaking: an enrolled person, a recurring guest, or nobody.
///
/// Holds enrolled profiles and guest clusters in memory; the caller persists
/// any [SpeakerMatch.cluster] it gets back.
class SpeakerIdentifier {
  SpeakerIdentifier({
    required List<SpeakerProfile> profiles,
    List<UnknownCluster> clusters = const [],
    this.config = const IdentifierConfig(),
    required this.newClusterId,
    required this.nextGuestLabel,
    this.clock = systemClock,
  })  : _profiles = List.of(profiles),
        _clusters = List.of(clusters);

  IdentifierConfig config;
  List<SpeakerProfile> _profiles;
  List<UnknownCluster> _clusters;

  /// Produces a unique id / display label for each newly discovered voice.
  final String Function() newClusterId;
  final String Function() nextGuestLabel;
  final Clock clock;

  /// Replaces the enrolled people (call after enrollment changes).
  void updateProfiles(List<SpeakerProfile> profiles, [List<UnknownCluster>? clusters]) {
    _profiles = List.of(profiles);
    if (clusters != null) _clusters = List.of(clusters);
  }

  List<UnknownCluster> get clusters => List.unmodifiable(_clusters);

  /// Scores [embedding] against every enrolled person, best first.
  /// Useful for the "test my voice" screen.
  List<MapEntry<SpeakerProfile, double>> rank(Float32List embedding) {
    final scored = [for (final p in _profiles) MapEntry(p, cosine(embedding, p.centroid))]
      ..sort((a, b) => b.value.compareTo(a.value));
    return scored;
  }

  SpeakerMatch identify(Float32List embedding, {required double seconds}) {
    if (seconds < config.minEmbeddingSeconds) return const SpeakerMatch.none();

    final ranked = rank(embedding);
    if (ranked.isNotEmpty) {
      final best = ranked.first;
      final second = ranked.length > 1 ? ranked[1].value : -1.0;
      if (best.value >= config.threshold && best.value - second >= config.margin) {
        return SpeakerMatch.person(best.key.id, best.value);
      }
    }
    return _matchGuest(embedding);
  }

  SpeakerMatch _matchGuest(Float32List embedding) {
    UnknownCluster? bestCluster;
    var bestScore = -1.0;
    for (final c in _clusters) {
      final s = cosine(embedding, c.centroid);
      if (s > bestScore) {
        bestScore = s;
        bestCluster = c;
      }
    }
    if (bestCluster != null && bestScore >= config.clusterThreshold) {
      _absorb(bestCluster, embedding);
      return SpeakerMatch.guest(bestCluster, bestScore);
    }
    if (_clusters.length >= config.maxClusters) {
      // Too many strangers: don't invent more, attach to the closest one.
      if (bestCluster != null) return SpeakerMatch.guest(bestCluster, bestScore);
      return const SpeakerMatch.none();
    }
    final fresh = UnknownCluster(
      id: newClusterId(),
      label: nextGuestLabel(),
      centroid: l2Normalize(embedding),
      count: 1,
      updatedAt: clock(),
    );
    _clusters.add(fresh);
    return SpeakerMatch.guest(fresh, 1, createdCluster: true);
  }

  void _absorb(UnknownCluster c, Float32List embedding) {
    final w = c.count.clamp(1, config.maxClusterWeight);
    final e = l2Normalize(embedding);
    final merged = Float32List(c.centroid.length);
    for (var i = 0; i < merged.length; i++) {
      merged[i] = c.centroid[i] * w + e[i];
    }
    c
      ..centroid = l2Normalize(merged)
      ..count = c.count + 1
      ..updatedAt = clock();
  }
}
