import 'dart:typed_data';

/// An enrolled person, represented by the mean of their voice embeddings.
class SpeakerProfile {
  const SpeakerProfile({
    required this.id,
    required this.name,
    required this.embeddingModel,
    required this.centroid,
    required this.sampleCount,
    required this.createdAt,
  });

  final String id;
  final String name;

  /// Identifies which embedding model produced [centroid]; vectors from
  /// different models are not comparable.
  final String embeddingModel;
  final Float32List centroid;
  final int sampleCount;
  final DateTime createdAt;
}

/// A not-yet-named voice ("Guest 1") discovered while listening.
class UnknownCluster {
  UnknownCluster({
    required this.id,
    required this.label,
    required this.centroid,
    required this.count,
    required this.updatedAt,
  });

  final String id;
  final String label;
  Float32List centroid;
  int count;
  DateTime updatedAt;
}

/// A transcribed utterance with its resolved speaker, ready for display.
class SegmentView {
  const SegmentView({
    required this.id,
    required this.conversationId,
    required this.startedAt,
    required this.duration,
    required this.text,
    this.speakerId,
    this.speakerName,
    this.clusterId,
    this.clusterLabel,
    this.score,
  });

  final int id;
  final int conversationId;
  final DateTime startedAt;
  final Duration duration;
  final String text;
  final String? speakerId;
  final String? speakerName;
  final String? clusterId;
  final String? clusterLabel;
  final double? score;

  DateTime get endedAt => startedAt.add(duration);

  /// Best human-readable speaker label.
  String get speakerLabel => speakerName ?? clusterLabel ?? 'Unknown';

  bool get isKnownSpeaker => speakerId != null;
}

class ConversationSummary {
  const ConversationSummary({
    required this.id,
    required this.startedAt,
    required this.endedAt,
    required this.segmentCount,
    required this.preview,
  });

  final int id;
  final DateTime startedAt;
  final DateTime endedAt;
  final int segmentCount;
  final String preview;
}

/// Filters for searching the transcript archive.
class SegmentQuery {
  const SegmentQuery({
    this.keywords = const [],
    this.speakerId,
    this.from,
    this.to,
    this.limit = 50,
  });

  final List<String> keywords;
  final String? speakerId;
  final DateTime? from;
  final DateTime? to;
  final int limit;
}
