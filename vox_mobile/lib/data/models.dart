import 'dart:typed_data';

/// An enrolled person, represented by the mean of their voice embeddings
/// plus a few "patterns" (averages of similar-sounding groups of samples).
class SpeakerProfile {
  const SpeakerProfile({
    required this.id,
    required this.name,
    required this.embeddingModel,
    required this.centroid,
    required this.sampleCount,
    required this.createdAt,
    this.patterns = const [],
    this.negatives = const [],
  });

  final String id;
  final String name;

  /// Identifies which embedding model produced [centroid]; vectors from
  /// different models are not comparable.
  final String embeddingModel;
  final Float32List centroid;
  final int sampleCount;
  final DateTime createdAt;

  /// Averages of groups of similar samples (up to 5); empty until the
  /// person has enough samples.
  final List<Float32List> patterns;

  /// Voices the user marked as "not this person".
  final List<Float32List> negatives;
}

/// A not-yet-named voice ("Guest 1") discovered while listening.
class UnknownCluster {
  UnknownCluster({
    required this.id,
    required this.label,
    required this.centroid,
    required this.count,
    required this.updatedAt,
    this.background = false,
  });

  final String id;
  final String label;
  Float32List centroid;
  int count;
  DateTime updatedAt;

  /// Marked as TV, radio or other background voice: hidden when reading back.
  bool background;
}

/// A transcribed utterance with its resolved speaker, ready for display.
/// Which pass produced a line event: [fast] right after the sentence (text
/// and a first voice match), [finished] after the chunk pass (cut at speaker
/// changes, final names), [both] when one pass did everything.
enum LineStage { fast, finished, both }

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
    this.overlap = false,
    this.background = false,
    this.importedLabel,
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

  /// Someone else was talking at the same time during this line.
  final bool overlap;

  /// Said by a voice marked as TV or background.
  final bool background;

  /// Speaker name of a line imported from a text export (no voiceprint).
  final String? importedLabel;

  DateTime get endedAt => startedAt.add(duration);

  /// Best human-readable speaker label.
  String get speakerLabel => speakerName ?? clusterLabel ?? importedLabel ?? 'Unknown';

  bool get isKnownSpeaker => speakerId != null;

  /// Identifies the voice for filtering: a person, a guest voice, an
  /// imported name, or 'unknown'.
  String get voiceKey => speakerId != null
      ? 'person:$speakerId'
      : clusterId != null
          ? 'guest:$clusterId'
          : importedLabel != null
              ? 'name:$importedLabel'
              : 'unknown';
}

class ConversationSummary {
  const ConversationSummary({
    required this.id,
    required this.startedAt,
    required this.endedAt,
    required this.segmentCount,
    required this.preview,
    this.participants = const [],
  });

  final int id;
  final DateTime startedAt;
  final DateTime endedAt;
  final int segmentCount;
  final String preview;

  /// Speaker labels in order of how much they spoke.
  final List<String> participants;

  Duration get duration => endedAt.difference(startedAt);
}

/// How much was said on one day.
class DaySummary {
  const DaySummary({required this.day, required this.conversations, required this.segments});

  final DateTime day;
  final int conversations;
  final int segments;
}

/// Filters for searching the transcript archive.
class SegmentQuery {
  const SegmentQuery({
    this.keywords = const [],
    this.speakerId,
    this.speakerIds = const {},
    this.includeBackground = false,
    this.from,
    this.to,
    this.limit = 50,
  });

  final List<String> keywords;
  final String? speakerId;

  /// Only lines said by any of these people (empty: anyone).
  final Set<String> speakerIds;

  /// Include lines from voices marked as TV or background.
  final bool includeBackground;
  final DateTime? from;
  final DateTime? to;
  final int limit;
}
