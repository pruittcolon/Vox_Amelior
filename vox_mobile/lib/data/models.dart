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

/// A voice Vox heard but cannot name yet, with what you need to recognise
/// it: a few of its clearest lines and the people it was heard with.
class VoiceToName {
  const VoiceToName({
    required this.clusterId,
    required this.label,
    required this.lines,
    required this.lastHeard,
    this.samples = const [],
    this.heardWith = const [],
  });

  final String clusterId;

  /// "Guest 3".
  final String label;

  /// Lines it said that still carry no name.
  final int lines;
  final DateTime lastHeard;
  final List<SegmentView> samples;

  /// Named people in the same conversations, most lines first.
  final List<String> heardWith;
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
    this.emotion,
    this.sound,
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

  /// Tone of voice heard in this line ([Tone.names]), when the tone model ran.
  final String? emotion;

  /// Sound heard in this line besides speech ([Tone.sounds]), e.g. 'laughter'.
  final String? sound;

  DateTime get endedAt => startedAt.add(duration);

  /// How the line sounded, for Gemma: e.g. ['angry', 'laughing']. Neutral
  /// and lines without a tone give nothing.
  List<String> get toneNotes => [
        if (emotion != null && emotion != 'neutral') emotion!,
        if (sound != null) sound == 'laughter' ? 'laughing' : '$sound in background',
      ];

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
    this.emotions = const {},
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

  /// Only lines said in one of these tones ([Tone.names]; empty: any).
  final Set<String> emotions;
  final DateTime? from;
  final DateTime? to;
  final int limit;
}

/// Tones of voice and sounds the tone model reports, with how they are shown.
class Tone {
  const Tone._();

  /// Emotions, in the order they are offered as filters.
  static const List<String> names = ['angry', 'sad', 'happy', 'surprised', 'fearful', 'disgusted', 'neutral'];

  /// Sounds other than speech.
  static const List<String> sounds = ['laughter', 'music', 'applause', 'crying', 'coughing', 'sneezing'];

  static const Map<String, String> _emoji = {
    'angry': '😠',
    'sad': '😢',
    'happy': '😊',
    'surprised': '😮',
    'fearful': '😨',
    'disgusted': '🤢',
    'neutral': '😐',
    'laughter': '😂',
    'music': '🎵',
    'applause': '👏',
    'crying': '😭',
    'coughing': '🤧',
    'sneezing': '🤧',
  };

  static String emoji(String tone) => _emoji[tone] ?? '•';

  /// "Angry" for 'angry'.
  static String label(String tone) => tone.isEmpty ? tone : tone[0].toUpperCase() + tone.substring(1);

  /// Tones that suggest tension (used for "fights" filters and summaries).
  static const Set<String> tense = {'angry', 'disgusted', 'fearful', 'sad'};

  /// Reads a SenseVoice tag such as `<|ANGRY|>` or `<|Laughter|>` into
  /// [names] / [sounds] terms; null for unknown, neutral speech or empty.
  static String? fromEmotionTag(String tag) {
    final t = _bare(tag);
    return switch (t) {
      'happy' || 'angry' || 'sad' || 'neutral' || 'fearful' || 'disgusted' || 'surprised' => t,
      _ => null,
    };
  }

  static String? fromEventTag(String tag) => switch (_bare(tag)) {
        'laughter' => 'laughter',
        'bgm' => 'music',
        'applause' => 'applause',
        'cry' => 'crying',
        'cough' => 'coughing',
        'sneeze' => 'sneezing',
        _ => null, // 'speech', 'breath', unknown
      };

  static String _bare(String tag) => tag.replaceAll(RegExp(r'[<|>]'), '').trim().toLowerCase();
}

/// How each tone was spread over some lines (e.g. one conversation).
class MoodCount {
  const MoodCount(this.counts);

  /// Tone → number of lines (only tones that occur; neutral included).
  final Map<String, int> counts;

  int get total => counts.values.fold(0, (a, b) => a + b);

  /// Non-neutral tones, most frequent first.
  List<MapEntry<String, int>> get notable =>
      counts.entries.where((e) => e.key != 'neutral').toList()..sort((a, b) => b.value.compareTo(a.value));

  bool get isEmpty => counts.isEmpty;
}
