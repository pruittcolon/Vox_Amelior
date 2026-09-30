import 'package:vox_amelior_mobile/assistant/time_window.dart';
import 'package:vox_amelior_mobile/data/models.dart';

/// What kind of answer a question needs.
enum QueryIntent {
  /// A specific fact ("when is the plumber coming?") — search for it.
  lookup,

  /// A broad view of a period ("what did we talk about last week?") —
  /// read across the whole timeline.
  overview,
}

/// A question broken into things the archive can filter on.
class ParsedQuery {
  const ParsedQuery({required this.keywords, this.speaker, this.window, this.intent = QueryIntent.lookup});

  final List<String> keywords;
  final SpeakerProfile? speaker;
  final TimeWindow? window;
  final QueryIntent intent;

  DateTime? get from => window?.from;
  DateTime? get to => window?.to;
  String? get timeLabel => window?.label;
}

const Set<String> _stopWords = {
  'a', 'about', 'after', 'again', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'but', 'by', 'can', 'could', 'did', 'do', 'does',
  'doing', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'here', 'hers', 'him',
  'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or',
  'our', 'ours', 'said', 'say', 'says', 'she', 'so', 'some', 'tell', 'that', 'the', 'their',
  'them', 'then', 'there', 'they', 'this', 'to', 'told', 'up', 'us', 'was', 'we', 'were', 'what',
  'when', 'where', 'which', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
  'remind', 'remember', 'mention', 'mentioned', 'talk', 'talked', 'talking', 'discuss', 'discussed',
  'conversation', 'conversations', 'anything', 'something', 'thing', 'things', 'everything',
  'vox', 'hey', 'please', 'happen', 'happened', 'going', 'ago', 'last', 'past', 'week',
  'weeks', 'day', 'days', 'month', 'today', 'yesterday', 'earlier', 'recently', 'lately',
  'summarize', 'summarise', 'summary', 'recap', 'overview', 'highlights', 'catch', 'give', 'let',
  'know', 'lot', 'much', 'many', 'go', 'went', 'over', 'out', 'one', 'kind',
};

/// Explicit requests for a broad view of a period.
final RegExp _overviewCue = RegExp(r'\b(summari[sz]e|summary|recap|overview|highlights|catch me up|digest)\b');

/// Turns natural questions into search filters, without needing the LLM.
class QueryParser {
  QueryParser({required this.people, this.timeParser = const TimeWindowParser()});

  final List<SpeakerProfile> people;
  final TimeWindowParser timeParser;

  ParsedQuery parse(String question, {required DateTime now}) {
    var lower = question.toLowerCase();

    final time = timeParser.parse(lower, now);
    if (time != null) lower = lower.replaceFirst(time.$2, ' ');

    SpeakerProfile? speaker;
    for (final p in people) {
      final name = p.name.toLowerCase();
      if (RegExp("\\b${RegExp.escape(name)}(?:'s)?\\b").hasMatch(lower)) {
        speaker = p;
        break;
      }
    }

    final speakerWords = <String>{
      for (final p in people) ...p.name.toLowerCase().split(RegExp(r'\s+')),
    };
    final keywords = RegExp(r'[\p{L}\p{N}]+', unicode: true)
        .allMatches(lower)
        .map((m) => m.group(0)!)
        .where((w) => w.length > 1 && !_stopWords.contains(w) && !speakerWords.contains(w))
        .toSet()
        .toList();

    final broad = _overviewCue.hasMatch(question.toLowerCase());
    final intent = time != null && (keywords.isEmpty || broad) ? QueryIntent.overview : QueryIntent.lookup;
    return ParsedQuery(keywords: keywords, speaker: speaker, window: time?.$1, intent: intent);
  }
}
