import 'package:vox_amelior_mobile/data/models.dart';

/// A question broken into things the archive can filter on.
class ParsedQuery {
  const ParsedQuery({
    required this.keywords,
    this.speaker,
    this.from,
    this.to,
    this.timeLabel,
  });

  final List<String> keywords;
  final SpeakerProfile? speaker;
  final DateTime? from;
  final DateTime? to;

  /// Phrase that set the time window, for the prompt ("yesterday").
  final String? timeLabel;
}

const Set<String> _stopWords = {
  'a', 'about', 'after', 'again', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'but', 'by', 'can', 'could', 'did', 'do', 'does',
  'doing', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'here', 'hers', 'him',
  'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or',
  'our', 'ours', 'said', 'say', 'says', 'she', 'so', 'some', 'tell', 'that', 'the', 'their',
  'them', 'then', 'there', 'they', 'this', 'to', 'told', 'up', 'us', 'was', 'we', 'were', 'what',
  'when', 'where', 'which', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
  'remind', 'remember', 'mention', 'mentioned', 'talk', 'talked', 'discuss', 'discussed',
  'conversation', 'today', 'yesterday', 'tonight', 'morning', 'afternoon', 'evening', 'last',
  'week', 'month', 'night', 'ago', 'hour', 'hours', 'day', 'days', 'earlier', 'recently',
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
  'vox', 'hey', 'please', 'anything', 'something', 'thing', 'things',
};

/// Turns natural questions into search filters, without needing the LLM.
class QueryParser {
  QueryParser({required this.people});

  final List<SpeakerProfile> people;

  ParsedQuery parse(String question, {required DateTime now}) {
    final lower = question.toLowerCase();

    SpeakerProfile? speaker;
    for (final p in people) {
      final name = p.name.toLowerCase();
      if (RegExp('\\b${RegExp.escape(name)}(?:\'s)?\\b').hasMatch(lower)) {
        speaker = p;
        break;
      }
    }

    final window = _timeWindow(lower, now);

    final speakerWords = <String>{
      for (final p in people) ...p.name.toLowerCase().split(RegExp(r'\s+')),
    };
    final words = RegExp(r'[\p{L}\p{N}]+', unicode: true)
        .allMatches(lower)
        .map((m) => m.group(0)!)
        .where((w) => w.length > 1 && !_stopWords.contains(w) && !speakerWords.contains(w))
        .toList();

    return ParsedQuery(
      keywords: words.toSet().toList(),
      speaker: speaker,
      from: window?.from,
      to: window?.to,
      timeLabel: window?.label,
    );
  }

  _Window? _timeWindow(String q, DateTime now) {
    final startOfToday = DateTime(now.year, now.month, now.day);
    if (q.contains('yesterday')) {
      final y = startOfToday.subtract(const Duration(days: 1));
      return _Window(y, startOfToday, 'yesterday');
    }
    if (RegExp(r'\blast night\b').hasMatch(q)) {
      final y = startOfToday.subtract(const Duration(days: 1));
      return _Window(y.add(const Duration(hours: 17)), startOfToday.add(const Duration(hours: 5)), 'last night');
    }
    if (RegExp(r'\b(this morning)\b').hasMatch(q)) {
      return _Window(startOfToday, startOfToday.add(const Duration(hours: 12)), 'this morning');
    }
    if (RegExp(r'\b(this afternoon)\b').hasMatch(q)) {
      return _Window(startOfToday.add(const Duration(hours: 12)), startOfToday.add(const Duration(hours: 18)), 'this afternoon');
    }
    if (RegExp(r'\b(tonight|this evening)\b').hasMatch(q)) {
      return _Window(startOfToday.add(const Duration(hours: 17)), startOfToday.add(const Duration(days: 1)), 'this evening');
    }
    if (RegExp(r'\btoday\b').hasMatch(q)) {
      return _Window(startOfToday, startOfToday.add(const Duration(days: 1)), 'today');
    }
    final agoHours = RegExp(r'\b(?:last|past)\s+(\d+)\s+hours?\b').firstMatch(q);
    if (agoHours != null) {
      final h = int.parse(agoHours.group(1)!);
      return _Window(now.subtract(Duration(hours: h)), now.add(const Duration(minutes: 1)), 'the last $h hours');
    }
    if (RegExp(r'\b(last|past) hour\b').hasMatch(q)) {
      return _Window(now.subtract(const Duration(hours: 1)), now.add(const Duration(minutes: 1)), 'the last hour');
    }
    final agoDays = RegExp(r'\b(?:last|past)\s+(\d+)\s+days?\b').firstMatch(q);
    if (agoDays != null) {
      final d = int.parse(agoDays.group(1)!);
      return _Window(startOfToday.subtract(Duration(days: d)), now.add(const Duration(minutes: 1)), 'the last $d days');
    }
    if (RegExp(r'\blast week\b').hasMatch(q) || RegExp(r'\bpast week\b').hasMatch(q)) {
      return _Window(startOfToday.subtract(const Duration(days: 7)), now.add(const Duration(minutes: 1)), 'the last week');
    }
    if (RegExp(r'\bthis week\b').hasMatch(q)) {
      final monday = startOfToday.subtract(Duration(days: startOfToday.weekday - 1));
      return _Window(monday, now.add(const Duration(minutes: 1)), 'this week');
    }
    if (RegExp(r'\blast month\b').hasMatch(q) || RegExp(r'\bpast month\b').hasMatch(q)) {
      return _Window(startOfToday.subtract(const Duration(days: 30)), now.add(const Duration(minutes: 1)), 'the last month');
    }
    const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
    for (var i = 0; i < days.length; i++) {
      if (RegExp('\\b(?:on |last )?${days[i]}\\b').hasMatch(q)) {
        var diff = startOfToday.weekday - (i + 1);
        if (diff <= 0) diff += 7;
        final day = startOfToday.subtract(Duration(days: diff));
        return _Window(day, day.add(const Duration(days: 1)), days[i][0].toUpperCase() + days[i].substring(1));
      }
    }
    return null;
  }
}

class _Window {
  const _Window(this.from, this.to, this.label);
  final DateTime from;
  final DateTime to;
  final String label;
}
