/// A span of time a question refers to, e.g. "last week".
class TimeWindow {
  const TimeWindow(this.from, this.to, this.label);

  /// Inclusive start.
  final DateTime from;

  /// Exclusive end.
  final DateTime to;

  /// The phrase as the user said it ("last week").
  final String label;

  Duration get length => to.difference(from);

  /// "last week (Mon 21 Sep – Sun 27 Sep)"
  String describe() {
    final lastDay = to.subtract(const Duration(milliseconds: 1));
    final sameDay = from.year == lastDay.year && from.month == lastDay.month && from.day == lastDay.day;
    return sameDay ? '$label (${_day(from)})' : '$label (${_day(from)} – ${_day(lastDay)})';
  }

  static const _wd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  static const _mo = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  static String _day(DateTime d) => '${_wd[d.weekday - 1]} ${d.day} ${_mo[d.month - 1]}';

  @override
  String toString() => 'TimeWindow($label: $from → $to)';
}

const List<String> _weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
const List<String> _months = [
  'january', 'february', 'march', 'april', 'may', 'june',
  'july', 'august', 'september', 'october', 'november', 'december',
];
const Map<String, int> _numberWords = {
  'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
  'eight': 8, 'nine': 9, 'ten': 10, 'couple': 2, 'few': 3,
};

/// Finds the time span a question refers to. Weeks start on Monday.
///
/// Calendar phrases are calendar-exact: "last week" is the previous
/// Monday–Sunday, "this week" is Monday until now. Rolling phrases
/// ("the past week", "last 3 days") count back from now.
class TimeWindowParser {
  const TimeWindowParser();

  /// Returns the window and the matched text, or null if none is mentioned.
  (TimeWindow, String)? parse(String question, DateTime now) {
    final q = question.toLowerCase().replaceAll('\u2019', "'");
    final today = DateTime(now.year, now.month, now.day);
    final monday = today.subtract(Duration(days: today.weekday - 1));
    final soon = now.add(const Duration(minutes: 1));

    (TimeWindow, String) w(DateTime from, DateTime to, String label, String matched) =>
        (TimeWindow(from, to, label), matched);

    RegExpMatch? m;

    // Rolling windows: "past/last N hours|days|weeks|months", "in the last week".
    m = RegExp(r'\b(?:in the |over the |during the |for the )?(?:past|last|previous)\s+(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|couple of|few)\s+(hour|day|week|month)s?\b')
        .firstMatch(q);
    if (m != null) {
      final n = _number(m.group(1)!);
      final unit = m.group(2)!;
      final from = switch (unit) {
        'hour' => now.subtract(Duration(hours: n)),
        'day' => today.subtract(Duration(days: n - 1)),
        'week' => today.subtract(Duration(days: 7 * n - 1)),
        _ => DateTime(now.year, now.month - n, now.day),
      };
      return w(from, soon, 'the last $n ${unit}s', m.group(0)!);
    }
    m = RegExp(r'\b(?:in|over|during|for) the (?:past|last) (hour|day|week|month|year)\b').firstMatch(q) ??
        RegExp(r'\bthe past (hour|day|week|month|year)\b').firstMatch(q) ??
        RegExp(r'\b(?:past|last) (24 hours|7 days|30 days)\b').firstMatch(q);
    if (m != null) {
      final unit = m.group(1)!;
      final from = switch (unit) {
        'hour' => now.subtract(const Duration(hours: 1)),
        'day' || '24 hours' => now.subtract(const Duration(hours: 24)),
        'week' || '7 days' => today.subtract(const Duration(days: 6)),
        'month' || '30 days' => today.subtract(const Duration(days: 29)),
        _ => DateTime(now.year - 1, now.month, now.day),
      };
      return w(from, soon, 'the past $unit', m.group(0)!);
    }

    // "N days ago", "N weeks ago".
    m = RegExp(r'\b(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|couple of|few) (day|week|month)s? ago\b').firstMatch(q);
    if (m != null) {
      final n = _number(m.group(1)!);
      switch (m.group(2)) {
        case 'day':
          final d = today.subtract(Duration(days: n));
          return w(d, d.add(const Duration(days: 1)), '$n days ago', m.group(0)!);
        case 'week':
          final start = monday.subtract(Duration(days: 7 * n));
          return w(start, start.add(const Duration(days: 7)), '$n weeks ago', m.group(0)!);
        default:
          final start = DateTime(now.year, now.month - n);
          return w(start, DateTime(now.year, now.month - n + 1), '$n months ago', m.group(0)!);
      }
    }

    if ((m = RegExp(r'\bweek before last\b').firstMatch(q)) != null) {
      final start = monday.subtract(const Duration(days: 14));
      return w(start, start.add(const Duration(days: 7)), 'the week before last', m!.group(0)!);
    }
    if ((m = RegExp(r'\b(?:last|previous) week\b').firstMatch(q)) != null) {
      return w(monday.subtract(const Duration(days: 7)), monday, 'last week', m!.group(0)!);
    }
    if ((m = RegExp(r'\bthis week\b').firstMatch(q)) != null) {
      return w(monday, soon, 'this week', m!.group(0)!);
    }
    if ((m = RegExp(r'\b(this|last|past) weekend\b').firstMatch(q)) != null) {
      final saturdayThisWeek = monday.add(const Duration(days: 5));
      final inWeekend = today.weekday >= 6;
      var start = inWeekend ? saturdayThisWeek : saturdayThisWeek.subtract(const Duration(days: 7));
      if (m!.group(1) == 'last' && inWeekend) start = start.subtract(const Duration(days: 7));
      final end = start.add(const Duration(days: 2));
      return w(start, end.isAfter(now) ? soon : end, '${m.group(1)} weekend', m.group(0)!);
    }
    if ((m = RegExp(r'\bthis month\b').firstMatch(q)) != null) {
      return w(DateTime(now.year, now.month), soon, 'this month', m!.group(0)!);
    }
    if ((m = RegExp(r'\b(?:last|previous) month\b').firstMatch(q)) != null) {
      return w(DateTime(now.year, now.month - 1), DateTime(now.year, now.month), 'last month', m!.group(0)!);
    }
    if ((m = RegExp(r'\bthis year\b').firstMatch(q)) != null) {
      return w(DateTime(now.year), soon, 'this year', m!.group(0)!);
    }

    if ((m = RegExp(r'\bday before yesterday\b').firstMatch(q)) != null) {
      final d = today.subtract(const Duration(days: 2));
      return w(d, d.add(const Duration(days: 1)), 'the day before yesterday', m!.group(0)!);
    }
    final yesterday = today.subtract(const Duration(days: 1));
    if ((m = RegExp(r'\blast night\b').firstMatch(q)) != null) {
      return w(yesterday.add(const Duration(hours: 17)), today.add(const Duration(hours: 5)), 'last night', m!.group(0)!);
    }
    m = RegExp(r'\byesterday(?: (morning|afternoon|evening|night))?\b').firstMatch(q);
    if (m != null) {
      final part = m.group(1);
      if (part == null) return w(yesterday, today, 'yesterday', m.group(0)!);
      final (a, b) = _partOfDay(part);
      return w(yesterday.add(a), yesterday.add(b), 'yesterday $part', m.group(0)!);
    }
    m = RegExp(r'\b(?:this (morning|afternoon|evening)|(tonight)|earlier today|today)\b').firstMatch(q);
    if (m != null) {
      final part = m.group(1) ?? (m.group(2) != null ? 'evening' : null);
      if (part == null) return w(today, soon, 'today', m.group(0)!);
      final (a, b) = _partOfDay(part);
      final end = today.add(b);
      return w(today.add(a), end.isAfter(now) ? soon : end, part == 'evening' && m.group(2) != null ? 'tonight' : 'this $part', m.group(0)!);
    }

    // Weekdays: "on Tuesday", "last Tuesday", "this Tuesday".
    m = RegExp(r'\b(?:(last|this|on|past) )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b').firstMatch(q);
    if (m != null) {
      final target = _weekdays.indexOf(m.group(2)!) + 1;
      DateTime day;
      if (m.group(1) == 'this') {
        day = monday.add(Duration(days: target - 1));
        if (day.isAfter(today)) day = day.subtract(const Duration(days: 7));
      } else {
        var diff = today.weekday - target;
        if (diff <= 0) diff += 7;
        day = today.subtract(Duration(days: diff));
      }
      final name = m.group(2)![0].toUpperCase() + m.group(2)!.substring(1);
      return w(day, day.add(const Duration(days: 1)), name, m.group(0)!);
    }

    // Dates: "March 3", "3 March", "3rd of March", "the 3rd".
    final monthsRe = _months.join('|');
    m = RegExp('\\b($monthsRe) (\\d{1,2})(?:st|nd|rd|th)?\\b').firstMatch(q) ??
        RegExp('\\b(\\d{1,2})(?:st|nd|rd|th)? (?:of )?($monthsRe)\\b').firstMatch(q);
    if (m != null) {
      final first = m.group(1)!;
      final monthName = _months.contains(first) ? first : m.group(2)!;
      final dayNum = int.parse(_months.contains(first) ? m.group(2)! : first);
      var d = DateTime(now.year, _months.indexOf(monthName) + 1, dayNum);
      if (d.isAfter(today)) d = DateTime(now.year - 1, d.month, d.day);
      final label = '${monthName[0].toUpperCase()}${monthName.substring(1)} $dayNum';
      return w(d, d.add(const Duration(days: 1)), label, m.group(0)!);
    }
    m = RegExp(r'\bon the (\d{1,2})(?:st|nd|rd|th)\b').firstMatch(q);
    if (m != null) {
      final dayNum = int.parse(m.group(1)!);
      var d = DateTime(now.year, now.month, dayNum);
      if (d.isAfter(today)) d = DateTime(now.year, now.month - 1, dayNum);
      return w(d, d.add(const Duration(days: 1)), 'the ${m.group(1)}', m.group(0)!);
    }
    m = RegExp('\\bin ($monthsRe)\\b').firstMatch(q);
    if (m != null) {
      final month = _months.indexOf(m.group(1)!) + 1;
      var start = DateTime(now.year, month);
      if (start.isAfter(today)) start = DateTime(now.year - 1, month);
      final end = DateTime(start.year, start.month + 1);
      final name = m.group(1)!;
      return w(start, end.isAfter(now) ? soon : end, name[0].toUpperCase() + name.substring(1), m.group(0)!);
    }

    if ((m = RegExp(r'\b(recently|lately|earlier)\b').firstMatch(q)) != null) {
      return w(today.subtract(const Duration(days: 2)), soon, 'recently', m!.group(0)!);
    }
    return null;
  }

  static int _number(String s) => int.tryParse(s) ?? _numberWords[s.replaceAll(' of', '')] ?? 1;

  static (Duration, Duration) _partOfDay(String part) => switch (part) {
        'morning' => (const Duration(hours: 5), const Duration(hours: 12)),
        'afternoon' => (const Duration(hours: 12), const Duration(hours: 17)),
        'evening' => (const Duration(hours: 17), const Duration(hours: 23, minutes: 59)),
        _ => (const Duration(hours: 17), const Duration(hours: 29)), // night, into early next morning
      };
}
