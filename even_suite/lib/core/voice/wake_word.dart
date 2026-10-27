/// How careful the wake word is about stray mentions.
enum WakeStrictness {
  /// "Claude" anywhere in a sentence counts.
  relaxed,

  /// Only at the start of what you say ("Claude, ..." / "Hey Claude, ...").
  normal,

  /// Only "Hey Claude" / "OK Claude" at the start.
  strict,
}

/// Finds the wake word in recognized speech.
///
/// Recognizers rarely spell "Claude" right, so a set of sound-alikes counts
/// too ("clod", "cloud", "clawed", ...). How strict it is about where the
/// word sits is [strictness]: by default only at the start of what was heard
/// (or right after "hey"/"ok"), so "I asked Claude yesterday" in a
/// conversation nearby does not wake it. "cloud" is an everyday word, so it
/// only ever counts there.
class WakeWord {
  WakeWord([Iterable<String>? words, this.strictness = WakeStrictness.normal])
      : words = _normalizeAll(words ?? defaults);

  final WakeStrictness strictness;

  static const List<String> defaults = [
    'claude', 'claud', 'clode', 'clod', 'klaud', 'klod', 'clawed', 'cloud',
  ];

  /// Words that only count at the start of an utterance.
  static const Set<String> _startOnly = {'cloud'};
  static const Set<String> _greetings = {'hey', 'hi', 'ok', 'okay', 'yo', 'hello'};

  final Set<String> words;

  static Set<String> _normalizeAll(Iterable<String> list) =>
      list.map(_norm).where((w) => w.isNotEmpty).toSet();

  static String _norm(String w) {
    var t = w.toLowerCase().replaceAll(RegExp(r"[^a-z']"), '');
    if (t.endsWith("'s")) t = t.substring(0, t.length - 2);
    return t.replaceAll("'", '');
  }

  static final _token = RegExp(r"[A-Za-z']+");

  /// The first wake word in [text], and everything said after it (repeated
  /// wake words such as "claude... claude... clod" are skipped). Null if none.
  WakeMatch? find(String text) {
    final tokens = _token.allMatches(text).toList();
    for (var i = 0; i < tokens.length; i++) {
      final w = _norm(tokens[i][0]!);
      if (!words.contains(w)) continue;
      final ok = switch (strictness) {
        WakeStrictness.relaxed => !_startOnly.contains(w) || _atStart(tokens, i),
        WakeStrictness.normal => _atStart(tokens, i),
        WakeStrictness.strict => i == 1 && _greetings.contains(_norm(tokens[0][0]!)),
      };
      if (!ok) continue;
      // Skip straight repeats: "claude claude clod what time is it".
      var last = i;
      while (last + 1 < tokens.length && _isWake(tokens, last + 1)) {
        last++;
      }
      final restStart = tokens[last].end;
      return WakeMatch(
        start: tokens[i].start,
        restStart: restStart,
        rest: _trimLead(text.substring(restStart)),
      );
    }
    return null;
  }

  bool _isWake(List<RegExpMatch> tokens, int i) => words.contains(_norm(tokens[i][0]!));

  bool _atStart(List<RegExpMatch> tokens, int i) {
    if (i == 0) return true;
    final prev = _norm(tokens[i - 1][0]!);
    return _greetings.contains(prev) || words.contains(prev);
  }

  static String _trimLead(String s) => s.replaceFirst(RegExp(r'^[\s,.!?:;\-]+'), '').trimRight();
}

class WakeMatch {
  /// Character offset where the wake word starts.
  final int start;

  /// Character offset where what was said after the wake word(s) begins.
  final int restStart;

  /// What was said after the wake word(s), trimmed.
  final String rest;

  const WakeMatch({required this.start, required this.restStart, required this.rest});

  @override
  String toString() => 'WakeMatch($start, $restStart, "$rest")';
}
