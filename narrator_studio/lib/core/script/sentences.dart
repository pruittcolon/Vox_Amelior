/// Splits a paragraph into sentence ranges (start/end offsets), so the screen
/// can highlight exactly what is being read and a tap can jump to it.
library;

class TextRange {
  const TextRange(this.start, this.end);
  final int start;
  final int end;

  @override
  bool operator ==(Object other) => other is TextRange && other.start == start && other.end == end;

  @override
  int get hashCode => Object.hash(start, end);

  @override
  String toString() => '[$start,$end)';
}

/// Words that end with a period but don't end a sentence.
const _noBreakAfter = {
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'vs', 'etc', 'e.g', 'i.e', 'eg', 'ie', 'cf', 'al',
  'fig', 'figs', 'vol', 'no', 'nos', 'ch', 'pp', 'p', 'approx', 'ca', 'gen', 'col', 'lt', 'sgt', 'capt',
  'rev', 'hon', 'inc', 'ltd', 'co', 'corp', 'dept', 'univ', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug',
  'sep', 'sept', 'oct', 'nov', 'dec', 'u.s', 'u.k', 'a.m', 'p.m', 'ph.d', 'b.c', 'a.d', 'op', 'ed', 'eds',
};

const maxSentenceChars = 260;

final _end = RegExp(r'[.!?…]+["”’)\]]*');

List<TextRange> splitSentences(String text) {
  final ranges = <TextRange>[];
  var start = _skipSpace(text, 0);
  for (final m in _end.allMatches(text)) {
    final end = m.end;
    if (end <= start) continue;
    // The next thing must be whitespace followed by a sentence opener.
    if (end < text.length) {
      if (!_isSpace(text.codeUnitAt(end))) continue;
      final next = _skipSpace(text, end);
      if (next >= text.length) break;
      if (!_opensSentence(text[next])) continue;
    }
    final mark = text[m.start];
    if (mark == '.' && !_periodEnds(text, m.start)) continue;
    _add(text, start, end, ranges);
    start = _skipSpace(text, end);
  }
  if (start < text.length) _add(text, start, text.length, ranges);
  return ranges;
}

bool _isSpace(int c) => c == 0x20 || c == 0x0A || c == 0x09 || c == 0x0D || c == 0xA0;

int _skipSpace(String t, int i) {
  while (i < t.length && _isSpace(t.codeUnitAt(i))) {
    i++;
  }
  return i;
}

bool _opensSentence(String c) => RegExp(r'''[A-Z0-9"“‘'(\[¿¡À-Þ]''').hasMatch(c);

/// Decides whether the period at [dot] ends a sentence.
bool _periodEnds(String text, int dot) {
  var i = dot - 1;
  while (i >= 0 && !_isSpace(text.codeUnitAt(i)) && text[i] != '(' && text[i] != '"') {
    i--;
  }
  final word = text.substring(i + 1, dot).toLowerCase().replaceAll(RegExp(r'^[\[(“"‘]+'), '');
  if (word.isEmpty) return true;
  if (_noBreakAfter.contains(word)) return false;
  // Initials like "J." in "J. R. R. Tolkien".
  if (RegExp(r'^[a-z]$').hasMatch(word) && RegExp(r'[A-Z]').hasMatch(text[i + 1 < dot ? i + 1 : dot])) {
    return false;
  }
  return true;
}

/// Adds [start, end) trimmed; sentences longer than [maxSentenceChars] are cut
/// at the best clause boundary (;  :  ,  dash) so each piece starts quickly.
void _add(String text, int start, int end, List<TextRange> out) {
  while (end > start && _isSpace(text.codeUnitAt(end - 1))) {
    end--;
  }
  if (end <= start) return;
  while (end - start > maxSentenceChars) {
    final cut = _clauseCut(text, start, end);
    if (cut == null) break;
    out.add(TextRange(start, cut));
    start = _skipSpace(text, cut);
  }
  out.add(TextRange(start, end));
}

int? _clauseCut(String text, int start, int end) {
  final limit = start + maxSentenceChars;
  final minimum = start + 60;
  int? best;
  var bestScore = -1;
  for (var i = minimum; i < limit && i < end - 20; i++) {
    final c = text[i];
    final score = switch (c) {
      ';' => 3,
      ':' => 3,
      '—' => 2,
      ',' => 1,
      _ => 0,
    };
    if (score == 0) continue;
    if (i + 1 < text.length && !_isSpace(text.codeUnitAt(i + 1)) && c != '—') continue;
    // Prefer stronger marks; among equals prefer the later one.
    if (score >= bestScore) {
      best = i + 1;
      bestScore = score;
    }
  }
  if (best != null) return best;
  // No punctuation: cut at the last space before the limit.
  for (var i = limit; i > minimum; i--) {
    if (_isSpace(text.codeUnitAt(i))) return i;
  }
  return null;
}
