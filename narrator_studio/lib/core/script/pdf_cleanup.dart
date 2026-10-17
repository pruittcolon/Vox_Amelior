/// Rebuilds readable paragraphs from text pulled out of a PDF.
///
/// PDF text arrives as lines (one per visual line) with page numbers, running
/// headers/footers and words hyphenated across line breaks. Read raw, a voice
/// would say "Page 12 · The Origin of Species" every page and pause at every
/// line end. This removes the furniture and joins the lines back up.
library;

String cleanPdfPages(List<String> pages) {
  final pageLines = [
    for (final p in pages)
      p.replaceAll('\r\n', '\n').replaceAll('\r', '\n').split('\n').map(_normalizeLine).toList(),
  ];
  final furniture = _repeatedLines(pageLines);

  final kept = <String>[];
  for (final lines in pageLines) {
    final trimmed = <String>[];
    for (var i = 0; i < lines.length; i++) {
      final line = lines[i];
      final key = _furnitureKey(line);
      final edge = i < 3 || i >= lines.length - 3;
      if (edge && (furniture.contains(key) || _isPageNumber(line))) continue;
      trimmed.add(line);
    }
    // A page break is just a line break; paragraphs continue across pages.
    kept.addAll(trimmed);
    kept.add('\u0000page');
  }
  return _joinParagraphs(kept);
}

String _normalizeLine(String l) => l
    .replaceAll('ﬁ', 'fi')
    .replaceAll('ﬂ', 'fl')
    .replaceAll('ﬀ', 'ff')
    .replaceAll('ﬃ', 'ffi')
    .replaceAll('ﬄ', 'ffl')
    .replaceAll(' ', ' ')
    .replaceAll(RegExp(r'[ \t]+'), ' ')
    .trim();

/// Digits stripped so "Page 3 of 90" and "Page 4 of 90" count as the same line.
String _furnitureKey(String line) => line.toLowerCase().replaceAll(RegExp(r'\d+'), '#');

bool _isPageNumber(String line) =>
    RegExp(r'^(page\s*)?[-–—(\[]?\s*(\d{1,4}|[ivxlcdm]{1,6})\s*[-–—)\]]?(\s*(of|/)\s*\d{1,4})?$', caseSensitive: false)
        .hasMatch(line);

/// Lines that appear near the top or bottom of at least 30% of pages.
Set<String> _repeatedLines(List<List<String>> pages) {
  if (pages.length < 3) return {};
  final counts = <String, int>{};
  for (final lines in pages) {
    final seen = <String>{};
    for (var i = 0; i < lines.length; i++) {
      if (i >= 3 && i < lines.length - 3) continue;
      final l = lines[i];
      if (l.isEmpty || l.length > 90) continue;
      seen.add(_furnitureKey(l));
    }
    for (final k in seen) {
      counts[k] = (counts[k] ?? 0) + 1;
    }
  }
  final needed = (pages.length * 0.3).ceil().clamp(2, 1 << 30);
  return {
    for (final e in counts.entries)
      if (e.value >= needed) e.key,
  };
}

String _joinParagraphs(List<String> lines) {
  final out = StringBuffer();
  final para = StringBuffer();
  var lastLen = 0;
  final typical = _typicalLineLength(lines);

  void endParagraph() {
    final t = para.toString().trim();
    if (t.isNotEmpty) {
      out
        ..write(t)
        ..write('\n\n');
    }
    para.clear();
  }

  for (var i = 0; i < lines.length; i++) {
    final line = lines[i];
    if (line == '\u0000page') continue;
    if (line.isEmpty) {
      endParagraph();
      lastLen = 0;
      continue;
    }
    final current = para.toString();
    if (current.isEmpty) {
      para.write(line);
    } else if (current.endsWith('-') && RegExp(r'[a-z]-$').hasMatch(current) && RegExp(r'^[a-z]').hasMatch(line)) {
      // "photo-\nsynthesis" -> "photosynthesis"
      final s = current.substring(0, current.length - 1);
      para
        ..clear()
        ..write(s)
        ..write(line);
    } else if (_startsNewParagraph(current, line, lastLen, typical)) {
      endParagraph();
      para.write(line);
    } else {
      para
        ..write(' ')
        ..write(line);
    }
    lastLen = line.length;
  }
  endParagraph();
  return out.toString().trim();
}

int _typicalLineLength(List<String> lines) {
  final lens = [
    for (final l in lines)
      if (l.length > 20 && l != '\u0000page') l.length,
  ]..sort();
  if (lens.isEmpty) return 60;
  return lens[(lens.length * 0.75).floor().clamp(0, lens.length - 1)];
}

bool _startsNewParagraph(String current, String line, int lastLen, int typical) {
  final endsSentence = RegExp(r'''[.!?:"”’)]$''').hasMatch(current);
  // A short previous line that ended a sentence: the paragraph ended there.
  if (endsSentence && lastLen > 0 && lastLen < typical * 0.75) return true;
  // Headings: short line without a full stop, followed by a capitalized line.
  if (lastLen > 0 && lastLen < 60 && !RegExp(r'[.,;:!?]$').hasMatch(current) && current.length == lastLen) {
    if (RegExp(r'^[A-Z0-9]').hasMatch(line) && RegExp(r'^[A-Z0-9]').hasMatch(current)) return true;
  }
  // Bullets.
  if (RegExp(r'^([•▪◦‣\-*]|\d{1,2}[.)])\s').hasMatch(line)) return true;
  return false;
}
