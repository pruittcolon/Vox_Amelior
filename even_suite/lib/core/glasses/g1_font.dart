/// Widths of the G1's own font, so text can be broken into lines exactly
/// where the glasses would run out of room.
///
/// A character takes (glyph width + 1) x 2 pixels; the glyph widths are the
/// ones the open-source MentraOS project measured on the glasses
/// (profiles/g1.ts, G1 font 1.5.6). Characters not in the table count as the
/// widest Latin letter (16 px), so a line never comes out too long.
class G1Font {
  G1Font._();

  static const Map<String, int> _glyph = {
    ' ': 2, '!': 1, '"': 2, '#': 6, r'$': 5, '%': 6, '&': 7, "'": 1, '(': 2, ')': 2, '*': 3, '+': 4,
    ',': 1, '-': 4, '.': 1, '/': 3, '0': 5, '1': 3, '2': 5, '3': 5, '4': 5, '5': 5, '6': 5, '7': 5,
    '8': 5, '9': 5, ':': 1, ';': 1, '<': 4, '=': 4, '>': 4, '?': 5, '@': 7, 'A': 6, 'B': 5, 'C': 5,
    'D': 5, 'E': 4, 'F': 4, 'G': 5, 'H': 5, 'I': 2, 'J': 3, 'K': 5, 'L': 4, 'M': 7, 'N': 5, 'O': 5,
    'P': 5, 'Q': 5, 'R': 5, 'S': 5, 'T': 5, 'U': 5, 'V': 6, 'W': 7, 'X': 6, 'Y': 6, 'Z': 5, '[': 2,
    r'\': 3, ']': 2, '^': 4, '_': 3, '`': 2, 'a': 5, 'b': 4, 'c': 4, 'd': 4, 'e': 4, 'f': 4, 'g': 4,
    'h': 4, 'i': 1, 'j': 2, 'k': 4, 'l': 1, 'm': 7, 'n': 4, 'o': 4, 'p': 4, 'q': 4, 'r': 3, 's': 4,
    't': 3, 'u': 5, 'v': 5, 'w': 7, 'x': 5, 'y': 5, 'z': 4, '{': 3, '|': 1, '}': 3, '~': 7,
  };

  /// Zero-width: the firmware uses it to switch the rest of a dashboard note to dim.
  static const dimMark = '​';

  /// Pixels one character takes on the glasses.
  static int advance(int rune) {
    if (rune == 0x200B || rune == 0x0A || rune == 0x0D) return 0;
    final w = _glyph[String.fromCharCode(rune)];
    if (w != null) return (w + 1) * 2;
    if ((rune >= 0x4E00 && rune <= 0x9FFF) || (rune >= 0x3040 && rune <= 0x30FF) || (rune >= 0x0400 && rune <= 0x04FF)) {
      return 18;
    }
    if (rune >= 0xAC00 && rune <= 0xD7AF) return 24;
    return 16;
  }

  /// Pixels a line takes on the glasses.
  static int width(String line) => line.runes.fold(0, (n, r) => n + advance(r));

  /// Break [text] into lines of at most [maxPx], at spaces (a word longer than
  /// a whole line is split). Existing line breaks are kept.
  static List<String> wrap(String text, int maxPx) {
    final out = <String>[];
    for (final para in text.split('\n')) {
      var line = '';
      for (final word in para.split(' ').where((w) => w.isNotEmpty)) {
        final candidate = line.isEmpty ? word : '$line $word';
        if (width(candidate) <= maxPx) {
          line = candidate;
          continue;
        }
        if (line.isNotEmpty) out.add(line);
        line = word;
        // A word that does not fit on a line of its own is cut into pieces.
        while (width(line) > maxPx) {
          var cut = line.length;
          while (cut > 1 && width(line.substring(0, cut)) > maxPx) {
            cut--;
          }
          out.add(line.substring(0, cut));
          line = line.substring(cut);
        }
      }
      if (line.isNotEmpty) out.add(line);
    }
    return out;
  }

  /// [line] cut to fit [maxPx], ending in "..." when it had to be cut.
  static String fit(String line, int maxPx) {
    if (width(line) <= maxPx) return line;
    var s = line;
    while (s.isNotEmpty && width('${s.trimRight()}...') > maxPx) {
      s = s.substring(0, s.length - 1);
    }
    return '${s.trimRight()}...';
  }

  /// [text] wrapped into at most [maxLines] lines; the last one ends in
  /// "..." if anything was left over.
  static List<String> wrapMax(String text, int maxPx, int maxLines) {
    final lines = wrap(text, maxPx);
    if (lines.length <= maxLines) return lines;
    return [...lines.take(maxLines - 1), fit(lines.skip(maxLines - 1).join(' '), maxPx)];
  }
}
