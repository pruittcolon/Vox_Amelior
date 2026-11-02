/// Terminal colour codes (SGR, what `tmux capture-pane -e` writes) turned
/// into styled runs of text, so the phone shows Claude Code in its colours.
library;

/// One run of text in one style. Colours are 0xAARRGGBB; null is the
/// terminal's default.
class AnsiSpan {
  final String text;
  final int? fg;
  final int? bg;
  final bool bold;
  final bool dim;
  final bool italic;
  final bool underline;
  final bool inverse;
  const AnsiSpan(this.text,
      {this.fg, this.bg, this.bold = false, this.dim = false, this.italic = false, this.underline = false, this.inverse = false});

  @override
  String toString() => 'AnsiSpan("$text", fg: ${fg?.toRadixString(16)}, bg: ${bg?.toRadixString(16)}, bold: $bold)';
}

class _Style {
  int? fg;
  int? bg;
  bool bold = false, dim = false, italic = false, underline = false, inverse = false;

  AnsiSpan span(String text) => AnsiSpan(text,
      fg: fg, bg: bg, bold: bold, dim: dim, italic: italic, underline: underline, inverse: inverse);

  void reset() {
    fg = bg = null;
    bold = dim = italic = underline = inverse = false;
  }

  void apply(String params) {
    final p = params.isEmpty ? const [0] : [for (final x in params.split(RegExp('[;:]'))) int.tryParse(x) ?? 0];
    for (var i = 0; i < p.length; i++) {
      final c = p[i];
      if (c == 0) {
        reset();
      } else if (c == 1) {
        bold = true;
      } else if (c == 2) {
        dim = true;
      } else if (c == 3) {
        italic = true;
      } else if (c == 4) {
        underline = true;
      } else if (c == 7) {
        inverse = true;
      } else if (c == 22) {
        bold = dim = false;
      } else if (c == 23) {
        italic = false;
      } else if (c == 24) {
        underline = false;
      } else if (c == 27) {
        inverse = false;
      } else if (c >= 30 && c <= 37) {
        fg = Ansi.palette[c - 30];
      } else if (c >= 90 && c <= 97) {
        fg = Ansi.palette[c - 90 + 8];
      } else if (c >= 40 && c <= 47) {
        bg = Ansi.palette[c - 40];
      } else if (c >= 100 && c <= 107) {
        bg = Ansi.palette[c - 100 + 8];
      } else if (c == 39) {
        fg = null;
      } else if (c == 49) {
        bg = null;
      } else if (c == 38 || c == 48) {
        // 38;5;n (256 colours) or 38;2;r;g;b (true colour).
        int? color;
        if (i + 2 < p.length && p[i + 1] == 5) {
          color = Ansi.color256(p[i + 2]);
          i += 2;
        } else if (i + 4 < p.length && p[i + 1] == 2) {
          color = 0xFF000000 | (p[i + 2].clamp(0, 255) << 16) | (p[i + 3].clamp(0, 255) << 8) | p[i + 4].clamp(0, 255);
          i += 4;
        }
        if (c == 38) {
          fg = color;
        } else {
          bg = color;
        }
      }
    }
  }
}

class Ansi {
  static final _sgr = RegExp(r'\x1b\[([0-9;:]*)m');

  /// Every other escape: titles and links (OSC), cursor moves (any CSI but
  /// colours, which end in "m"), charset picks.
  static final _other = RegExp(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?<>=]*[A-Za-ln-z@`]|\x1b[()][0-9A-Za-z]|\x1b[=>78DEHMNOZc]');

  /// The text without colour codes.
  static String strip(String s) => s.contains('\x1b') ? s.replaceAll(_other, '').replaceAll(_sgr, '') : s;

  /// Styled runs, line by line. A colour carries on to the next line until
  /// it is changed (tmux writes it once).
  static List<List<AnsiSpan>> parse(String text) {
    final style = _Style();
    final lines = <List<AnsiSpan>>[];
    for (final raw in text.split('\n')) {
      final line = raw.contains('\x1b') ? raw.replaceAll(_other, '') : raw;
      final spans = <AnsiSpan>[];
      var pos = 0;
      for (final m in _sgr.allMatches(line)) {
        if (m.start > pos) spans.add(style.span(line.substring(pos, m.start)));
        style.apply(m.group(1)!);
        pos = m.end;
      }
      if (pos < line.length) spans.add(style.span(line.substring(pos)));
      lines.add(spans);
    }
    return lines;
  }

  /// The 16 basic colours, tuned for a dark background.
  static const palette = [
    0xFF1B1F27, 0xFFE5686B, 0xFF7FD18B, 0xFFE8C26E, 0xFF6FA8F5, 0xFFC792EA, 0xFF5FD3D9, 0xFFD7E0EE, //
    0xFF5C6370, 0xFFFF8A8D, 0xFFA3E9AE, 0xFFF5D88E, 0xFF92C1FF, 0xFFDDB2F7, 0xFF8BE9EE, 0xFFFFFFFF,
  ];

  /// xterm's 256 colours: the 16 above, a 6x6x6 cube, then 24 greys.
  static int color256(int n) {
    if (n < 0 || n > 255) return palette[7];
    if (n < 16) return palette[n];
    if (n < 232) {
      final i = n - 16;
      int level(int v) => v == 0 ? 0 : 55 + v * 40;
      return 0xFF000000 | (level(i ~/ 36) << 16) | (level(i ~/ 6 % 6) << 8) | level(i % 6);
    }
    final g = 8 + (n - 232) * 10;
    return 0xFF000000 | (g << 16) | (g << 8) | g;
  }
}
