import 'package:flutter/painting.dart';

/// Text for the G1 display: 5 lines per screen, wrapped the way the demo does
/// (488 px wide at font size 21).
class GlassesText {
  GlassesText._();

  static const int linesPerScreen = 5;
  static const double maxWidth = 488;
  static const double fontSize = 21;

  /// Wrap [text] into display lines (the demo's measureStringList).
  static List<String> wrap(String text, [double width = maxWidth]) {
    final paragraphs = text
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty);
    const style = TextStyle(fontSize: fontSize);
    final out = <String>[];
    for (final paragraph in paragraphs) {
      final painter = TextPainter(
        text: TextSpan(text: paragraph, style: style),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: width);
      final lineCount = painter.computeLineMetrics().length;
      var start = 0;
      for (var i = 0; i < lineCount && start < paragraph.length; i++) {
        final line = painter.getLineBoundary(TextPosition(offset: start));
        if (line.end <= start) break;
        out.add(paragraph.substring(line.start, line.end).trim());
        start = line.end;
      }
      painter.dispose();
    }
    return out;
  }

  static final _fence = RegExp(r'^\s*```.*$', multiLine: true);
  static final _heading = RegExp(r'^\s{0,3}#{1,6}\s+', multiLine: true);
  static final _bullet = RegExp(r'^(\s*)[*+•]\s+', multiLine: true);
  static final _emphasis = RegExp(r'(\*\*|__|\*|`)');
  static final _link = RegExp(r'\[([^\]]+)\]\(([^)]+)\)');
  static final _spaces = RegExp(r'[ \t]+');

  static const _map = {
    '●': '*', '⏺': '*', '⎿': '>', '❯': '>', '›': '>', '→': '->', '←': '<-',
    '✓': 'v', '✔': 'v', '✗': 'x', '✘': 'x', '…': '...', '•': '-', '·': '-',
    '“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '─': ' ',
    '│': ' ', ' ': ' ',
  };

  /// Make model or terminal output readable on the glasses: no Markdown
  /// markers, ASCII stand-ins for symbols the G1 font lacks, no emoji.
  static String clean(String text) {
    var t = text.replaceAll('\r', '');
    t = t.replaceAll(_fence, '');
    t = t.replaceAll(_heading, '');
    t = t.replaceAllMapped(_link, (m) => m[1]!);
    t = t.replaceAllMapped(_bullet, (m) => '${m[1]}- ');
    t = t.replaceAll(_emphasis, '');
    final b = StringBuffer();
    for (final rune in t.runes) {
      final ch = String.fromCharCode(rune);
      final mapped = _map[ch];
      if (mapped != null) {
        b.write(mapped);
      } else if ((rune >= 0x2190 && rune <= 0x2BFF) || rune >= 0x1F000 || (rune >= 0xFE00 && rune <= 0xFE0F)) {
        // arrows, box drawing, dingbats, emoji: drop
      } else {
        b.write(ch);
      }
    }
    return b
        .toString()
        .split('\n')
        .map((l) => l.replaceAll(_spaces, ' ').trimRight())
        .join('\n')
        .replaceAll(RegExp(r'\n{3,}'), '\n\n')
        .trim();
  }
}

/// Splits wrapped lines into 5-line screens and remembers which one is shown.
class GlassesPager {
  GlassesPager(List<String> lines, {this.followTail = false})
      : _lines = List.of(lines) {
    if (followTail) _page = lastPage;
  }

  factory GlassesPager.fromText(String text, {bool followTail = false}) =>
      GlassesPager(GlassesText.wrap(GlassesText.clean(text)), followTail: followTail);

  List<String> _lines;
  int _page = 0;

  /// Live text (terminal, dictation): stay on the newest screen while the
  /// user has not scrolled back.
  final bool followTail;

  static const int per = GlassesText.linesPerScreen;

  List<String> get lines => _lines;
  int get page => _page;
  int get pageCount => _lines.isEmpty ? 1 : (_lines.length + per - 1) ~/ per;
  int get lastPage => pageCount - 1;
  bool get atStart => _page == 0;
  bool get atEnd => _page >= lastPage;

  /// Replace the lines (live content). Keeps the position, or follows the
  /// end if the user was already looking at it.
  void update(List<String> lines) {
    final wasAtEnd = atEnd;
    _lines = List.of(lines);
    if (followTail && wasAtEnd) {
      _page = lastPage;
    } else {
      _page = _page.clamp(0, lastPage);
    }
  }

  bool up() => _go(_page - 1);
  bool down() => _go(_page + 1);
  bool top() => _go(0);
  bool bottom() => _go(lastPage);

  bool _go(int p) {
    final next = p.clamp(0, lastPage);
    if (next == _page) return false;
    _page = next;
    return true;
  }

  /// The text for the current screen. A short single screen is pushed down a
  /// little like the demo does, so it sits in the middle of the view.
  String screen() {
    if (_lines.isEmpty) return '';
    final start = _page * per;
    final end = (start + per).clamp(0, _lines.length);
    final body = _lines.sublist(start, end).map((s) => '$s\n').join();
    if (pageCount == 1) {
      if (_lines.length < 4) return '\n\n$body';
      if (_lines.length == 4) return '\n$body';
    }
    return body;
  }
}
