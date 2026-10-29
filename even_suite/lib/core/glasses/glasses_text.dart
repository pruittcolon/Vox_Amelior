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

  /// Split [text] at blank lines into chunks (one thought each) and wrap each.
  static List<List<String>> wrapChunks(String text) => text
      .split(RegExp(r'\n\s*\n'))
      .map(wrap)
      .where((lines) => lines.isNotEmpty)
      .toList();

  static ScreenSize? _size;

  /// How much text fits: measured once with the same layout the glasses use,
  /// so Claude can be told how long its chunks may be.
  static ScreenSize screenSize() => _size ??= _measure();

  static ScreenSize _measure() {
    const sample = 'The quick brown fox jumps over the lazy dog while the five boxing wizards '
        'jump quickly. Pack my box with five dozen liquor jugs, then sphinx of black quartz, '
        'judge my vow. How vexingly quick daft zebras jump when the night is still young. ';
    final lines = wrap(sample * 3);
    final chars = lines.fold<int>(0, (n, l) => n + l.length + 1);
    final perLine = lines.isEmpty ? 40 : (chars / lines.length).floor();
    return ScreenSize(perLine, linesPerScreen);
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

class ScreenSize {
  final int charsPerLine;
  final int lines;
  const ScreenSize(this.charsPerLine, this.lines);
  int get chars => charsPerLine * lines;
}

/// Splits text into 5-line screens and remembers which one is shown.
///
/// Finished text is paged by chunk (blank-line separated): a chunk that fits
/// on the current screen stays together, so a thought is never cut in half.
/// Live text (terminal, dictation) is paged every 5 lines and can follow the
/// newest screen.
class GlassesPager {
  GlassesPager(List<String> lines, {this.followTail = false}) {
    _setPages(_fixed(lines));
    if (followTail) _page = lastPage;
  }

  GlassesPager._chunked(List<List<String>> chunks) : followTail = false {
    _setPages(_byChunk(chunks));
  }

  /// Model or note text: cleaned, wrapped and paged by chunk.
  factory GlassesPager.fromText(String text) =>
      GlassesPager._chunked(GlassesText.wrapChunks(GlassesText.clean(text)));

  /// Live text (dictation, terminal): stay on the newest screen while the
  /// user has not scrolled back.
  final bool followTail;

  static const int per = GlassesText.linesPerScreen;

  List<List<String>> _pages = const [[]];
  int _page = 0;

  List<String> get lines => [for (final p in _pages) ...p];
  int get page => _page;
  int get pageCount => _pages.length;
  int get lastPage => pageCount - 1;
  bool get atStart => _page == 0;
  bool get atEnd => _page >= lastPage;

  /// Lines on the current screen.
  List<String> get pageLines => _pages[_page];

  void _setPages(List<List<String>> pages) => _pages = pages.isEmpty ? const [[]] : pages;

  static List<List<String>> _fixed(List<String> lines) => [
        for (var i = 0; i < lines.length; i += per)
          lines.sublist(i, (i + per).clamp(0, lines.length)),
      ];

  static List<List<String>> _byChunk(List<List<String>> chunks) {
    final pages = <List<String>>[];
    var current = <String>[];
    for (final chunk in chunks) {
      if (chunk.length > per) {
        // Too long for one screen: start it on a fresh one and spill over.
        if (current.isNotEmpty) pages.add(current);
        final split = _fixed(chunk);
        pages.addAll(split.take(split.length - 1));
        current = List.of(split.last);
      } else if (current.length + chunk.length <= per) {
        current.addAll(chunk);
      } else {
        pages.add(current);
        current = List.of(chunk);
      }
    }
    if (current.isNotEmpty) pages.add(current);
    return pages;
  }

  /// Replace the lines (live content). Keeps the position, or follows the
  /// end if the user was already looking at it.
  void update(List<String> lines) {
    final wasAtEnd = atEnd;
    _setPages(_fixed(lines));
    if (followTail && wasAtEnd) {
      _page = lastPage;
    } else {
      _page = _page.clamp(0, lastPage);
    }
  }

  /// Go to [p] (clamped), e.g. to stay on the same page after new text.
  void jumpTo(int p) => _go(p);

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

  /// Words on the current screen (for reading-speed paging).
  int get pageWords => pageLines.fold(0, (n, l) => n + l.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).length);

  /// The text for the current screen. A short single screen is pushed down a
  /// little like the demo does, so it sits in the middle of the view.
  String screen() {
    final body = pageLines.map((s) => '$s\n').join();
    if (body.isEmpty) return '';
    if (pageCount == 1) {
      if (pageLines.length < 4) return '\n\n$body';
      if (pageLines.length == 4) return '\n$body';
    }
    return body;
  }
}
