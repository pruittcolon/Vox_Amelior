import 'package:flutter/material.dart';

import '../core/bridge/ansi.dart';

/// A terminal's screen in its colours, newest line at the bottom. The font
/// shrinks so one terminal line fits the phone's width, the way it looks in
/// Termux. Long-press to select and copy.
class TerminalScreen extends StatefulWidget {
  const TerminalScreen({super.key, required this.text, this.cols = 0, this.onFit});

  /// The screen, colour codes included.
  final String text;

  /// The terminal's width in characters (0: not known).
  final int cols;

  /// Columns and rows that fit here at a comfortable size (to size sessions
  /// that only the phone shows).
  final void Function(int cols, int rows)? onFit;

  static const background = Color(0xFF070B14);
  static const foreground = Color(0xFFD7E0EE);

  @override
  State<TerminalScreen> createState() => _TerminalScreenState();
}

class _TerminalScreenState extends State<TerminalScreen> {
  final _scroll = ScrollController();
  bool _follow = true;
  String _parsed = '\u0000';
  List<List<AnsiSpan>> _lines = const [];
  double? _ratio; // glyph width / font size of the monospace font
  String _fitted = '';

  static final _rule = RegExp(r'^[─━═\s]+$');

  @override
  void initState() {
    super.initState();
    _scroll.addListener(() {
      if (!_scroll.hasClients) return;
      // The list is reversed: offset 0 is the bottom (newest output).
      _follow = _scroll.offset < 40;
    });
  }

  @override
  void dispose() {
    _scroll.dispose();
    super.dispose();
  }

  double get ratio {
    if (_ratio != null) return _ratio!;
    final p = TextPainter(
      text: const TextSpan(text: 'MMMMMMMMMMMMMMMMMMMM', style: TextStyle(fontFamily: 'monospace', fontSize: 20)),
      textDirection: TextDirection.ltr,
    )..layout();
    final r = p.width / 20 / 20;
    p.dispose();
    return _ratio = r > 0.3 && r < 1 ? r : 0.6;
  }

  TextSpan _span(AnsiSpan s, double size) {
    var fg = s.fg == null ? TerminalScreen.foreground : Color(s.fg!);
    var bg = s.bg == null ? null : Color(s.bg!);
    if (s.inverse) {
      final f = fg;
      fg = bg ?? TerminalScreen.background;
      bg = f;
    }
    return TextSpan(
      text: s.text,
      style: TextStyle(
        color: s.dim ? fg.withOpacity(0.6) : fg,
        backgroundColor: bg,
        fontWeight: s.bold ? FontWeight.w700 : null,
        fontStyle: s.italic ? FontStyle.italic : null,
        decoration: s.underline ? TextDecoration.underline : null,
      ),
    );
  }

  Widget _line(List<AnsiSpan> spans, double size) {
    final plain = spans.map((s) => s.text).join();
    if (plain.trim().isNotEmpty && _rule.hasMatch(plain)) {
      // A rule (Claude Code's input box): a line across, whatever the font.
      final color = spans.firstWhere((s) => s.text.trim().isNotEmpty).fg;
      return SizedBox(
        height: size * 1.3,
        child: Center(
          child: Container(height: 1, color: (color == null ? TerminalScreen.foreground : Color(color)).withOpacity(0.5)),
        ),
      );
    }
    return Text.rich(TextSpan(children: [for (final s in spans) _span(s, size)]), softWrap: true);
  }

  @override
  Widget build(BuildContext context) {
    if (widget.text != _parsed) {
      _parsed = widget.text;
      final lines = Ansi.parse(widget.text.trimRight());
      _lines = lines;
      if (_follow && _scroll.hasClients && _scroll.offset != 0) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_scroll.hasClients) _scroll.jumpTo(0);
        });
      }
    }
    return LayoutBuilder(builder: (context, c) {
      const pad = 10.0;
      final width = c.maxWidth - pad * 2;
      final glyph = ratio;
      // A whole terminal line across the screen (a little spare for wide glyphs).
      final size = widget.cols > 0 ? (width / (widget.cols * glyph * 1.02)).clamp(6.5, 15.0) : 12.5;
      final fitCols = (width / (12.5 * glyph)).floor();
      final fitRows = ((c.maxHeight - pad * 2) / (12.5 * 1.3)).floor();
      final key = '$fitCols:$fitRows';
      if (widget.onFit != null && key != _fitted && fitCols > 10 && fitRows > 4) {
        _fitted = key;
        WidgetsBinding.instance.addPostFrameCallback((_) => widget.onFit?.call(fitCols, fitRows));
      }
      final style = TextStyle(fontFamily: 'monospace', fontSize: size, height: 1.3, color: TerminalScreen.foreground);
      // Fitted to the width: the phone's text size would push lines over.
      return MediaQuery.withNoTextScaling(
        child: DefaultTextStyle(
        style: style,
        child: SelectionArea(
          child: Scrollbar(
            controller: _scroll,
            child: ListView.builder(
              controller: _scroll,
              reverse: true,
              padding: const EdgeInsets.all(pad),
              itemCount: _lines.length,
              itemBuilder: (context, i) => _line(_lines[_lines.length - 1 - i], size),
            ),
          ),
        ),
      ),
      );
    });
  }
}
