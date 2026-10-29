import 'dart:math';
import 'dart:typed_data';

import '../screen/mono_encoder.dart';
import 'g1_font.dart';
import 'look_layout.dart';

/// A thin "HUD" outline for your look-up screen, sent as a picture right
/// after its text: the glasses keep the text and draw the picture over it,
/// and only lit pixels show.
///
/// The text and the picture share the same 576 x 136 area. The five text
/// rows sit 27 px apart with letters from about y = 3 + 27r to y = 22 + 27r,
/// and text never goes right of x = 570. So the outline only uses the gaps
/// between rows and the strip at the right edge, and never touches a letter:
///
/// * each group of text (an item, or items stacked at the same x in rows next
///   to each other with the same width) gets corner brackets 4 px from its
///   first and last letter,
/// * a scale of ticks runs down the right edge, longer at each row gap.
///
/// Styles: hud (top-left and bottom-right corners), corners (all four),
/// underline (a line under each group), off.
class HudFrame {
  HudFrame._();

  static const styles = [('hud', 'HUD'), ('corners', 'Corners'), ('underline', 'Underline'), ('off', 'Off')];

  static const w = MonoEncoder.width, h = MonoEncoder.height;
  static const pitch = 27;

  /// The line just above row [r]'s letters, and just below them.
  static int top(int r) => r * pitch;
  static int bottom(int r) => r * pitch + 25;

  /// Where the text is: (x0, x1, first row, last row) of each group, 4 px
  /// out from its letters (x0 < 0 when the text starts at the left edge).
  static List<(int, int, int, int)> groups(LookLayout layout, List<LookPiece> pieces) {
    final ink = <int, List<int>>{}; // item -> [left, right, first row, last row]
    for (final p in pieces) {
      if (p.item >= layout.items.length || layout.items[p.item].kind == LookKind.line) continue;
      final text = p.text.trim();
      if (text.isEmpty) continue;
      final lead = p.text.length - p.text.trimLeft().length;
      final a = p.x + G1Font.width(p.text.substring(0, lead));
      final b = a + G1Font.width(text) - 2; // the last letter's own spacing
      final e = ink[p.item];
      if (e == null) {
        ink[p.item] = [a, b, p.row, p.row];
      } else {
        e[0] = min(e[0], a);
        e[1] = max(e[1], b);
        e[2] = min(e[2], p.row);
        e[3] = max(e[3], p.row);
      }
    }
    // Items stacked in rows next to each other with the same box (x and
    // width) - a column - share one outline.
    String column(int i) => '${layout.items[i].x}:${layout.items[i].w}';
    final items = [for (final e in ink.entries) (column(e.key), e.value)]
      ..sort((a, b) => a.$1 != b.$1 ? a.$1.compareTo(b.$1) : a.$2[2].compareTo(b.$2[2]));
    final out = <(String, List<int>)>[];
    for (final (x, e) in items) {
      if (out.isNotEmpty && out.last.$1 == x && out.last.$2[3] + 1 == e[2]) {
        final g = out.last.$2;
        g[0] = min(g[0], e[0]);
        g[1] = max(g[1], e[1]);
        g[3] = e[3];
      } else {
        out.add((x, [...e]));
      }
    }
    return [
      for (final (_, g) in out) (g[0] - 4, min(g[1] + 4, 571), g[2], g[3]),
    ];
  }

  /// Lit flags (1 = lit), [w] x [h].
  static Uint8List lit(String style, LookLayout layout, List<LookPiece> pieces) {
    final px = Uint8List(w * h);
    if (style == 'off') return px;
    void dot(int x, int y) {
      if (x >= 0 && x < w && y >= 0 && y < h) px[y * w + x] = 1;
    }

    void hline(int x0, int x1, int y, {int every = 1}) {
      for (var x = x0; x <= x1; x += every) {
        dot(x, y);
      }
    }

    void vline(int x, int y0, int y1) {
      for (var y = y0; y <= y1; y++) {
        dot(x, y);
      }
    }

    // The scale down the right edge (text stops at x 570).
    for (var y = 3; y < h - 3; y += 9) {
      hline(573, 575, y);
    }
    for (var r = 1; r < 5; r++) {
      hline(571, 575, r * pitch - 1);
    }

    for (final (gx0, x1, r0, r1) in groups(layout, pieces)) {
      final yt = top(r0), yb = bottom(r1);
      final x0 = max(0, gx0);
      final arm = ((x1 - x0) ~/ 4).clamp(6, 14);
      // No room left of text at the very edge: there the corner is flat.
      void left(int y0, int y1) {
        if (gx0 >= 0) vline(x0, y0, y1);
      }

      switch (style) {
        case 'underline':
          hline(x0, x0 + 16, yb);
          hline(x0 + 20, x1, yb, every: 3);
          vline(x1, yb - 3, yb);
        default: // hud, corners
          hline(x0, x0 + arm, yt);
          left(yt, yt + 4);
          hline(x1 - arm, x1, yb);
          vline(x1, yb - 4, yb);
          if (style == 'corners') {
            hline(x1 - arm, x1, yt);
            vline(x1, yt, yt + 4);
            hline(x0, x0 + arm, yb);
            left(yb - 4, yb);
          }
      }
    }
    return px;
  }

  /// The picture to send, or null for "off".
  static Uint8List? bmp(String style, LookLayout layout, List<LookPiece> pieces) =>
      style == 'off' ? null : MonoEncoder.bmp(lit(style, layout, pieces));
}
