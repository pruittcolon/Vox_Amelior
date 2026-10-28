import 'dart:typed_data';

import '../screen/mono_encoder.dart';

/// A thin "HUD" frame for your look-up screen, sent as a picture right after
/// its text: the glasses keep the text and draw the picture over it, and only
/// lit pixels show, so the frame sits around the text without covering it.
///
/// Styles: corners (brackets in the four corners), frame (brackets, edge
/// ticks and dotted rules top and bottom), grid (frame plus dotted guides
/// between the five text rows).
class HudFrame {
  HudFrame._();

  static const styles = [('corners', 'Corners'), ('frame', 'Frame'), ('grid', 'Grid'), ('off', 'Off')];

  static const w = MonoEncoder.width, h = MonoEncoder.height;

  /// Lit flags (1 = lit), [w] x [h].
  static Uint8List lit(String style) {
    final px = Uint8List(w * h);
    void dot(int x, int y) {
      if (x >= 0 && x < w && y >= 0 && y < h) px[y * w + x] = 1;
    }

    void hline(int x0, int x1, int y, {int every = 1}) {
      for (var x = x0; x <= x1; x += every) {
        dot(x, y);
      }
    }

    void vline(int x, int y0, int y1, {int every = 1}) {
      for (var y = y0; y <= y1; y += every) {
        dot(x, y);
      }
    }

    // Corner brackets: 2 px thick, 28 across, 16 down.
    const len = 28, down = 16;
    for (var t = 0; t < 2; t++) {
      hline(0, len, t);
      vline(t, 0, down);
      hline(w - 1 - len, w - 1, t);
      vline(w - 1 - t, 0, down);
      hline(0, len, h - 1 - t);
      vline(t, h - 1 - down, h - 1);
      hline(w - 1 - len, w - 1, h - 1 - t);
      vline(w - 1 - t, h - 1 - down, h - 1);
    }
    if (style == 'corners') return px;

    // Dotted rules between the brackets, a short solid mark in the middle,
    // and tick marks down both edges.
    hline(len + 8, w - len - 9, 0, every: 4);
    hline(len + 8, w - len - 9, h - 1, every: 4);
    for (var t = 0; t < 2; t++) {
      hline(w ~/ 2 - 20, w ~/ 2 + 20, t);
      hline(w ~/ 2 - 20, w ~/ 2 + 20, h - 1 - t);
    }
    for (var y = down + 8; y < h - down - 8; y += 9) {
      hline(0, 4, y);
      hline(w - 5, w - 1, y);
    }
    if (style == 'frame') return px;

    // Guides between the five text rows (the glasses' rows are ~27 px).
    for (var r = 1; r < 5; r++) {
      final y = r * h ~/ 5;
      hline(12, w - 13, y, every: 6);
    }
    return px;
  }

  /// The picture to send, or null for "off".
  static Uint8List? bmp(String style) => style == 'off' ? null : MonoEncoder.bmp(lit(style));
}
