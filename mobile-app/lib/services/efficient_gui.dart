import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:demo_ai_even/services/evenai_proto.dart';
import 'package:demo_ai_even/services/fast_bmp.dart';

/// What a HUD screen shows. Kept tiny on purpose: a header, a few lines,
/// an optional progress bar.
class HudState {
  final String title;
  final String clock;
  final int battery;
  final List<String> lines;
  final int selected; // index into [lines] marked with '>', -1 for none
  final double? progress; // 0..1, null hides the bar

  const HudState({
    required this.title,
    required this.clock,
    required this.battery,
    this.lines = const [],
    this.selected = -1,
    this.progress,
  });
}

/// Text HUD: the glasses draw the characters themselves, so a whole screen is
/// a few hundred bytes sent with the Text Sending command (0x4E). Lives in
/// RAM on the glasses, no flash writes.
class TextHud {
  /// Characters per line. The G1 text area is 488 px at font size 21;
  /// 38 keeps wide letters from wrapping.
  static const int cols = 38;
  static const int rows = 5;

  /// Plain ASCII only: the built-in font's symbol coverage is unverified.
  static String _ascii(String s) =>
      String.fromCharCodes(s.runes.map((c) => c >= 0x20 && c < 0x7f ? c : 0x3f));

  static String _clip(String s) =>
      s.length <= cols ? s : '${s.substring(0, cols - 2)}..';

  static String _spread(String left, String mid, String right) {
    final gap = cols - left.length - mid.length - right.length;
    if (gap < 2) return _clip('$left $mid $right');
    final l = gap ~/ 2;
    return '$left${' ' * l}$mid${' ' * (gap - l)}$right';
  }

  static String progressBar(double p, {int width = 20}) {
    final n = (p.clamp(0.0, 1.0) * width).round();
    return '[${'#' * n}${'-' * (width - n)}] ${(p * 100).round()}%';
  }

  /// At most [rows] lines of at most [cols] characters.
  static String render(HudState s) {
    final out = <String>[
      _spread(_ascii(s.clock), _ascii(s.title), 'BAT ${s.battery}%'),
    ];
    final bodyRows = rows - 1 - (s.progress == null ? 0 : 1);
    // Keep the selected line visible.
    final first = s.selected < bodyRows ? 0 : s.selected - bodyRows + 1;
    for (var i = first; i < min(s.lines.length, first + bodyRows); i++) {
      out.add(_clip('${i == s.selected ? '>' : ' '} ${_ascii(s.lines[i])}'));
    }
    if (s.progress != null) out.add(progressBar(s.progress!));
    return out.join('\n');
  }

  /// The exact BLE packets for one screen (sent to each arm).
  static List<Uint8List> packets(String screen) =>
      EvenaiProto.evenaiMultiPackListV2(0x4E,
          data: utf8.encode(screen),
          syncSeq: 0,
          newScreen: 0x71,
          pos: 0,
          current_page_num: 1,
          max_page_num: 1);
}

/// Bitmap HUD built around how the G1 stores images: the BMP file is written
/// to flash in 4 KB blocks starting with the header and the *bottom* rows, and
/// blocks that are not sent keep their old content. So the live part sits in a
/// strip at the bottom and an update only sends header + strip rows.
///
/// Layout (576x136):
///   y 0..[topEnd)            static: title + labels, sent with a full frame
///   y [topEnd)..[stripTop)   always blank (shares the first flash block with
///                            the strip and gets erased on each strip update)
///   y [stripTop)..136        live strip
class StripHud {
  static const int headerBytes = 62;
  static const int blockBytes = 4096;

  /// Rows (from the top) fully outside the first 4 KB flash block.
  static int get topEnd =>
      G1Canvas.height - (blockBytes - headerBytes + G1Canvas.stride - 1) ~/ G1Canvas.stride;

  final int stripRows;
  StripHud({this.stripRows = 20})
      : assert(stripRows * G1Canvas.stride + headerBytes <= blockBytes);

  int get stripTop => G1Canvas.height - stripRows;

  /// Full frame: static top + live strip.
  Uint8List frame(HudState s) {
    final c = G1Canvas();
    c.text(4, 2, s.title, scale: 2);
    c.text(G1Canvas.width - 6 * 2 * 8 - 4, 2, 'BAT ${s.battery}%', scale: 2);
    c.rect(0, 20, G1Canvas.width, 21);
    for (var i = 0; i < min(3, s.lines.length); i++) {
      c.text(4, 26 + i * 17, '${i == s.selected ? '>' : ' '} ${s.lines[i]}', scale: 2);
    }
    _strip(c, s);
    return c.toBmp();
  }

  void _strip(G1Canvas c, HudState s) {
    final y = stripTop + (stripRows - 14) ~/ 2;
    c.text(4, y, s.clock, scale: 2);
    if (s.progress != null) {
      c.rect(120, y, 572, y + 14, fill: false);
      c.rect(122, y + 2, 122 + ((572 - 124) * s.progress!.clamp(0.0, 1.0)).round(), y + 12);
    }
  }

  /// Bytes for a strip-only update: BMP header + the strip's rows.
  Uint8List stripUpdate(Uint8List fullFrame) =>
      fullFrame.sublist(0, headerBytes + stripRows * G1Canvas.stride);

  /// Smallest payload that turns [shown] into [next]: nothing, the strip, or
  /// the whole frame when the static part changed.
  Uint8List? delta(Uint8List? shown, Uint8List next) {
    if (shown != null && _same(shown, next, 0, next.length)) return null;
    final stripEnd = headerBytes + stripRows * G1Canvas.stride;
    if (shown != null && _same(shown, next, stripEnd, next.length)) {
      return stripUpdate(next);
    }
    return next;
  }

  static bool _same(Uint8List a, Uint8List b, int from, int to) {
    for (var i = from; i < to; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Rough per-update time on the G1, from the parts measured in the firmware
/// and datasheets. Used to compare designs, not a measurement.
class G1Cost {
  /// Air time for one packet on the 1M PHY incl. headers, ack and gaps.
  static double airMs(int bytes) => (bytes + 14) * 8 / 1000 + 0.4;

  /// Bitmap upload: packets to both arms, end + CRC replies, flash erase +
  /// program of each touched 4 KB block, wait for the 20 Hz refresh.
  static double bitmapMs(int payloadBytes) {
    final packs = FastBmp.packets(Uint8List(payloadBytes));
    final air = packs.fold<double>(0, (t, p) => t + airMs(p.length)) * 2;
    final blocks = (payloadBytes / StripHud.blockBytes).ceil();
    final pages = (payloadBytes / 256).ceil();
    return air + 2 * 30 + blocks * 40 + pages * 0.85 + 25;
  }

  /// Text screen: packets to the left arm, ack, then the right arm, ack.
  static double textMs(List<Uint8List> packs) =>
      packs.fold<double>(0, (t, p) => t + airMs(p.length) + 30) * 2 + 25;
}
