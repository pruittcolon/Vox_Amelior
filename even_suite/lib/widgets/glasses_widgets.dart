import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/ble/ble_manager.dart';
import '../core/glasses/claude_panel.dart';
import '../core/glasses/g1_font.dart';
import '../core/glasses/glasses_display.dart';
import 'common.dart';

/// The glasses' battery (the lower arm), e.g. "73%" - only while they are
/// connected and have told it. Rebuilds itself when it changes.
class GlassesBattery extends StatelessWidget {
  const GlassesBattery({super.key, required this.ble, this.compact = true});
  final BleManager ble;

  /// Just the icon and the number (headers); otherwise a pill with a label.
  final bool compact;

  static IconData icon(int pct) => pct >= 90
      ? Icons.battery_full_rounded
      : pct >= 65
          ? Icons.battery_5_bar_rounded
          : pct >= 40
              ? Icons.battery_4_bar_rounded
              : pct >= 20
                  ? Icons.battery_2_bar_rounded
                  : Icons.battery_alert_rounded;

  static Color color(int pct) => pct >= 40 ? AppColors.ok : pct >= 20 ? AppColors.warn : AppColors.error;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
        listenable: ble,
        builder: (context, _) {
          final pct = ble.battery;
          if (!ble.isConnected || pct == null) return const SizedBox.shrink();
          final c = color(pct);
          final arms = [
            if (ble.batteryL != null) 'left ${ble.batteryL}%',
            if (ble.batteryR != null) 'right ${ble.batteryR}%',
          ].join(', ');
          final row = Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon(pct), size: compact ? 18 : 16, color: c),
            const SizedBox(width: 3),
            Text('$pct%', style: TextStyle(color: c, fontWeight: FontWeight.w700, fontSize: compact ? 13 : 12.5)),
          ]);
          return Tooltip(
            message: 'Glasses battery: $arms',
            child: compact
                ? Padding(padding: const EdgeInsets.symmetric(horizontal: 6), child: row)
                : Container(
                    padding: const EdgeInsets.fromLTRB(10, 5, 12, 5),
                    decoration: BoxDecoration(
                      color: c.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: c.withOpacity(0.35)),
                    ),
                    child: row,
                  ),
          );
        },
      );
}

/// Simple line drawing of the G1 (two rounded lenses and a bridge).
class GlassesGlyph extends StatelessWidget {
  const GlassesGlyph({super.key, this.color = AppColors.text, this.size = 56, this.left = true, this.right = true});

  final Color color;
  final double size;
  final bool left;
  final bool right;

  @override
  Widget build(BuildContext context) => CustomPaint(
        size: Size(size, size * 0.42),
        painter: _GlassesPainter(color, left, right),
      );
}

class _GlassesPainter extends CustomPainter {
  _GlassesPainter(this.color, this.left, this.right);
  final Color color;
  final bool left;
  final bool right;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width, h = size.height;
    final stroke = w * 0.06;
    Paint p(bool on) => Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..strokeCap = StrokeCap.round
      ..color = on ? color : color.withOpacity(0.25);
    final lens = Size(w * 0.40, h * 0.86);
    final r = Radius.circular(h * 0.38);
    canvas.drawRRect(RRect.fromRectAndRadius(Offset(stroke / 2, stroke / 2) & lens, r), p(left));
    canvas.drawRRect(
        RRect.fromRectAndRadius(Offset(w - lens.width - stroke / 2, stroke / 2) & lens, r), p(right));
    final bridge = Path()
      ..moveTo(lens.width + stroke / 2, h * 0.32)
      ..quadraticBezierTo(w / 2, h * 0.08, w - lens.width - stroke / 2, h * 0.32);
    canvas.drawPath(bridge, p(left && right));
  }

  @override
  bool shouldRepaint(_GlassesPainter old) => old.color != color || old.left != left || old.right != right;
}

/// What is on the glasses right now, drawn like the G1's green display,
/// with page controls.
class GlassesPreview extends StatelessWidget {
  const GlassesPreview({super.key, required this.display, this.connected = true});

  final GlassesDisplay display;
  final bool connected;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: display,
      builder: (context, _) {
        final lines = display.previewLines;
        final idle = display.mode == DisplayMode.idle;
        return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          // Fixed size like the real display, whatever the phone's text size.
          MediaQuery.withNoTextScaling(
           child: Container(
            height: 150,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            decoration: BoxDecoration(
              color: AppColors.hudBg,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.hud.withOpacity(0.18)),
            ),
            child: idle
                ? Center(
                    child: Text(
                      connected ? 'Nothing on the glasses' : 'Glasses not connected',
                      style: TextStyle(color: AppColors.hud.withOpacity(0.45), fontSize: 14),
                    ),
                  )
                // Lines exactly as the glasses break them, shrunk to fit.
                : FittedBox(
                    fit: BoxFit.scaleDown,
                    alignment: Alignment.centerLeft,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        for (final l in lines)
                          Text(l,
                              softWrap: false,
                              style: const TextStyle(color: AppColors.hud, fontSize: 14, height: 1.45)),
                      ],
                    ),
                  ),
           ),
          ),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(
              child: Text(
                idle ? 'Tap a reply to show it here' : display.title,
                style: const TextStyle(color: AppColors.muted, fontSize: 13),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (display.canScroll)
              IconButton(
                tooltip: display.scrolling ? 'Pause scrolling' : 'Scroll at reading speed',
                onPressed: display.togglePause,
                icon: Icon(display.scrolling ? Icons.pause_circle_rounded : Icons.play_circle_rounded,
                    color: AppColors.hud),
              ),
            PageButtons(
              compact: true,
              label: display.pageLabel,
              onUp: display.canUp ? display.up : null,
              onDown: display.canDown ? display.down : null,
            ),
            const SizedBox(width: 4),
            IconButton(
              tooltip: 'Clear the glasses',
              onPressed: idle ? null : display.clear,
              icon: const Icon(Icons.layers_clear_rounded),
            ),
          ]),
          if (display.error != null && connected)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(display.error!, style: const TextStyle(color: AppColors.warn, fontSize: 12.5)),
            ),
        ]);
      },
    );
  }
}

/// The G1 look-up screen (dual layout) drawn the way the firmware lays it
/// out: date, clock and weather on the left, a divider at x 172, and the
/// right panel - an icon, a title line and up to four body lines, every
/// character at the width the G1 font gives it. [guides] outlines the
/// firmware's boxes, to check the spacing.
class GlassesDashboardView extends StatelessWidget {
  const GlassesDashboardView({
    super.key,
    required this.now,
    required this.title,
    required this.rows,
    this.h24 = false,
    this.temp = '',
    this.paneLabel,
    this.guides = false,
  });

  final DateTime now;
  final String title;

  /// The body lines (dim ones drawn darker, as the glasses do).
  final List<PanelLine> rows;
  final bool h24;
  final String temp;

  /// For the glasses' own panels (calendar, news...): just their name.
  final String? paneLabel;
  final bool guides;

  @override
  Widget build(BuildContext context) => AspectRatio(
        aspectRatio: DashLayout.width / DashLayout.height,
        child: CustomPaint(
          painter: _DashPainter(this),
        ),
      );
}

class _DashPainter extends CustomPainter {
  _DashPainter(this.v);
  final GlassesDashboardView v;

  static const _days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  void _text(Canvas c, String s, double x, double y, double size, Color color, {FontWeight? weight}) {
    final tp = TextPainter(
      text: TextSpan(
          text: s,
          style: TextStyle(fontFamily: 'Roboto', fontSize: size, color: color, height: 1.0, fontWeight: weight)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(c, Offset(x, y));
    tp.dispose();
  }

  /// One line in the G1 font's spacing: each character starts where the
  /// glasses would put it.
  void _g1(Canvas c, String line, double x, double y, Color color) {
    var cx = x;
    for (final rune in line.runes) {
      final adv = G1Font.advance(rune);
      if (adv > 0 && rune != 0x20) _text(c, String.fromCharCode(rune), cx, y + 3, 19, color);
      cx += adv;
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    final s = size.width / DashLayout.width;
    canvas.save();
    canvas.scale(s);
    const w = 576.0, h = 136.0;
    canvas.drawRect(const Rect.fromLTWH(0, 0, w, h), Paint()..color = AppColors.hudBg);
    const on = AppColors.hud;
    final dimColor = AppColors.hud.withOpacity(0.45);
    if (v.guides) {
      final g = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1
        ..color = const Color(0x55FF8A65);
      canvas.drawRect(const Rect.fromLTRB(0, 0, DashLayout.dividerX + 0.0, h), g);
      canvas.drawRect(const Rect.fromLTRB(DashLayout.titleX + 0.0, DashLayout.titleY + 0.0, DashLayout.titleRight + 0.0, 28), g);
      canvas.drawRect(const Rect.fromLTRB(DashLayout.bodyX + 0.0, DashLayout.bodyY + 0.0, DashLayout.bodyRight + 0.0, h), g);
      for (var k = 1; k < DashLayout.bodyLines; k++) {
        final y = DashLayout.bodyY + k * DashLayout.lineHeight + 0.0;
        canvas.drawLine(Offset(DashLayout.bodyX + 0.0, y), Offset(DashLayout.bodyRight + 0.0, y), g);
      }
    }
    // Left: date, clock, weather (firmware boxes: date 0..140 x 2..29,
    // clock 0..150 x 52..85, weather icon at 0,109 and temperature at 28,109).
    final hour = v.h24 ? v.now.hour : (v.now.hour % 12 == 0 ? 12 : v.now.hour % 12);
    final clock = '${v.h24 ? hour.toString().padLeft(2, '0') : hour}:${v.now.minute.toString().padLeft(2, '0')}';
    _text(canvas, '${_days[v.now.weekday - 1]}, ${v.now.month.toString().padLeft(2, '0')}/${v.now.day.toString().padLeft(2, '0')}',
        0, 6, 17, on);
    _text(canvas, clock, 0, 46, 38, on, weight: FontWeight.w600);
    if (v.temp.isNotEmpty) {
      canvas.drawCircle(const Offset(11, 121), 9, Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = on);
      _text(canvas, v.temp, 28, 110, 18, on);
    }
    canvas.drawLine(const Offset(DashLayout.dividerX + 0.5, 4), const Offset(DashLayout.dividerX + 0.5, h - 4),
        Paint()
          ..color = on.withOpacity(0.6)
          ..strokeWidth = 1.5);
    // Right panel.
    if (v.paneLabel != null) {
      _text(canvas, v.paneLabel!, DashLayout.bodyX + 0.0, 50, 20, dimColor);
    } else {
      final icon = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = on;
      canvas.drawRRect(
          RRect.fromRectAndRadius(const Rect.fromLTWH(DashLayout.iconX + 1.0, 5, 18, 20), const Radius.circular(3)), icon);
      canvas.drawLine(const Offset(DashLayout.iconX + 5.0, 11), const Offset(DashLayout.iconX + 15.0, 11), icon);
      canvas.drawLine(const Offset(DashLayout.iconX + 5.0, 17), const Offset(DashLayout.iconX + 13.0, 17), icon);
      _g1(canvas, v.title, DashLayout.titleX + 0.0, DashLayout.titleY + 0.0, on);
      for (var k = 0; k < v.rows.length && k < DashLayout.bodyLines; k++) {
        _g1(canvas, v.rows[k].text, DashLayout.bodyX + 0.0, DashLayout.bodyY + k * DashLayout.lineHeight + 0.0,
            v.rows[k].dim ? dimColor : on);
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(_DashPainter old) => true;
}
