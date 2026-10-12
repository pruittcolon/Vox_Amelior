import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/glasses/glasses_display.dart';
import 'common.dart';

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
                idle ? 'Tap a reply or say “Claude, show the reply”' : display.title,
                style: const TextStyle(color: AppColors.muted, fontSize: 13),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
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
