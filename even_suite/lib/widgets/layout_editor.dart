import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/glasses/claude_panel.dart';
import '../core/glasses/g1_font.dart';

/// Drag-and-drop editor for one of Claude's glasses screens: three blocks
/// (what you asked, Claude, an info line) in any order. Your question takes
/// 0-3 lines, the info line is on or off, and Claude gets the rest. Each
/// block is drawn as tall as the lines it takes, in the glasses' green.
class LayoutEditor extends StatelessWidget {
  const LayoutEditor({super.key, required this.layout, required this.onChanged, required this.labels});

  final PanelLayout layout;
  final ValueChanged<PanelLayout> onChanged;

  /// Block -> (name, what it shows).
  final Map<String, (String, String)> labels;

  @override
  Widget build(BuildContext context) {
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Text(
        'Hold and drag a block to move it. ${layout.total} lines: your question ${layout.askLines}, '
        '${labels['meta']!.$1.toLowerCase()} ${layout.metaLines}, Claude the rest (${layout.claudeLines}).',
        style: const TextStyle(color: AppColors.muted, fontSize: 13),
      ),
      const SizedBox(height: 10),
      Container(
        padding: const EdgeInsets.all(6),
        decoration: BoxDecoration(
          color: AppColors.hudBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.hud.withOpacity(0.2)),
        ),
        child: ReorderableListView(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          buildDefaultDragHandles: false,
          proxyDecorator: (child, _, __) => Material(color: Colors.transparent, elevation: 6, child: child),
          onReorder: (from, to) {
            final order = List.of(layout.order);
            if (to > from) to--;
            order.insert(to, order.removeAt(from));
            onChanged(layout.copyWith(order: order));
          },
          children: [
            for (var i = 0; i < layout.order.length; i++)
              _Block(
                key: ValueKey(layout.order[i]),
                index: i,
                block: layout.order[i],
                layout: layout,
                label: labels[layout.order[i]]!,
                onChanged: onChanged,
              ),
          ],
        ),
      ),
    ]);
  }
}

class _Block extends StatelessWidget {
  const _Block({
    super.key,
    required this.index,
    required this.block,
    required this.layout,
    required this.label,
    required this.onChanged,
  });

  final int index;
  final String block;
  final PanelLayout layout;
  final (String, String) label;
  final ValueChanged<PanelLayout> onChanged;

  @override
  Widget build(BuildContext context) {
    final lines = switch (block) { 'asked' => layout.askLines, 'meta' => layout.metaLines, _ => layout.claudeLines };
    final dim = block != 'claude';
    final color = dim ? AppColors.hud.withOpacity(0.55) : AppColors.hud;
    final hidden = lines == 0;
    return ReorderableDelayedDragStartListener(
      index: index,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 3),
        constraints: BoxConstraints(minHeight: 30.0 + 18.0 * lines),
        padding: const EdgeInsets.fromLTRB(2, 6, 10, 6),
        decoration: BoxDecoration(
          color: hidden ? Colors.transparent : color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(hidden ? 0.25 : 0.6)),
        ),
        child: Row(children: [
          ReorderableDragStartListener(
            index: index,
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Icon(Icons.drag_handle_rounded, color: color),
            ),
          ),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
              Text(label.$1, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
              Text(hidden ? 'Hidden' : label.$2, style: TextStyle(color: color.withOpacity(0.8), fontSize: 12.5)),
              const SizedBox(height: 4),
              if (block == 'asked')
                Wrap(spacing: 6, runSpacing: 4, children: [
                  for (final n in const [0, 1, 2, 3])
                    if (n < layout.total)
                      ChoiceChip(
                        label: Text(n == 0 ? 'Hide' : '$n'),
                        visualDensity: VisualDensity.compact,
                        selected: layout.askLines == n,
                        onSelected: (_) => onChanged(layout.copyWith(askLines: n)),
                      ),
                ])
              else if (block == 'meta')
                Wrap(spacing: 6, children: [
                  for (final n in const [0, 1])
                    ChoiceChip(
                      label: Text(n == 0 ? 'Hide' : 'Show'),
                      visualDensity: VisualDensity.compact,
                      selected: layout.metaLines == n,
                      onSelected: (_) => onChanged(layout.copyWith(metaLines: n)),
                    ),
                ])
              else
                Text('$lines line${lines == 1 ? '' : 's'} - whatever is left',
                    style: TextStyle(color: color, fontSize: 12.5, fontWeight: FontWeight.w600)),
            ]),
          ),
        ]),
      ),
    );
  }
}

/// The screen in front of you (the glasses' main text area: five lines from
/// x 88, as the Even AI screen draws them), in the G1 font's spacing.
class GlassesScreenView extends StatelessWidget {
  const GlassesScreenView({super.key, required this.rows});

  /// The lines; dim ones (your question) are drawn darker here.
  final List<PanelLine> rows;

  @override
  Widget build(BuildContext context) => AspectRatio(
        aspectRatio: 576 / 136,
        child: CustomPaint(painter: _ScreenPainter(rows)),
      );
}

class _ScreenPainter extends CustomPainter {
  _ScreenPainter(this.rows);
  final List<PanelLine> rows;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / 576);
    canvas.drawRect(const Rect.fromLTWH(0, 0, 576, 136), Paint()..color = AppColors.hudBg);
    for (var k = 0; k < rows.length && k < 5; k++) {
      var x = 88.0;
      final color = rows[k].dim ? AppColors.hud.withOpacity(0.5) : AppColors.hud;
      for (final rune in rows[k].text.runes) {
        final adv = G1Font.advance(rune);
        if (adv > 0 && rune != 0x20) {
          final tp = TextPainter(
            text: TextSpan(
                text: String.fromCharCode(rune),
                style: TextStyle(fontFamily: 'Roboto', fontSize: 19, height: 1.0, color: color)),
            textDirection: TextDirection.ltr,
          )..layout();
          tp.paint(canvas, Offset(x, 3.0 + k * 27));
          tp.dispose();
        }
        x += adv;
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(_ScreenPainter old) => true;
}
