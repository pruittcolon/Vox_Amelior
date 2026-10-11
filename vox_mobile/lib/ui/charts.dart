import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Colours for tones of voice. Each emotion keeps its colour everywhere;
/// [order] is also the stacking order, which was checked so neighbours
/// stay apart for colour-blind readers, in light and dark mode.
abstract final class MoodColors {
  /// Emotions in stacking order (neutral is never stacked).
  static const List<String> order = ['angry', 'sad', 'happy', 'fearful', 'surprised', 'disgusted'];

  static const Map<String, Color> _light = {
    'angry': Color(0xFFE34948),
    'sad': Color(0xFF2A78D6),
    'happy': Color(0xFFEDA100),
    'fearful': Color(0xFF1BAF7A),
    'surprised': Color(0xFF4A3AA7),
    'disgusted': Color(0xFF008300),
  };
  static const Map<String, Color> _dark = {
    'angry': Color(0xFFE66767),
    'sad': Color(0xFF3987E5),
    'happy': Color(0xFFC98500),
    'fearful': Color(0xFF199E70),
    'surprised': Color(0xFF9085E9),
    'disgusted': Color(0xFF008300),
  };

  static Color of(BuildContext context, String tone) {
    final t = Theme.of(context);
    if (tone == 'neutral' || tone == 'laughter') return t.colorScheme.outlineVariant;
    return (t.brightness == Brightness.dark ? _dark : _light)[tone] ?? t.colorScheme.outline;
  }
}

/// Small coloured dot that carries a series' identity next to its label.
class Swatch extends StatelessWidget {
  const Swatch(this.color, {super.key, this.size = 10});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) =>
      Container(width: size, height: size, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(size / 3)));
}

/// A headline number: label, value and an optional change against the
/// previous period.
class StatTile extends StatelessWidget {
  const StatTile({super.key, required this.label, required this.value, this.icon, this.delta, this.onTap});

  final String label;
  final String value;
  final IconData? icon;

  /// Relative change vs the previous period (0.12 = +12%); null hides it.
  final double? delta;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final d = delta;
    return Card(
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  if (icon != null) ...[Icon(icon, size: 16, color: t.colorScheme.onSurfaceVariant), const SizedBox(width: 6)],
                  Expanded(
                    child: Text(label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: t.textTheme.labelMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              FittedBox(
                fit: BoxFit.scaleDown,
                alignment: Alignment.centerLeft,
                child: Text(value, style: t.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
              ),
              if (d != null) ...[
                const SizedBox(height: 2),
                Text(
                  d.abs() < 0.005 ? 'same as before' : '${d > 0 ? '▲' : '▼'} ${(d.abs() * 100).round()}% vs before',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

/// A thin 100% bar split by tone, e.g. under a conversation or a person.
/// Neutral is left out; nothing is drawn without any feeling.
class MoodStrip extends StatelessWidget {
  const MoodStrip({super.key, required this.counts, this.height = 6});

  final Map<String, int> counts;
  final double height;

  @override
  Widget build(BuildContext context) {
    final parts = [for (final m in MoodColors.order) if ((counts[m] ?? 0) > 0) (m, counts[m]!)];
    if (parts.isEmpty) return const SizedBox.shrink();
    final total = parts.fold<int>(0, (a, p) => a + p.$2);
    final surface = Theme.of(context).colorScheme.surface;
    return Semantics(
      label: parts.map((p) => '${p.$2} ${p.$1}').join(', '),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(height / 2),
        child: SizedBox(
          height: height,
          child: Row(
            children: [
              for (var i = 0; i < parts.length; i++) ...[
                if (i > 0) Container(width: 2, color: surface),
                Expanded(flex: math.max(1, (parts[i].$2 * 1000 / total).round()), child: Container(color: MoodColors.of(context, parts[i].$1))),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

/// Columns of stacked tones over time. Tap a column to select it.
class StackedColumnChart extends StatelessWidget {
  const StackedColumnChart({
    super.key,
    required this.columns,
    required this.labels,
    this.selected,
    this.onSelect,
    this.height = 160,
  });

  /// Per column: tone → count, stacked in [MoodColors.order].
  final List<Map<String, int>> columns;

  /// Axis labels for a few columns (index → text).
  final Map<int, String> labels;
  final int? selected;
  final ValueChanged<int?>? onSelect;
  final double height;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final colors = {for (final m in MoodColors.order) m: MoodColors.of(context, m)};
    return LayoutBuilder(builder: (context, box) {
      final painter = _StackPainter(
        columns: columns,
        colors: colors,
        selected: selected,
        surface: t.cardTheme.color ?? t.colorScheme.surfaceContainerLow,
        grid: t.colorScheme.outlineVariant.withValues(alpha: 0.5),
        ink: t.colorScheme.onSurfaceVariant,
        labels: labels,
        labelStyle: t.textTheme.labelSmall!.copyWith(color: t.colorScheme.onSurfaceVariant),
      );
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTapDown: onSelect == null
            ? null
            : (d) {
                final i = painter.indexAt(d.localPosition.dx, box.maxWidth);
                onSelect!(i == selected ? null : i);
              },
        child: CustomPaint(size: Size(box.maxWidth, height), painter: painter),
      );
    });
  }
}

class _StackPainter extends CustomPainter {
  _StackPainter({
    required this.columns,
    required this.colors,
    required this.selected,
    required this.surface,
    required this.grid,
    required this.ink,
    required this.labels,
    required this.labelStyle,
  });

  final List<Map<String, int>> columns;
  final Map<String, Color> colors;
  final int? selected;
  final Color surface;
  final Color grid;
  final Color ink;
  final Map<int, String> labels;
  final TextStyle labelStyle;

  static const double _axis = 18; // room for x labels
  static const double _left = 28; // room for y labels

  int get _max => columns.fold<int>(0, (a, c) => math.max(a, MoodColors.order.fold<int>(0, (s, m) => s + (c[m] ?? 0))));

  double _slot(double width) => columns.isEmpty ? 0 : (width - _left) / columns.length;

  /// Column under x, or null.
  int? indexAt(double x, double width) {
    if (columns.isEmpty || x < _left) return null;
    return ((x - _left) / _slot(width)).floor().clamp(0, columns.length - 1);
  }

  /// A round number at or above [v] for the top gridline.
  static int niceCeil(int v) {
    if (v <= 4) return math.max(v, 1);
    final p = math.pow(10, (math.log(v) / math.ln10).floor()).toInt();
    for (final m in [1, 2, 5, 10]) {
      if (m * p >= v) return m * p;
    }
    return 10 * p;
  }

  void _text(Canvas canvas, String s, Offset at, {TextAlign align = TextAlign.center}) {
    final tp = TextPainter(text: TextSpan(text: s, style: labelStyle), textDirection: TextDirection.ltr)..layout();
    final dx = switch (align) {
      TextAlign.right => at.dx - tp.width,
      TextAlign.left => at.dx,
      _ => at.dx - tp.width / 2,
    };
    tp.paint(canvas, Offset(dx, at.dy - tp.height / 2));
  }

  @override
  void paint(Canvas canvas, Size size) {
    final plotH = size.height - _axis;
    final top = niceCeil(_max);
    final gridPaint = Paint()
      ..color = grid
      ..strokeWidth = 1;
    // Recessive hairlines at 0, half and top, with clean labels.
    for (final f in [0.0, 0.5, 1.0]) {
      final y = plotH - plotH * f + 0.5;
      canvas.drawLine(Offset(_left, y), Offset(size.width, y), gridPaint);
      final v = (top * f).round();
      if (f == 0 || v != 0) _text(canvas, '$v', Offset(_left - 6, y), align: TextAlign.right);
    }
    if (columns.isEmpty) return;
    final slot = _slot(size.width);
    final barW = math.min(24.0, math.max(2.0, slot * 0.62));
    for (var i = 0; i < columns.length; i++) {
      final c = columns[i];
      final x = _left + slot * i + (slot - barW) / 2;
      final dim = selected != null && selected != i;
      var y = plotH;
      final present = [for (final m in MoodColors.order) if ((c[m] ?? 0) > 0) m];
      for (var k = 0; k < present.length; k++) {
        final m = present[k];
        final h = plotH * c[m]! / top;
        final isTop = k == present.length - 1;
        // 2px surface gap between stacked pieces; 4px rounded data end on top only.
        final rect = Rect.fromLTWH(x, y - h, barW, math.max(0, h - (isTop ? 0 : 2)));
        final paint = Paint()..color = colors[m]!.withValues(alpha: dim ? 0.3 : 1);
        if (isTop) {
          canvas.drawRRect(
            RRect.fromRectAndCorners(rect, topLeft: Radius.circular(math.min(4, barW / 2)), topRight: Radius.circular(math.min(4, barW / 2))),
            paint,
          );
        } else {
          canvas.drawRect(rect, paint);
        }
        y -= h;
      }
      final label = labels[i];
      if (label != null) _text(canvas, label, Offset(x + barW / 2, plotH + _axis / 2 + 2));
    }
  }

  @override
  bool shouldRepaint(_StackPainter old) => old.columns != columns || old.selected != selected || old.colors != colors || old.surface != surface;
}

/// Talk time by weekday (rows, Monday first) and hour (columns): one hue,
/// darker is more. Tap a cell to select it.
class WeekHourHeatmap extends StatelessWidget {
  const WeekHourHeatmap({super.key, required this.seconds, this.selected, this.onSelect});

  /// 7 × 24 values, Monday 00:00 first.
  final List<int> seconds;
  final int? selected;
  final ValueChanged<int?>? onSelect;

  static const List<String> days = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final style = t.textTheme.labelSmall!.copyWith(color: t.colorScheme.onSurfaceVariant);
    const left = 16.0;
    return LayoutBuilder(builder: (context, box) {
      final cell = (box.maxWidth - left) / 24;
      final rowH = math.min(22.0, math.max(10.0, cell * 1.3));
      final peak = seconds.fold<int>(0, math.max);
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTapDown: onSelect == null
            ? null
            : (d) {
                final c = ((d.localPosition.dx - left) / cell).floor();
                final r = (d.localPosition.dy / rowH).floor();
                if (c < 0 || c > 23 || r < 0 || r > 6) return;
                final i = r * 24 + c;
                onSelect!(i == selected ? null : i);
              },
        child: CustomPaint(
          size: Size(box.maxWidth, rowH * 7 + 18),
          painter: _HeatPainter(
            seconds: seconds,
            peak: peak,
            cell: cell,
            rowH: rowH,
            left: left,
            selected: selected,
            empty: t.colorScheme.surfaceContainerHighest,
            hue: t.colorScheme.primary,
            ring: t.colorScheme.onSurface,
            style: style,
          ),
        ),
      );
    });
  }
}

class _HeatPainter extends CustomPainter {
  _HeatPainter({
    required this.seconds,
    required this.peak,
    required this.cell,
    required this.rowH,
    required this.left,
    required this.selected,
    required this.empty,
    required this.hue,
    required this.ring,
    required this.style,
  });

  final List<int> seconds;
  final int peak;
  final double cell;
  final double rowH;
  final double left;
  final int? selected;
  final Color empty;
  final Color hue;
  final Color ring;
  final TextStyle style;

  void _text(Canvas canvas, String s, Offset center) {
    final tp = TextPainter(text: TextSpan(text: s, style: style), textDirection: TextDirection.ltr)..layout();
    tp.paint(canvas, center - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (var r = 0; r < 7; r++) {
      _text(canvas, WeekHourHeatmap.days[r], Offset(left / 2 - 2, r * rowH + rowH / 2));
      for (var c = 0; c < 24; c++) {
        final v = seconds[r * 24 + c];
        // Square-root scale so quiet hours still show; zero stays the empty tone.
        final f = peak == 0 || v == 0 ? 0.0 : 0.18 + 0.82 * math.sqrt(v / peak);
        final color = v == 0 ? empty : Color.lerp(empty, hue, f)!;
        final rect = Rect.fromLTWH(left + c * cell + 1, r * rowH + 1, cell - 2, rowH - 2);
        canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(2)), Paint()..color = color);
        if (selected == r * 24 + c) {
          canvas.drawRRect(
            RRect.fromRectAndRadius(rect.inflate(1), const Radius.circular(3)),
            Paint()
              ..color = ring
              ..style = PaintingStyle.stroke
              ..strokeWidth = 2,
          );
        }
      }
    }
    for (final h in [0, 6, 12, 18]) {
      _text(canvas, '$h', Offset(left + h * cell + cell / 2, rowH * 7 + 9));
    }
  }

  @override
  bool shouldRepaint(_HeatPainter old) => old.seconds != seconds || old.selected != selected || old.hue != hue || old.empty != empty;
}

/// A labelled horizontal bar for ranking (people, words): name, value text,
/// and a thin bar whose length is [fraction] of the row.
class RankBar extends StatelessWidget {
  const RankBar({super.key, required this.label, required this.value, required this.fraction, this.leading, this.onTap, this.color, this.below});

  final String label;
  final String value;
  final double fraction;
  final Widget? leading;
  final VoidCallback? onTap;
  final Color? color;

  /// Optional extra line under the bar (e.g. a [MoodStrip]).
  final Widget? below;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        child: Row(
          children: [
            if (leading != null) ...[leading!, const SizedBox(width: 12)],
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(label, maxLines: 1, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
                      ),
                      const SizedBox(width: 8),
                      // Shares the row with the name, so neither can push past the edge.
                      Flexible(
                        child: Text(value,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            textAlign: TextAlign.end,
                            style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  LayoutBuilder(
                    builder: (context, box) => Align(
                      alignment: Alignment.centerLeft,
                      child: Container(
                        width: math.max(4.0, box.maxWidth * fraction.clamp(0.0, 1.0)),
                        height: 8,
                        decoration: BoxDecoration(
                          color: color ?? t.colorScheme.primary,
                          borderRadius: const BorderRadius.horizontal(right: Radius.circular(4)),
                        ),
                      ),
                    ),
                  ),
                  if (below != null) ...[const SizedBox(height: 6), below!],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
