import 'dart:async';

import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/g1_font.dart';
import '../../core/glasses/look_layout.dart';
import '../../widgets/common.dart';

/// Your look-up screen, built by hand: drag anything (the time too) to where
/// you want it, pull the corner to make it wider or taller, tap it to change
/// how it looks. Above it, exactly what the glasses will draw.
class LookDesigner extends StatefulWidget {
  const LookDesigner({super.key});

  @override
  State<LookDesigner> createState() => _LookDesignerState();
}

class _LookDesignerState extends State<LookDesigner> {
  LookLayout? _draft; // while dragging (saved when you let go)
  int? _selected;
  Timer? _clock;

  @override
  void initState() {
    super.initState();
    // The preview's clock keeps time.
    _clock = Timer.periodic(const Duration(seconds: 20), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _clock?.cancel();
    super.dispose();
  }

  LookLayout _layout(BuildContext context) => _draft ?? SuiteScope.of(context).lookLayout;

  void _save(LookLayout layout) {
    _draft = null;
    SuiteScope.of(context).settings.set('lookLayout', layout.encode());
  }

  void _drag(LookLayout layout) => setState(() => _draft = layout);

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return ListenableBuilder(
      listenable: Listenable.merge([suite.settings, suite.dashboard]),
      builder: (context, _) {
        final layout = _layout(context);
        final data = suite.lookData();
        final sel = _selected != null && _selected! < layout.items.length ? _selected : null;
        return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          _HudFrame(
            child: LookScreen(layout: layout, data: data, highlight: sel),
          ),
          const SizedBox(height: 10),
          Row(children: [
            const Expanded(
              child: Text('Exactly what the glasses draw. It replaces their dashboard the moment you look up.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ),
            const SizedBox(width: 8),
            FilledButton.tonalIcon(
              onPressed: () {
                if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
                suite.showLookNow();
              },
              icon: const Icon(Icons.visibility_rounded, size: 18),
              label: const Text('Try it'),
            ),
          ]),
          const SizedBox(height: 18),
          _Heading('Arrange', 'Drag to move · pull the corner to resize · tap to edit'),
          const SizedBox(height: 10),
          LookCanvas(
            layout: layout,
            data: data,
            selected: sel,
            onSelect: (i) => setState(() => _selected = i),
            onDrag: _drag,
            onDrop: (l) => setState(() => _save(l)),
          ),
          if (sel != null) ...[
            const SizedBox(height: 12),
            _ItemPanel(
              key: ValueKey('panel-$sel'),
              item: layout.items[sel],
              onChanged: (item) => setState(() => _save(layout.withItem(sel, item))),
              onDelete: () => setState(() {
                _selected = null;
                _save(layout.remove(sel));
              }),
            ),
          ],
          const SizedBox(height: 18),
          _Heading('Add', 'Tap, or drag onto the screen'),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8, children: [
            for (final kind in LookKind.values)
              Draggable<LookKind>(
                data: kind,
                feedback: Material(color: Colors.transparent, child: _KindChip(kind, lifted: true)),
                childWhenDragging: Opacity(opacity: 0.4, child: _KindChip(kind)),
                child: _KindChip(kind, onTap: () {
                  final next = layout.add(layout.placeNew(kind));
                  setState(() {
                    _selected = next.items.length - 1;
                    _save(next);
                  });
                }),
              ),
          ]),
          const SizedBox(height: 18),
          _Heading('Start from', 'Pick one, then make it yours'),
          const SizedBox(height: 10),
          SizedBox(
            height: 92,
            child: ListView(scrollDirection: Axis.horizontal, children: [
              for (final e in LookLayout.presets.entries)
                Padding(
                  padding: const EdgeInsets.only(right: 10),
                  child: _PresetCard(
                    name: LookLayout.presetNames[e.key] ?? e.key,
                    layout: e.value,
                    data: data,
                    selected: e.value == layout,
                    onTap: () => setState(() {
                      _selected = null;
                      _save(e.value);
                    }),
                  ),
                ),
            ]),
          ),
        ]);
      },
    );
  }
}

class _Heading extends StatelessWidget {
  const _Heading(this.title, this.hint);
  final String title;
  final String hint;

  @override
  Widget build(BuildContext context) => Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
        Text(title, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(width: 10),
        Expanded(
          child: Text(hint,
              style: const TextStyle(color: AppColors.muted, fontSize: 12), maxLines: 1, overflow: TextOverflow.ellipsis),
        ),
      ]);
}

class _HudFrame extends StatelessWidget {
  const _HudFrame({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: AppColors.hudBg,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.hud.withOpacity(0.22)),
          boxShadow: [BoxShadow(color: AppColors.hud.withOpacity(0.08), blurRadius: 24, spreadRadius: 1)],
        ),
        child: child,
      );
}

/// The look-up screen as the glasses draw it: every piece where the spaces
/// put it, every letter at the G1 font's width.
class LookScreen extends StatelessWidget {
  const LookScreen({super.key, required this.layout, required this.data, this.highlight});

  final LookLayout layout;
  final LookData data;
  final int? highlight;

  @override
  Widget build(BuildContext context) => AspectRatio(
        aspectRatio: LookLayout.widthPx / 136,
        child: CustomPaint(painter: _LookPainter(LookRenderer.pieces(layout, data), highlight)),
      );
}

class _LookPainter extends CustomPainter {
  _LookPainter(this.pieces, this.highlight);
  final List<LookPiece> pieces;
  final int? highlight;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / LookLayout.widthPx);
    for (final p in pieces) {
      final color = highlight == null || highlight == p.item ? AppColors.hud : AppColors.hud.withOpacity(0.45);
      var x = p.x.toDouble();
      for (final rune in p.text.runes) {
        final adv = G1Font.advance(rune);
        if (adv > 0 && rune != 0x20) {
          final tp = TextPainter(
            text: TextSpan(
                text: String.fromCharCode(rune),
                style: TextStyle(fontFamily: 'Roboto', fontSize: 19, height: 1.0, color: color)),
            textDirection: TextDirection.ltr,
          )..layout();
          tp.paint(canvas, Offset(x, 3.0 + p.row * 27));
          tp.dispose();
        }
        x += adv;
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(_LookPainter old) => true;
}

/// The editor: the screen's 5 rows stretched tall enough for fingers, each
/// item a block you can move and resize. Things snap to the glasses' 6 px
/// grid and to rows.
class LookCanvas extends StatefulWidget {
  const LookCanvas({
    super.key,
    required this.layout,
    required this.data,
    required this.selected,
    required this.onSelect,
    required this.onDrag,
    required this.onDrop,
  });

  final LookLayout layout;
  final LookData data;
  final int? selected;
  final ValueChanged<int?> onSelect;
  final ValueChanged<LookLayout> onDrag;
  final ValueChanged<LookLayout> onDrop;

  static const rowHeight = 46.0;

  @override
  State<LookCanvas> createState() => _LookCanvasState();
}

class _LookCanvasState extends State<LookCanvas> {
  LookItem? _start;
  Offset _moved = Offset.zero;
  LookLayout? _last;
  final _key = GlobalKey();

  static int _shownWidth(LookItem item) => item.w > 0
      ? item.w
      : item.kind == LookKind.bigClock
          ? LookRenderer.bigClockWidth('12:59', item.styleOrDefault)
          : LookLayout.defaultWidth(item.kind);

  void _begin(int i) {
    _start = widget.layout.items[i];
    _moved = Offset.zero;
    widget.onSelect(i);
  }

  void _update(int i, DragUpdateDetails d, double scale, {required bool resize}) {
    final s = _start;
    if (s == null) return;
    _moved += d.delta;
    final dx = _moved.dx / scale;
    final dRows = (_moved.dy / LookCanvas.rowHeight).round();
    final next = resize
        ? s.copyWith(w: _shownWidth(s) + dx.round(), h: s.kind.tall ? s.h + dRows : s.h)
        : s.copyWith(x: s.x + dx.round(), row: s.row + dRows);
    _last = widget.layout.withItem(i, next);
    widget.onDrag(_last!);
  }

  void _end() {
    if (_last != null) widget.onDrop(_last!);
    _last = null;
    _start = null;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      final width = c.maxWidth;
      final scale = width / LookLayout.widthPx;
      const rh = LookCanvas.rowHeight;
      return DragTarget<LookKind>(
        onAcceptWithDetails: (d) {
          final box = _key.currentContext?.findRenderObject() as RenderBox?;
          if (box == null) return;
          final local = box.globalToLocal(d.offset);
          final kind = d.data;
          final item = LookItem(kind,
              row: (local.dy / rh).floor(),
              x: (local.dx / scale).round(),
              w: kind == LookKind.bigClock ? 0 : LookLayout.defaultWidth(kind),
              h: kind == LookKind.bigClock ? LookLayout.rows : kind.defaultRows);
          final next = widget.layout.add(item);
          widget.onSelect(next.items.length - 1);
          widget.onDrop(next);
        },
        builder: (context, candidates, _) => GestureDetector(
          onTap: () => widget.onSelect(null),
          child: Container(
            key: _key,
            height: rh * LookLayout.rows,
            decoration: BoxDecoration(
              color: AppColors.hudBg,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                  color: candidates.isNotEmpty ? AppColors.hud : AppColors.hud.withOpacity(0.18),
                  width: candidates.isNotEmpty ? 2 : 1),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(15),
              child: Stack(children: [
                // Row bands, so it is easy to see where things land.
                for (var r = 0; r < LookLayout.rows; r++)
                  Positioned(
                    left: 0,
                    right: 0,
                    top: r * rh,
                    height: rh,
                    child: Container(color: r.isOdd ? AppColors.hud.withOpacity(0.025) : Colors.transparent),
                  ),
                for (var i = 0; i < widget.layout.items.length; i++)
                  _block(i, widget.layout.items[i], scale, rh, width),
              ]),
            ),
          ),
        ),
      );
    });
  }

  Widget _block(int i, LookItem item, double scale, double rh, double width) {
    final selected = widget.selected == i;
    final maxW = (width - item.x * scale).clamp(24.0, width);
    final w = (_shownWidth(item) * scale).clamp(24.0, maxW);
    final lines = LookRenderer.content(item, widget.data);
    final label = lines.where((l) => l.trim().isNotEmpty).map((l) => l.trim()).take(item.h).join('\n');
    return Positioned(
      left: item.x * scale,
      top: item.row * rh,
      width: w,
      height: item.h * rh,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () => widget.onSelect(i),
        onPanStart: (_) => _begin(i),
        onPanUpdate: (d) => _update(i, d, scale, resize: false),
        onPanEnd: (_) => _end(),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 120),
          margin: const EdgeInsets.all(2),
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
          decoration: BoxDecoration(
            color: selected ? AppColors.hud.withOpacity(0.16) : AppColors.hud.withOpacity(0.07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: selected ? AppColors.hud : AppColors.hud.withOpacity(0.25), width: selected ? 1.6 : 1),
          ),
          child: Stack(clipBehavior: Clip.none, children: [
            Align(
              alignment: switch (item.align) {
                'r' => Alignment.centerRight,
                'c' => Alignment.center,
                _ => Alignment.centerLeft,
              },
              child: Text(
                item.kind == LookKind.bigClock ? 'BIG CLOCK' : (label.isEmpty ? item.kind.label : label),
                maxLines: item.h,
                overflow: TextOverflow.ellipsis,
                softWrap: item.h > 1,
                style: TextStyle(
                  color: AppColors.hud,
                  fontSize: item.kind == LookKind.bigClock ? 16 : 11.5,
                  fontWeight: item.kind == LookKind.bigClock ? FontWeight.w800 : FontWeight.w500,
                  letterSpacing: item.kind == LookKind.bigClock ? 2 : 0,
                  height: 1.25,
                ),
              ),
            ),
            if (selected && item.kind != LookKind.bigClock)
              Positioned(
                right: -8,
                bottom: -6,
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onPanStart: (_) => _begin(i),
                  onPanUpdate: (d) => _update(i, d, scale, resize: true),
                  onPanEnd: (_) => _end(),
                  child: Container(
                    width: 22,
                    height: 22,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(color: AppColors.hud, borderRadius: BorderRadius.circular(8)),
                    child: const Icon(Icons.open_in_full_rounded, size: 13, color: AppColors.hudBg),
                  ),
                ),
              ),
          ]),
        ),
      ),
    );
  }
}

IconData lookIcon(LookKind k) => switch (k) {
      LookKind.time => Icons.schedule_rounded,
      LookKind.bigClock => Icons.watch_later_rounded,
      LookKind.date => Icons.event_rounded,
      LookKind.weather => Icons.wb_sunny_rounded,
      LookKind.claude => Icons.auto_awesome_rounded,
      LookKind.work => Icons.task_alt_rounded,
      LookKind.phone => Icons.notifications_rounded,
      LookKind.battery => Icons.battery_std_rounded,
      LookKind.text => Icons.short_text_rounded,
      LookKind.line => Icons.horizontal_rule_rounded,
      LookKind.vline => Icons.more_vert_rounded,
    };

class _KindChip extends StatelessWidget {
  const _KindChip(this.kind, {this.onTap, this.lifted = false});
  final LookKind kind;
  final VoidCallback? onTap;
  final bool lifted;

  @override
  Widget build(BuildContext context) => Material(
        color: lifted ? AppColors.surfaceHigh : AppColors.surface,
        elevation: lifted ? 8 : 0,
        shape: StadiumBorder(side: BorderSide(color: lifted ? AppColors.hud : AppColors.outline)),
        child: InkWell(
          customBorder: const StadiumBorder(),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(lookIcon(kind), size: 16, color: AppColors.hud),
              const SizedBox(width: 6),
              Text(kind.label, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
            ]),
          ),
        ),
      );
}

/// Change the picked item: its look, where its text sits, its height, or remove it.
class _ItemPanel extends StatefulWidget {
  const _ItemPanel({super.key, required this.item, required this.onChanged, required this.onDelete});
  final LookItem item;
  final ValueChanged<LookItem> onChanged;
  final VoidCallback onDelete;

  @override
  State<_ItemPanel> createState() => _ItemPanelState();
}

class _ItemPanelState extends State<_ItemPanel> {
  late final _text = TextEditingController(text: widget.item.text);

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final item = widget.item;
    final styles = item.kind.styles;
    final alignable = item.kind != LookKind.bigClock && item.kind != LookKind.vline && item.kind != LookKind.line;
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 14),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Row(children: [
            IconBadge(lookIcon(item.kind), color: AppColors.hud, size: 34),
            const SizedBox(width: 10),
            Expanded(child: Text(item.kind.label, style: Theme.of(context).textTheme.titleMedium)),
            IconButton(
              tooltip: 'Full width',
              onPressed: item.kind == LookKind.bigClock
                  ? null
                  : () => widget.onChanged(item.copyWith(x: 0, w: LookLayout.maxPx)),
              icon: const Icon(Icons.settings_ethernet_rounded),
            ),
            IconButton(
              tooltip: 'Remove',
              onPressed: widget.onDelete,
              icon: const Icon(Icons.delete_outline_rounded, color: AppColors.error),
            ),
          ]),
          if (styles.length > 1) ...[
            const SizedBox(height: 8),
            Wrap(spacing: 6, runSpacing: 6, children: [
              for (final (v, label) in styles)
                ChoiceChip(
                  label: Text(label),
                  visualDensity: VisualDensity.compact,
                  selected: item.styleOrDefault == v,
                  onSelected: (_) => widget.onChanged(item.copyWith(style: v)),
                ),
            ]),
          ],
          if (alignable || item.kind.tall) ...[
            const SizedBox(height: 10),
            Row(children: [
              if (alignable)
                SegmentedButton<String>(
                  showSelectedIcon: false,
                  style: const ButtonStyle(visualDensity: VisualDensity.compact),
                  segments: const [
                    ButtonSegment(value: 'l', icon: Icon(Icons.format_align_left_rounded, size: 18)),
                    ButtonSegment(value: 'c', icon: Icon(Icons.format_align_center_rounded, size: 18)),
                    ButtonSegment(value: 'r', icon: Icon(Icons.format_align_right_rounded, size: 18)),
                  ],
                  selected: {item.align},
                  onSelectionChanged: (v) => widget.onChanged(item.copyWith(align: v.first)),
                ),
              const Spacer(),
              if (item.kind.tall) ...[
                IconButton(
                  tooltip: 'Fewer rows',
                  onPressed: item.h > 1 ? () => widget.onChanged(item.copyWith(h: item.h - 1)) : null,
                  icon: const Icon(Icons.remove_rounded),
                ),
                Text('${item.h} row${item.h == 1 ? '' : 's'}', style: const TextStyle(fontWeight: FontWeight.w600)),
                IconButton(
                  tooltip: 'More rows',
                  onPressed: item.bottom < LookLayout.rows ? () => widget.onChanged(item.copyWith(h: item.h + 1)) : null,
                  icon: const Icon(Icons.add_rounded),
                ),
              ],
            ]),
          ],
          if (item.kind == LookKind.text) ...[
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: TextField(
                controller: _text,
                decoration: const InputDecoration(labelText: 'Your text'),
                onChanged: (v) => widget.onChanged(item.copyWith(text: v)),
              ),
            ),
          ],
        ]),
      ),
    );
  }
}

class _PresetCard extends StatelessWidget {
  const _PresetCard({
    required this.name,
    required this.layout,
    required this.data,
    required this.selected,
    required this.onTap,
  });

  final String name;
  final LookLayout layout;
  final LookData data;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Container(
          width: 172,
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: AppColors.hudBg,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: selected ? AppColors.hud : AppColors.outline, width: selected ? 1.6 : 1),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            LookScreen(layout: layout, data: data),
            const Spacer(),
            Text(name,
                style: TextStyle(
                    color: selected ? AppColors.hud : AppColors.text, fontWeight: FontWeight.w700, fontSize: 13)),
          ]),
        ),
      );
}
