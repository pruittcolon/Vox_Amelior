/// The episode's index: chapters with their summaries and start times, to
/// skim and jump around in, like the chapters of a YouTube video.
library;

import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/render/plan.dart';
import 'theme.dart';

/// Estimated start time of every utterance at the speed you listen at.
PlanTimes sessionTimes(Session s) => s.plan.times(s.cast[null]?.speed ?? 1.0);

/// 1:05, 12:40, 1:02:03.
String formatClock(Duration d) {
  final s = d.inSeconds;
  final ss = (s % 60).toString().padLeft(2, '0');
  if (s >= 3600) return '${s ~/ 3600}:${(s ~/ 60 % 60).toString().padLeft(2, '0')}:$ss';
  return '${s ~/ 60}:$ss';
}

/// "physics" -> "Physics" (topics are written however Claude wrote them).
String topicLabel(String t) => t.isEmpty ? t : t[0].toUpperCase() + t.substring(1);

Duration chapterStart(Session s, PlanChapter c) => Duration(milliseconds: sessionTimes(s).starts[c.first]);

/// One row of the index: time, title, summary. The chapter playing is lit.
class ChapterTile extends StatelessWidget {
  const ChapterTile({
    super.key,
    required this.session,
    required this.chapter,
    required this.current,
    required this.onTap,
    this.summaryLines = 3,
  });

  final Session session;
  final PlanChapter chapter;
  final bool current;
  final VoidCallback onTap;
  final int summaryLines;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final text = theme.textTheme;
    final summary = chapter.summary;
    return Material(
      color: current ? scheme.primaryContainer.withValues(alpha: 0.55) : Colors.transparent,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(10, 10, 12, 10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 30,
                height: 30,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: current ? scheme.primary : scheme.surfaceContainerHighest,
                  shape: BoxShape.circle,
                ),
                child: current
                    ? Icon(Icons.graphic_eq_rounded, size: 16, color: scheme.onPrimary)
                    : Text('${chapter.index + 1}', style: text.labelMedium?.copyWith(color: scheme.onSurfaceVariant)),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      formatClock(chapterStart(session, chapter)),
                      style: text.labelSmall?.copyWith(color: scheme.secondary, letterSpacing: 0.6),
                    ),
                    const SizedBox(height: 2),
                    Text(chapter.title, style: text.titleSmall),
                    if (summary != null) ...[
                      const SizedBox(height: 3),
                      Text(
                        summary,
                        maxLines: summaryLines,
                        overflow: TextOverflow.ellipsis,
                        style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant, height: 1.4),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// "In this episode": the index at the top of the reader.
class ChapterCard extends StatelessWidget {
  const ChapterCard({super.key, required this.session, required this.current, required this.onJump, required this.onShowAll});

  final Session session;

  /// The utterance being read.
  final int current;
  final void Function(int utterance) onJump;
  final VoidCallback onShowAll;

  static const shown = 6;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final chapters = session.plan.chapters;
    final playing = session.plan.chapterOf(current)?.index;
    // Long books: a window around the chapter you're in.
    final from = chapters.length <= shown ? 0 : ((playing ?? 0) - 1).clamp(0, chapters.length - shown);
    final visible = chapters.skip(from).take(shown).toList();
    return Container(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: scheme.outlineVariant),
      ),
      padding: const EdgeInsets.fromLTRB(6, 12, 6, 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10),
            child: Row(
              children: [
                Icon(Icons.format_list_numbered_rounded, size: 18, color: scheme.secondary),
                const SizedBox(width: 8),
                Expanded(child: Text('In this episode · ${chapters.length} chapters', style: text.titleSmall)),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(10, 2, 10, 6),
            child: Text(
              'Tap a chapter to jump to it. Times are estimates; an export gives exact ones.',
              style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
            ),
          ),
          for (final c in visible)
            ChapterTile(
              session: session,
              chapter: c,
              current: c.index == playing,
              summaryLines: 2,
              onTap: () => onJump(c.first),
            ),
          if (chapters.length > shown)
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: onShowAll,
                icon: const Icon(Icons.unfold_more_rounded),
                label: Text('All ${chapters.length} chapters'),
              ),
            ),
        ],
      ),
    );
  }
}

/// The full index in a sheet. Returns the utterance to jump to, if any.
Future<int?> showChaptersSheet(BuildContext context, Session session, {required int current}) {
  final playing = session.plan.chapterOf(current)?.index ?? 0;
  return showModalBottomSheet<int>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    useSafeArea: true,
    builder: (sheet) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.7,
      maxChildSize: 0.95,
      builder: (context, controller) {
        final text = Theme.of(context).textTheme;
        final summary = session.script.description;
        return ListView(
          controller: controller,
          padding: const EdgeInsets.fromLTRB(14, 0, 14, 24),
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 6),
              child: Text('Chapters', style: text.titleLarge),
            ),
            if (summary != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(6, 6, 6, 4),
                child: Text(summary, style: text.bodyMedium?.copyWith(fontFamily: readingFont, height: 1.5)),
              ),
            const SizedBox(height: 8),
            for (final c in session.plan.chapters)
              ChapterTile(
                session: session,
                chapter: c,
                current: c.index == playing,
                summaryLines: 6,
                onTap: () => Navigator.pop(sheet, c.first),
              ),
          ],
        );
      },
    ),
  );
}

/// The seek bar, cut into chapters with small gaps (as on YouTube). Drag or
/// tap anywhere; [onPreview] names the spot while you drag.
class ChapterSeekBar extends StatefulWidget {
  const ChapterSeekBar({
    super.key,
    required this.session,
    required this.current,
    required this.onSeek,
    this.onPreview,
  });

  final Session session;
  final int current;
  final void Function(int utterance) onSeek;
  final void Function(int? utterance)? onPreview;

  @override
  State<ChapterSeekBar> createState() => _ChapterSeekBarState();
}

class _ChapterSeekBarState extends State<ChapterSeekBar> {
  double? _drag;

  PlanTimes get _times => sessionTimes(widget.session);

  int _utteranceAt(double fraction) => _times.utteranceAt((fraction.clamp(0.0, 1.0) * _times.total).round());

  void _move(double dx, double width) {
    final f = (dx / width).clamp(0.0, 1.0);
    setState(() => _drag = f);
    widget.onPreview?.call(_utteranceAt(f));
  }

  void _end() {
    final f = _drag;
    setState(() => _drag = null);
    widget.onPreview?.call(null);
    if (f != null) widget.onSeek(_utteranceAt(f));
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final times = _times;
    final total = times.total <= 0 ? 1 : times.total;
    final n = widget.session.plan.utterances.length;
    final cur = n == 0 ? 0 : widget.current.clamp(0, n - 1);
    final position = n == 0 ? 0.0 : times.starts[cur] / total;
    final bounds = [
      for (final c in widget.session.plan.chapters.skip(1)) times.starts[c.first] / total,
    ];
    return Semantics(
      slider: true,
      label: 'Position',
      value: formatClock(Duration(milliseconds: n == 0 ? 0 : times.starts[cur])),
      onIncrease: () => widget.onSeek((cur + 1).clamp(0, n - 1)),
      onDecrease: () => widget.onSeek((cur - 1).clamp(0, n - 1)),
      child: LayoutBuilder(
        builder: (context, box) {
          const pad = 12.0;
          final width = box.maxWidth - 2 * pad;
          return GestureDetector(
            behavior: HitTestBehavior.opaque,
            onHorizontalDragStart: (d) => _move(d.localPosition.dx - pad, width),
            onHorizontalDragUpdate: (d) => _move(d.localPosition.dx - pad, width),
            onHorizontalDragEnd: (_) => _end(),
            onTapDown: (d) => _move(d.localPosition.dx - pad, width),
            onTapUp: (_) => _end(),
            onTapCancel: () => setState(() => _drag = null),
            child: SizedBox(
              height: 30,
              width: double.infinity,
              child: CustomPaint(
                painter: _SeekPainter(
                  position: _drag ?? position,
                  bounds: bounds,
                  dragging: _drag != null,
                  track: scheme.surfaceContainerHighest,
                  fill: scheme.primary,
                  thumb: scheme.primary,
                  pad: pad,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _SeekPainter extends CustomPainter {
  _SeekPainter({
    required this.position,
    required this.bounds,
    required this.dragging,
    required this.track,
    required this.fill,
    required this.thumb,
    required this.pad,
  });

  final double position;
  final List<double> bounds;
  final bool dragging;
  final Color track;
  final Color fill;
  final Color thumb;
  final double pad;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width - 2 * pad;
    final y = size.height / 2;
    final h = dragging ? 6.0 : 4.0;
    const gap = 3.0;
    final edges = [0.0, ...bounds, 1.0];
    final x = pad + w * position;
    for (var i = 0; i + 1 < edges.length; i++) {
      final a = pad + w * edges[i] + (i == 0 ? 0 : gap / 2);
      final b = pad + w * edges[i + 1] - (i + 2 == edges.length ? 0 : gap / 2);
      if (b <= a) continue;
      final r = RRect.fromLTRBR(a, y - h / 2, b, y + h / 2, Radius.circular(h / 2));
      canvas.drawRRect(r, Paint()..color = track);
      if (x > a) {
        canvas.save();
        canvas.clipRect(Rect.fromLTRB(a, 0, x.clamp(a, b), size.height));
        canvas.drawRRect(r, Paint()..color = fill);
        canvas.restore();
      }
    }
    canvas.drawCircle(Offset(x, y), dragging ? 9 : 7, Paint()..color = thumb);
  }

  @override
  bool shouldRepaint(_SeekPainter old) =>
      old.position != position || old.dragging != dragging || old.bounds.length != bounds.length || old.fill != fill;
}
