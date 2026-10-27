import '../bridge/bridge_client.dart';
import 'g1_font.dart';
import 'glasses_text.dart';

/// Where things are on the G1 look-up screen in its "dual" layout, from the
/// firmware's DashBoard_Reflash (pixels on the 576 x 136 display): the date,
/// clock and weather on the left, a divider, and the right panel. The right
/// panel shows one note: an icon, a title line, and up to four body lines.
/// With a single note the "1/3" counter stays blank, and every note sent
/// makes the glasses show note 1 again.
class DashLayout {
  DashLayout._();

  static const width = 576, height = 136;
  static const dividerX = 172;
  static const iconX = 186, iconY = 1;
  static const titleX = 214, titleRight = 558, titleY = 1;
  static const bodyX = 186, bodyRight = 568, bodyY = 28;
  static const lineHeight = 27, bodyLines = 4;

  /// The glasses break lines letter by letter when text runs past the edge,
  /// so lines are broken here first, a little short of the edge (spacing
  /// between some letter pairs is not in the width table).
  static const titleMaxPx = titleRight - titleX - 8; // 336
  static const bodyMaxPx = bodyRight - bodyX - 10; // 372
}

/// Claude's panel on the right of the look-up screen: what Claude is doing
/// right now (bright), with what you asked and where (dim), or the result
/// of the last job.
class ClaudePanel {
  const ClaudePanel(this.title, this.bright, [this.dim = const []]);

  final String title;

  /// Lines at full brightness.
  final List<String> bright;

  /// Lines shown dimmer (after the firmware's dim marker).
  final List<String> dim;

  List<String> get lines => [...bright, ...dim];

  /// The note text as the glasses take it: one line per line, the dim
  /// marker in front of the first dim line.
  String get body => [
        ...bright,
        for (var i = 0; i < dim.length; i++) i == 0 ? '${G1Font.dimMark}${dim[i]}' : dim[i],
      ].join('\n');

  static String _clean(String s) => GlassesText.clean(s).replaceAll('\n', ' ').trim();

  static String _took(Duration d) {
    if (d.inSeconds < 60) return '${d.inSeconds.clamp(0, 59)}s';
    if (d.inMinutes < 60) return '${d.inMinutes} min';
    return '${d.inHours}h ${d.inMinutes % 60}m';
  }

  static String _clock(DateTime t, bool h24) {
    final m = t.minute.toString().padLeft(2, '0');
    if (h24) return '${t.hour.toString().padLeft(2, '0')}:$m';
    final h = t.hour % 12 == 0 ? 12 : t.hour % 12;
    return '$h:$m${t.hour < 12 ? 'am' : 'pm'}';
  }

  static ClaudePanel build({
    List<WorkJob> jobs = const [],
    (String, String)? asking,
    WorkItem? last,
    DateTime? now,
    bool h24 = false,
  }) {
    final t = now ?? DateTime.now();
    const w = DashLayout.bodyMaxPx;
    String title(String s) => G1Font.fit(s, DashLayout.titleMaxPx);

    if (jobs.isNotEmpty) {
      final j = jobs.first;
      final step = G1Font.fit(_clean(j.step.isEmpty ? 'Thinking' : j.step), w);
      final prompt = _clean(j.prompt);
      final meta = [
        if (j.project.isNotEmpty) j.project,
        if (j.steps > 0) '${j.steps} step${j.steps == 1 ? '' : 's'}',
        if (jobs.length > 1) '+${jobs.length - 1} more',
      ].join(' - ');
      return ClaudePanel(
        title('Claude - working ${_took(j.elapsed(t))}'),
        [step],
        [
          if (prompt.isNotEmpty) ...G1Font.wrapMax('"$prompt"', w, meta.isEmpty ? 3 : 2),
          if (meta.isNotEmpty) G1Font.fit(_clean(meta), w),
        ],
      );
    }
    if (asking != null) {
      final (step, question) = asking;
      return ClaudePanel(
        title('Claude - answering you'),
        [G1Font.fit(_clean(step.isEmpty ? 'Thinking' : step), w)],
        G1Font.wrapMax('"${_clean(question)}"', w, 3),
      );
    }
    if (last != null) {
      final meta = [
        if (last.project.isNotEmpty) last.project,
        if (last.ms > 0) _took(last.took),
        if (last.tools > 0) '${last.tools} step${last.tools == 1 ? '' : 's'}',
      ].join(' - ');
      final summary = G1Font.wrapMax(_clean(last.summary.isEmpty ? 'Finished.' : last.summary), w, 3);
      final prompt = _clean(last.prompt);
      final room = DashLayout.bodyLines - summary.length - (meta.isEmpty ? 0 : 1);
      return ClaudePanel(
        title('Claude - done ${_clock(last.time, h24)}'),
        summary,
        [
          if (prompt.isNotEmpty && room > 0) ...G1Font.wrapMax('You asked: $prompt', w, room),
          if (meta.isNotEmpty) G1Font.fit(meta, w),
        ],
      );
    }
    return const ClaudePanel('Claude', ['Nothing yet.'], ['Claude Code jobs show here', 'live while Claude works.']);
  }
}
