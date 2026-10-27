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

/// One line of the panel; dim lines are drawn darker by the glasses.
class PanelLine {
  const PanelLine(this.text, {this.dim = false});
  final String text;
  final bool dim;

  @override
  bool operator ==(Object other) => other is PanelLine && other.text == text && other.dim == dim;

  @override
  int get hashCode => Object.hash(text, dim);

  @override
  String toString() => dim ? '(dim) $text' : text;
}

/// Claude's panel on the right of the look-up screen: one short line of
/// what you asked (dim; [askLines] sets how many), and the rest is Claude -
/// what it is writing right now, the step it is on, or the result.
class ClaudePanel {
  const ClaudePanel(this.title, this.rows);

  final String title;
  final List<PanelLine> rows;

  List<String> get lines => [for (final r in rows) r.text];

  /// The note text as the glasses take it: one line per row, the firmware's
  /// dim marker wherever the brightness changes.
  String get body {
    var dim = false;
    final out = <String>[];
    for (final r in rows) {
      out.add(r.dim != dim ? '${G1Font.dimMark}${r.text}' : r.text);
      dim = r.dim;
    }
    return out.join('\n');
  }

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

  /// What you asked, in at most [n] dim lines.
  static List<PanelLine> _asked(String prompt, int n) {
    final p = _clean(prompt);
    if (n <= 0 || p.isEmpty) return const [];
    return [for (final l in G1Font.wrapMax('"$p"', DashLayout.bodyMaxPx, n)) PanelLine(l, dim: true)];
  }

  static List<PanelLine> _bright(Iterable<String> lines) => [for (final l in lines) PanelLine(l)];

  /// [asking]: a question from this app (step, question, the answer so far).
  /// [layout]: the order of the blocks and how many lines your question and
  /// the project line take; Claude gets the rest of the four lines.
  static ClaudePanel build({
    List<WorkJob> jobs = const [],
    (String, String, String)? asking,
    WorkItem? last,
    DateTime? now,
    bool h24 = false,
    PanelLayout layout = PanelLayout.standard,
  }) {
    final t = now ?? DateTime.now();
    const w = DashLayout.bodyMaxPx;

    String title;
    var prompt = '';
    var meta = '';
    // Claude's lines for a given room, best first.
    List<PanelLine> Function(int room) claude;

    if (jobs.isNotEmpty) {
      final j = jobs.first;
      title = 'Claude - working ${_took(j.elapsed(t))}';
      prompt = j.prompt;
      meta = [
        if (j.project.isNotEmpty) j.project,
        if (j.steps > 0) '${j.steps} step${j.steps == 1 ? '' : 's'}',
        if (jobs.length > 1) '+${jobs.length - 1} more',
      ].join(' - ');
      // The step it is on, then the ones before it (newest first).
      final steps = [
        PanelLine(G1Font.fit(_clean(j.step.isEmpty ? 'Thinking' : j.step), w)),
        for (final s in j.recent.reversed.skip(1).where((s) => s.isNotEmpty))
          PanelLine(G1Font.fit(_clean(s), w), dim: true),
      ];
      claude = (room) => steps.take(room).toList();
    } else if (asking != null) {
      final (step, question, live) = asking;
      title = 'Claude - answering you';
      prompt = question;
      final text = _clean(live);
      // The newest lines of what Claude is writing; before it writes, the step.
      claude = (room) {
        if (text.isEmpty) return [PanelLine(G1Font.fit(_clean(step.isEmpty ? 'Thinking' : step), w))];
        final said = G1Font.wrap(text, w);
        return _bright(said.length <= room ? said : said.sublist(said.length - room));
      };
    } else if (last != null && last.isMessage) {
      title = 'Claude Code - ${_clock(last.time, h24)}';
      meta = last.project;
      claude = (room) => _bright(G1Font.wrapMax(_clean(last.text.isEmpty ? last.summary : last.text), w, room));
    } else if (last != null) {
      title = 'Claude - done ${_clock(last.time, h24)}';
      prompt = last.prompt;
      meta = [
        if (last.project.isNotEmpty) last.project,
        if (last.ms > 0) _took(last.took),
        if (last.tools > 0) '${last.tools} step${last.tools == 1 ? '' : 's'}',
      ].join(' - ');
      claude = (room) => _bright(G1Font.wrapMax(_clean(last.summary.isEmpty ? 'Finished.' : last.summary), w, room));
    } else {
      return const ClaudePanel('Claude', [
        PanelLine('Nothing yet.'),
        PanelLine('Claude Code jobs show here', dim: true),
        PanelLine('live while Claude works.', dim: true),
      ]);
    }

    final asked = _asked(prompt, layout.askLines);
    final metaRows = layout.metaLines > 0 && meta.isNotEmpty ? [PanelLine(G1Font.fit(_clean(meta), w), dim: true)] : const <PanelLine>[];
    final room = (DashLayout.bodyLines - asked.length - metaRows.length).clamp(1, DashLayout.bodyLines);
    final blocks = {'asked': asked, 'claude': claude(room), 'meta': metaRows};
    return ClaudePanel(G1Font.fit(title, DashLayout.titleMaxPx), [for (final b in layout.order) ...?blocks[b]]);
  }

  /// Example panels for the layout editor's preview.
  static ClaudePanel example(String state, PanelLayout layout, {DateTime? now}) {
    final t = now ?? DateTime.now();
    double ago(int s) => t.subtract(Duration(seconds: s)).millisecondsSinceEpoch / 1000;
    switch (state) {
      case 'answering':
        return build(
            now: t,
            layout: layout,
            asking: (
              'Writing',
              'what should I pack for two days of hiking near Boston this weekend',
              'Pack layers: a warm fleece, a rain shell and two pairs of wool socks. Bring a headlamp, '
                  'water filter and snacks with salt.',
            ));
      case 'done':
        return build(
            now: t,
            layout: layout,
            last: WorkItem(
                id: 'x',
                ts: ago(120),
                project: 'even-suite',
                summary: 'Pushed the fix; CI is green and the APK is published to the release page.',
                prompt: 'fix the failing screenshot test and ship it',
                ms: 134000,
                tools: 6));
      default:
        return build(now: t, layout: layout, jobs: [
          WorkJob(
              session: 'x',
              project: 'vox_mobile',
              prompt: 'add mood charts to the Insights tab and make them work in dark mode',
              step: 'Running flutter test',
              steps: 14,
              recent: const ['Reading insights_page.dart', 'Editing mood_chart.dart', 'Running flutter test'],
              started: ago(250)),
        ]);
    }
  }
}

/// How Claude's panel is laid out: the order of its three blocks (what you
/// asked, Claude, the project line) and how many lines the first and last
/// take. Claude always gets the rest of the four lines. Saved as e.g.
/// "asked:1,claude,meta:0".
class PanelLayout {
  const PanelLayout(this.order, {this.askLines = 1, this.metaLines = 0});

  final List<String> order;
  final int askLines; // 0-3
  final int metaLines; // 0-1

  static const blocks = ['asked', 'claude', 'meta'];
  static const standard = PanelLayout(['asked', 'claude', 'meta']);

  /// Claude's share of the four lines.
  int get claudeLines => DashLayout.bodyLines - askLines - metaLines;

  PanelLayout copyWith({List<String>? order, int? askLines, int? metaLines}) {
    var ask = (askLines ?? this.askLines).clamp(0, 3);
    var meta = (metaLines ?? this.metaLines).clamp(0, 1);
    // Claude keeps at least one line.
    if (ask + meta > DashLayout.bodyLines - 1) {
      if (askLines != null) {
        meta = (DashLayout.bodyLines - 1 - ask).clamp(0, 1);
      } else {
        ask = DashLayout.bodyLines - 1 - meta;
      }
    }
    return PanelLayout(order ?? this.order, askLines: ask, metaLines: meta);
  }

  static PanelLayout parse(String s) {
    final order = <String>[];
    var ask = standard.askLines, meta = standard.metaLines;
    for (final part in s.split(',')) {
      final kv = part.trim().split(':');
      final name = kv.first;
      if (!blocks.contains(name) || order.contains(name)) continue;
      order.add(name);
      final n = kv.length > 1 ? int.tryParse(kv[1]) : null;
      if (name == 'asked' && n != null) ask = n;
      if (name == 'meta' && n != null) meta = n;
    }
    for (final b in blocks) {
      if (!order.contains(b)) order.add(b);
    }
    return const PanelLayout(['asked', 'claude', 'meta']).copyWith(order: order, askLines: ask, metaLines: meta);
  }

  @override
  String toString() => order.map((b) => b == 'asked' ? 'asked:$askLines' : b == 'meta' ? 'meta:$metaLines' : b).join(',');
}
