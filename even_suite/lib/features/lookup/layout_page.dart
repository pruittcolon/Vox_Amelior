import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/claude_panel.dart';
import '../../core/talk/conversation.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';
import '../../widgets/layout_editor.dart';

/// Arrange Claude's two glasses screens by dragging: the panel you see when
/// you look up, and the screen in front of you while Claude works. Each has
/// a live preview drawn the way the glasses draw it.
class LayoutPage extends StatefulWidget {
  const LayoutPage({super.key});

  @override
  State<LayoutPage> createState() => _LayoutPageState();
}

class _LayoutPageState extends State<LayoutPage> {
  String _sample = 'working';

  static const _panelLabels = {
    'asked': ('What you asked', 'Your question, dim'),
    'claude': ('Claude now', 'What it is writing, its steps, or the result'),
    'meta': ('Project line', 'Project · time · steps, dim'),
  };

  static const _screenLabels = {
    'meta': ('Status line', '“Claude · 12s ...”'),
    'asked': ('What you asked', 'Your question, dim'),
    'claude': ('Claude now', 'The step it is on, then the ones before'),
  };

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Glasses layout')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) {
          final panel = PanelLayout.parse(s.panelLayout);
          final screen = PanelLayout.parse(s.screenLayout, base: PanelLayout.working);
          final example = ClaudePanel.example(_sample, panel);
          final working = Conversation.workingRows(
            layout: screen,
            status: 'Claude · 12s ...',
            question: 'what should I pack for two days of hiking near Boston this weekend',
            step: 'Searching the web: trail weather',
            steps: const ['Reading packing_list.md', 'Searching the web: trail weather'],
          );
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              SectionCard(
                title: 'When you look up',
                icon: Icons.north_rounded,
                iconColor: AppColors.hud,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  if (s.dashRight != 'claude') ...[
                    const Text('The right side is not Claude right now.', style: TextStyle(color: AppColors.muted)),
                    const SizedBox(height: 8),
                    OutlinedButton(
                      onPressed: () => s.set('dashRight', 'claude'),
                      child: const Text('Put Claude on the right'),
                    ),
                    const SizedBox(height: 12),
                  ],
                  _Frame(
                    child: GlassesDashboardView(
                      now: DateTime.now(),
                      h24: s.dash24h,
                      temp: s.dashWeather ? (s.dashFahrenheit ? '68°F' : '20°C') : '',
                      title: example.title,
                      rows: example.rows,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(spacing: 6, runSpacing: 6, children: [
                    for (final (v, label) in const [('working', 'Working'), ('answering', 'Answering'), ('done', 'Done')])
                      ChoiceChip(
                        label: Text(label),
                        visualDensity: VisualDensity.compact,
                        selected: _sample == v,
                        onSelected: (_) => setState(() => _sample = v),
                      ),
                  ]),
                  const SizedBox(height: 12),
                  LayoutEditor(
                    layout: panel,
                    labels: _panelLabels,
                    onChanged: (l) => s.set('panelLayout', l.toString()),
                  ),
                  TextButton(
                    onPressed: () => s.set('panelLayout', PanelLayout.standard.toString()),
                    child: const Text('Back to the standard layout'),
                  ),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'In front of you while Claude works',
                icon: Icons.visibility_rounded,
                iconColor: AppColors.claude,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  _Frame(child: GlassesScreenView(rows: working)),
                  const SizedBox(height: 12),
                  LayoutEditor(
                    layout: screen,
                    labels: _screenLabels,
                    onChanged: (l) => s.set('screenLayout', l.toString()),
                  ),
                  TextButton(
                    onPressed: () => s.set('screenLayout', PanelLayout.working.toString()),
                    child: const Text('Back to the standard layout'),
                  ),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Close Claude’s screen after',
                icon: Icons.timer_outlined,
                iconColor: AppColors.warn,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  const Text(
                      'Counted from the last page (plus time to read it). It waits while you or the phone are talking.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  const SizedBox(height: 10),
                  Wrap(spacing: 8, runSpacing: 8, children: [
                    for (final (sec, label) in const [(5, '5 s'), (10, '10 s'), (15, '15 s'), (30, '30 s'), (60, '1 min'), (0, 'Never')])
                      ChoiceChip(
                        label: Text(label),
                        selected: s.closeAfterSec == sec,
                        onSelected: (_) => s.set('closeAfterSec', sec),
                      ),
                  ]),
                ]),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// A rounded frame with a margin, so its corners never cut the glasses' edges.
class _Frame extends StatelessWidget {
  const _Frame({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: AppColors.hudBg,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.hud.withOpacity(0.18)),
        ),
        child: child,
      );
}
