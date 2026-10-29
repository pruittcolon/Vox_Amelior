import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/settings.dart';
import '../../widgets/common.dart';

/// How long text stays on the glasses: page turns (reading speed, a fixed
/// time, or by hand), the extra pauses, how long an answer stays, and the
/// "*" when Claude is done. "Try it" shows a sample with these timings.
class TextTimingPage extends StatelessWidget {
  const TextTimingPage({super.key});

  static const sample = 'This is how answers move on your glasses. Each screen stays up for the time you set here, '
      'then the next one comes by itself. A right tap goes on at once and a left tap goes back and pauses.\n\n'
      'This is the second screen. If it moves too fast, slow it down or add a pause per page. If you would '
      'rather turn pages yourself, pick By hand and tap the right side to go on.\n\n'
      'Last screen. After it the answer stays for a while, then the glasses go back to their own screen.';

  static String _mode(Settings s) => s.readingWpm == 0 ? 'hand' : s.pageSec > 0 ? 'fixed' : 'read';

  static String _secs(int ms) => '${(ms / 1000).toStringAsFixed(ms % 1000 == 0 ? 0 : 1)} s';

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Text timing')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) {
          final mode = _mode(s);
          // A full screen holds about 40 words.
          final fullMs = mode == 'fixed'
              ? s.pageSec * 1000 + s.pageHoldMs
              : (40 * 60000 / (s.readingWpm == 0 ? 180 : s.readingWpm)).round() + 600 + s.pageHoldMs;
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              const Text('How long each screen of an answer stays on your glasses.',
                  style: TextStyle(color: AppColors.muted)),
              const SizedBox(height: 14),
              SectionCard(
                title: 'Pages turn',
                icon: Icons.auto_stories_rounded,
                iconColor: AppColors.hud,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  SegmentedButton<String>(
                    showSelectedIcon: false,
                    segments: const [
                      ButtonSegment(value: 'read', label: Text('Reading speed')),
                      ButtonSegment(value: 'fixed', label: Text('Fixed time')),
                      ButtonSegment(value: 'hand', label: Text('By hand')),
                    ],
                    selected: {mode},
                    onSelectionChanged: (v) {
                      switch (v.first) {
                        case 'hand':
                          s.set('readingWpm', 0);
                        case 'fixed':
                          if (s.readingWpm == 0) s.set('readingWpm', 180);
                          if (s.pageSec == 0) s.set('pageSec', 8);
                        default:
                          if (s.readingWpm == 0) s.set('readingWpm', 180);
                          s.set('pageSec', 0);
                      }
                    },
                  ),
                  const SizedBox(height: 10),
                  Text(
                    switch (mode) {
                      'hand' => 'Pages stay until you tap: right = next, left = back.',
                      'fixed' => 'Every screen stays the same time, however much is on it.',
                      _ => 'Longer screens stay longer, like reading at your speed.',
                    },
                    style: const TextStyle(color: AppColors.muted, fontSize: 13),
                  ),
                  if (mode == 'read') ...[
                    _Label('Speed: ${s.readingWpm} words a minute'),
                    Slider(
                      value: s.readingWpm.clamp(60, 400).toDouble(),
                      min: 60,
                      max: 400,
                      divisions: 34,
                      label: '${s.readingWpm}',
                      onChanged: (v) => s.set('readingWpm', v.round()),
                    ),
                  ],
                  if (mode == 'fixed') ...[
                    _Label('Each screen: ${s.pageSec} s'),
                    Slider(
                      value: s.pageSec.clamp(2, 30).toDouble(),
                      min: 2,
                      max: 30,
                      divisions: 28,
                      label: '${s.pageSec} s',
                      onChanged: (v) => s.set('pageSec', v.round()),
                    ),
                  ],
                  if (mode != 'hand') ...[
                    _Label('Extra on every screen: ${_secs(s.pageHoldMs)}'),
                    Slider(
                      value: s.pageHoldMs.clamp(0, 6000).toDouble(),
                      min: 0,
                      max: 6000,
                      divisions: 24,
                      label: _secs(s.pageHoldMs),
                      onChanged: (v) => s.set('pageHoldMs', v.round()),
                    ),
                    _Label('Extra on the first screen: ${_secs(s.firstPageMs)}'),
                    Slider(
                      value: s.firstPageMs.clamp(0, 8000).toDouble(),
                      min: 0,
                      max: 8000,
                      divisions: 16,
                      label: _secs(s.firstPageMs),
                      onChanged: (v) => s.set('firstPageMs', v.round()),
                    ),
                    Text('A full screen stays about ${_secs((fullMs / 500).round() * 500)}.',
                        style: const TextStyle(color: AppColors.glasses, fontWeight: FontWeight.w600)),
                  ],
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'After the last screen',
                icon: Icons.timer_rounded,
                iconColor: AppColors.claude,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  const Text('The answer stays this long, then the glasses go back to their own screen.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  const SizedBox(height: 10),
                  Wrap(spacing: 8, runSpacing: 8, children: [
                    for (final (sec, label) in const [(5, '5 s'), (10, '10 s'), (15, '15 s'), (30, '30 s'), (60, '1 min'), (0, 'Until I close it')])
                      ChoiceChip(
                        label: Text(label),
                        selected: s.closeAfterSec == sec,
                        onSelected: (_) => s.set('closeAfterSec', sec),
                      ),
                  ]),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Termux on the glasses',
                icon: Icons.terminal_rounded,
                iconColor: AppColors.glasses,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  const Text(
                      'After you send to a Termux session, its screen stays on the glasses this long after it last '
                      'changed, so you can read it and send the next. Double tap or say "hide" to close it sooner.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  const SizedBox(height: 10),
                  Wrap(spacing: 8, runSpacing: 8, children: [
                    for (final (sec, label) in const [
                      (10, '10 s'),
                      (30, '30 s'),
                      (60, '1 min'),
                      (120, '2 min'),
                      (300, '5 min'),
                      (-1, 'Until I close it'),
                      (0, 'Off: answer like chat'),
                    ])
                      ChoiceChip(
                        label: Text(label),
                        selected: s.termShowSec == sec,
                        onSelected: (_) => s.set('termShowSec', sec),
                      ),
                  ]),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'The * when Claude is done',
                icon: Icons.star_rounded,
                iconColor: AppColors.warn,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Show it', style: TextStyle(fontWeight: FontWeight.w600)),
                    value: s.doneBlink,
                    onChanged: (v) => s.set('doneBlink', v),
                  ),
                  if (s.doneBlink) ...[
                    _Label('Shows for ${_secs(s.doneBlinkMs)}'),
                    Slider(
                      value: s.doneBlinkMs.clamp(500, 10000).toDouble(),
                      min: 500,
                      max: 10000,
                      divisions: 19,
                      label: _secs(s.doneBlinkMs),
                      onChanged: (v) => s.set('doneBlinkMs', v.round()),
                    ),
                  ],
                ]),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: () {
                  if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
                  suite.display.showText(sample, title: 'Timing test');
                  showToast(context, 'Look at your glasses');
                },
                icon: const Icon(Icons.play_arrow_rounded),
                label: const Text('Try it on the glasses'),
              ),
              const SizedBox(height: 8),
              const Text('On the glasses: say “faster” or “slower”, or tap right / left.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ],
          );
        },
      ),
    );
  }
}

class _Label extends StatelessWidget {
  const _Label(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 12),
        child: Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
      );
}
