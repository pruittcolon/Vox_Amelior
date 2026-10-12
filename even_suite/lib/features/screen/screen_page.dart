import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/screen/screen_mirror.dart';
import '../../widgets/common.dart';
import '../../widgets/voice_widgets.dart';

/// Phone screen on the glasses, about one frame a second.
class ScreenPage extends StatelessWidget {
  const ScreenPage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final m = suite.mirror;
    final s = suite.settings;
    return ListenableBuilder(
      listenable: Listenable.merge([m, s, suite.ble]),
      builder: (context, _) {
        final (label, color) = m.running
            ? ('Sharing · ${m.fps.toStringAsFixed(1)} fps', AppColors.ok)
            : m.starting
                ? ('Asking Android…', AppColors.warn)
                : ('Off', AppColors.muted);
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            Row(children: [
              Expanded(
                child: Text('Screen',
                    style: Theme.of(context).textTheme.headlineSmall, maxLines: 1, overflow: TextOverflow.ellipsis),
              ),
              StatusPill(label, color: color, pulse: m.running || m.starting),
            ]),
            const SizedBox(height: 4),
            const Text('Your phone screen on the glasses, about once a second.',
                style: TextStyle(color: AppColors.muted)),
            const SizedBox(height: 14),
            _Preview(mirror: m),
            const SizedBox(height: 10),
            if (m.running || m.sent > 0)
              Text(
                '${m.sent} sent · ${m.skipped} unchanged${m.failed > 0 ? ' · ${m.failed} missed' : ''}'
                '${m.lastSendMs > 0 ? ' · ${m.lastSendMs} ms per frame' : ''}',
                style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
              ),
            if (m.error != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(m.error!, style: const TextStyle(color: AppColors.warn)),
              ),
            const SizedBox(height: 14),
            if (!m.running)
              FilledButton.icon(
                onPressed: m.starting
                    ? null
                    : () {
                        if (!suite.ble.isConnected) showToast(context, 'Glasses are not connected yet; frames wait for them');
                        m.start();
                      },
                icon: const Icon(Icons.cast_rounded),
                label: const Text('Show my screen on the glasses'),
              )
            else
              Row(children: [
                Expanded(
                  child: FilledButton.icon(
                    style: FilledButton.styleFrom(backgroundColor: AppColors.claude),
                    onPressed: m.stop,
                    icon: const Icon(Icons.stop_rounded),
                    label: const Text('Stop sharing'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: m.extend,
                  child: Text(m.offAt == null ? '+${s.autoOffMinutes} min' : 'Until ${clockTime(m.offAt!)}'),
                ),
              ]),
            const SizedBox(height: 6),
            Text(
              'Android asks each time; pick “Entire screen”. It turns itself off after ${s.autoOffMinutes} min, '
              'and the notification has a Stop sharing button.',
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            const SizedBox(height: 18),
            SectionCard(
              title: 'What to show',
              icon: Icons.crop_rounded,
              iconColor: AppColors.glasses,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (value, text) in const [('strip', 'Band'), ('top', 'Top half'), ('full', 'Whole screen')])
                    ChoiceChip(
                      label: Text(text),
                      selected: s.mirrorRegion == value,
                      onSelected: (_) => s.set('mirrorRegion', value),
                    ),
                ]),
                const SizedBox(height: 8),
                Text(
                  switch (s.mirrorRegion) {
                    'strip' => 'A full-width band in the glasses’ shape: the most readable.',
                    'top' => 'The top half of the screen, shrunk to fit.',
                    _ => 'The whole screen, shrunk to fit (small).',
                  },
                  style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                ),
                if (s.mirrorRegion == 'strip') ...[
                  const SizedBox(height: 8),
                  Row(children: [
                    const Text('Top', style: TextStyle(color: AppColors.muted)),
                    Expanded(
                      child: Slider(
                        value: s.mirrorStrip.toDouble(),
                        min: 0,
                        max: 100,
                        divisions: 20,
                        label: '${s.mirrorStrip}%',
                        onChanged: (v) => s.set('mirrorStrip', v.round()),
                      ),
                    ),
                    const Text('Bottom', style: TextStyle(color: AppColors.muted)),
                  ]),
                ],
              ]),
            ),
            const SizedBox(height: 12),
            SectionCard(
              title: 'Look and speed',
              icon: Icons.tune_rounded,
              iconColor: AppColors.hud,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                const Text('Frames per second', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (tenths, text) in const [(5, '0.5'), (10, '1'), (20, '2')])
                    ChoiceChip(
                      label: Text(text),
                      selected: s.mirrorFpsTenths == tenths,
                      onSelected: (_) => s.set('mirrorFpsTenths', tenths),
                    ),
                ]),
                const SizedBox(height: 14),
                const Text('Style', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                SegmentedButton<String>(
                  showSelectedIcon: false,
                  segments: const [
                    ButtonSegment(value: 'text', label: Text('Sharp text')),
                    ButtonSegment(value: 'photo', label: Text('Photo')),
                  ],
                  selected: {s.mirrorStyle},
                  onSelectionChanged: (v) => s.set('mirrorStyle', v.first),
                ),
                const SizedBox(height: 14),
                const Text('Dark background', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (value, text) in const [('auto', 'Auto'), ('on', 'Always invert'), ('off', 'As is')])
                    ChoiceChip(
                      label: Text(text),
                      selected: s.mirrorInvert == value,
                      onSelected: (_) => s.set('mirrorInvert', value),
                    ),
                ]),
                const SizedBox(height: 6),
                const Text('Auto turns light screens dark so the glasses do not glow white.',
                    style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
              ]),
            ),
          ],
        );
      },
    );
  }
}

class _Preview extends StatelessWidget {
  const _Preview({required this.mirror});
  final ScreenMirror mirror;

  @override
  Widget build(BuildContext context) {
    final bmp = mirror.preview;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.hudBg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.hud.withOpacity(0.18)),
      ),
      child: AspectRatio(
        aspectRatio: 576 / 136,
        child: bmp == null
            ? Center(
                child: Text(mirror.running ? 'Waiting for the first frame…' : 'What the glasses will show',
                    style: TextStyle(color: AppColors.hud.withOpacity(0.45))),
              )
            : ColorFiltered(
                // White = lit; drawn in the glasses' green.
                colorFilter: const ColorFilter.matrix([
                  0, 0, 0, 0, 0x6C, //
                  0, 0, 0, 0, 0xF5, //
                  0, 0, 0, 0, 0xA0, //
                  0.333, 0.333, 0.333, 0, 0, //
                ]),
                child: Image.memory(bmp,
                    gaplessPlayback: true, filterQuality: FilterQuality.none, fit: BoxFit.contain),
              ),
      ),
    );
  }
}
