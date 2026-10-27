import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/voice/parakeet_model.dart';
import '../../widgets/common.dart';

/// Which recognizer turns your voice into text: Android's, or NVIDIA's
/// Parakeet running on the phone (one download).
class SpeechPage extends StatelessWidget {
  const SpeechPage({super.key});

  static String mb(int bytes) => '${(bytes / 1000000).round()} MB';

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    final model = suite.speechModel;
    final parakeet = suite.parakeet;
    return Scaffold(
      appBar: AppBar(title: const Text('Speech engine')),
      body: ListenableBuilder(
        listenable: Listenable.merge([s, model, if (parakeet != null) parakeet]),
        builder: (context, _) {
          final usingParakeet = s.speechEngine == 'parakeet';
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              const Text('What turns your voice into text - for the TouchBar, the talk button and hands-free.',
                  style: TextStyle(color: AppColors.muted)),
              const SizedBox(height: 16),
              _EngineCard(
                selected: !usingParakeet,
                icon: Icons.android_rounded,
                color: AppColors.ok,
                title: 'Android',
                badge: 'Built in',
                body: 'The phone’s own recognizer. Nothing to download.',
                onTap: () => s.set('speechEngine', 'android'),
                child: SegmentedButton<String>(
                  showSelectedIcon: false,
                  segments: const [
                    ButtonSegment(value: 'auto', label: Text('Auto')),
                    ButtonSegment(value: 'ondevice', label: Text('On-device')),
                    ButtonSegment(value: 'default', label: Text('Google')),
                  ],
                  selected: {s.recognizer},
                  onSelectionChanged: (v) => s.set('recognizer', v.first),
                ),
              ),
              const SizedBox(height: 12),
              _EngineCard(
                selected: usingParakeet,
                icon: Icons.bolt_rounded,
                color: AppColors.hud,
                title: 'Parakeet',
                badge: 'NVIDIA NeMo · on the phone',
                body: 'The same model as your Vox app: fast, accurate, with punctuation, and no internet '
                    'needed. Words show while you speak and it sends the moment you pause. It also hears '
                    'the glasses’ own microphone.',
                onTap: () {
                  s.set('speechEngine', 'parakeet');
                  if (!model.ready && model.state != ModelState.downloading) model.download();
                },
                child: _ParakeetState(model: model, status: parakeet?.status ?? 'off', active: usingParakeet),
              ),
              const SizedBox(height: 12),
              const Text(
                  'Parakeet needs a 64-bit phone and about 1 GB of memory while it is on. Switching back to '
                  'Android frees it.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ],
          );
        },
      ),
    );
  }
}

class _EngineCard extends StatelessWidget {
  const _EngineCard({
    required this.selected,
    required this.icon,
    required this.color,
    required this.title,
    required this.badge,
    required this.body,
    required this.onTap,
    required this.child,
  });

  final bool selected;
  final IconData icon;
  final Color color;
  final String title;
  final String badge;
  final String body;
  final VoidCallback onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) => Material(
        color: AppColors.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: BorderSide(color: selected ? color : AppColors.outline, width: selected ? 1.6 : 1),
        ),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: Ink(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [color.withOpacity(selected ? 0.16 : 0.05), Colors.transparent],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                Row(children: [
                  IconBadge(icon, color: color, size: 42),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
                      Text(badge, style: TextStyle(color: color, fontSize: 12.5, fontWeight: FontWeight.w600)),
                    ]),
                  ),
                  Icon(selected ? Icons.check_circle_rounded : Icons.radio_button_off_rounded,
                      color: selected ? color : AppColors.muted),
                ]),
                const SizedBox(height: 10),
                Text(body, style: const TextStyle(color: AppColors.muted, fontSize: 13.5)),
                const SizedBox(height: 14),
                child,
              ]),
            ),
          ),
        ),
      );
}

class _ParakeetState extends StatelessWidget {
  const _ParakeetState({required this.model, required this.status, required this.active});
  final SpeechModel model;
  final String status;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final total = SpeechPage.mb(model.totalBytes);
    switch (model.state) {
      case ModelState.downloading:
        return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(value: model.progress, minHeight: 8, color: AppColors.hud),
          ),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(
              child: Text('Downloading ${SpeechPage.mb(model.received)} of $total · ${(model.progress * 100).floor()}%',
                  style: const TextStyle(fontWeight: FontWeight.w600)),
            ),
            TextButton(onPressed: model.cancel, child: const Text('Pause')),
          ]),
          const Text('You can leave this page; it keeps going.', style: TextStyle(color: AppColors.muted, fontSize: 12)),
        ]);
      case ModelState.ready:
        final (label, color) = !active
            ? ('Downloaded · tap to use it', AppColors.muted)
            : switch (status) {
                'ready' => ('On · ready', AppColors.ok),
                'loading' => ('Loading the model…', AppColors.warn),
                'off' => ('On', AppColors.ok),
                _ => (status, AppColors.error),
              };
        return Row(children: [
          Expanded(child: StatusPill(label, color: color, pulse: status == 'loading' && active)),
          TextButton.icon(
            onPressed: () async {
              final ok = await showDialog<bool>(
                context: context,
                builder: (_) => AlertDialog(
                  title: const Text('Delete Parakeet?'),
                  content: Text('Frees $total. You can download it again any time.'),
                  actions: [
                    TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Keep')),
                    FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
                  ],
                ),
              );
              if (ok == true) {
                if (context.mounted) SuiteScope.of(context).settings.set('speechEngine', 'android');
                await model.delete();
              }
            },
            icon: const Icon(Icons.delete_outline_rounded, size: 18),
            label: const Text('Delete'),
          ),
        ]);
      case ModelState.failed:
      case ModelState.missing:
      case ModelState.unknown:
        final resume = model.received > 0;
        return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          if (model.error != null) ...[
            Text(model.error!, style: const TextStyle(color: AppColors.warn, fontSize: 12.5)),
            const SizedBox(height: 8),
          ],
          FilledButton.icon(
            style: FilledButton.styleFrom(backgroundColor: AppColors.hud, foregroundColor: AppColors.hudBg),
            onPressed: model.download,
            icon: const Icon(Icons.download_rounded),
            label: Text(resume
                ? 'Resume download (${SpeechPage.mb(model.totalBytes - model.received)} left)'
                : 'Download Parakeet ($total)'),
          ),
          const SizedBox(height: 6),
          const Text('Best on Wi-Fi. It switches over by itself when the download is done.',
              textAlign: TextAlign.center, style: TextStyle(color: AppColors.muted, fontSize: 12)),
        ]);
    }
  }
}
