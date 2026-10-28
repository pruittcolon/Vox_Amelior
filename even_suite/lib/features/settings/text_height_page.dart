import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';

/// Move the text up or down in your view (and nearer or farther), like the
/// Even app's display height setting: the glasses show their own preview at
/// the new spot while you drag.
class TextHeightPage extends StatefulWidget {
  const TextHeightPage({super.key});

  @override
  State<TextHeightPage> createState() => _TextHeightPageState();
}

class _TextHeightPageState extends State<TextHeightPage> {
  late Suite _suite;
  bool _started = false;
  bool _saved = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_started) return;
    _started = true;
    _suite = SuiteScope.of(context);
    final s = _suite.settings;
    _suite.display.onGlassesExit(); // the preview takes the screen
    _suite.position.begin(s.displayHeight, s.displayDepth);
  }

  Future<void> _done() async {
    final p = _suite.position;
    await p.end();
    _saved = true;
    _suite.settings
      ..set('displayHeight', p.height)
      ..set('displayDepth', p.depth);
    if (_suite.ble.isConnected) {
      _suite.display.notice('Text shows here\nheight ${p.height + 1} · distance ${p.depth}',
          duration: const Duration(seconds: 4));
    }
  }

  @override
  void dispose() {
    // Left with the back button: keep the spot, close the preview.
    if (!_saved && _suite.position.open) {
      final p = _suite.position;
      p.end().then((_) {
        _suite.settings
          ..set('displayHeight', p.height)
          ..set('displayDepth', p.depth);
      });
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final p = _suite.position;
    return Scaffold(
      appBar: AppBar(title: const Text('Text height')),
      body: ListenableBuilder(
        listenable: Listenable.merge([p, _suite.ble]),
        builder: (context, _) {
          final connected = _suite.ble.isConnected;
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              Text(
                connected
                    ? 'Look at the glasses: they show where text will sit while you drag. Tap Done to keep it.'
                    : 'Connect your glasses first.',
                style: const TextStyle(color: AppColors.muted),
              ),
              const SizedBox(height: 16),
              _Preview(height: p.height, depth: p.depth),
              const SizedBox(height: 18),
              SectionCard(
                title: 'Height',
                icon: Icons.height_rounded,
                iconColor: AppColors.hud,
                trailing: Text('${p.height + 1} of 9', style: const TextStyle(fontWeight: FontWeight.w700)),
                child: Slider(
                  value: p.height.toDouble(),
                  min: 0,
                  max: 8,
                  divisions: 8,
                  label: '${p.height + 1}',
                  onChanged: connected ? (v) => p.set(h: v.round()) : null,
                ),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Distance',
                icon: Icons.straighten_rounded,
                iconColor: AppColors.glasses,
                trailing: Text('${p.depth} of 9', style: const TextStyle(fontWeight: FontWeight.w700)),
                child: Slider(
                  value: p.depth.toDouble(),
                  min: 1,
                  max: 9,
                  divisions: 8,
                  label: '${p.depth}',
                  onChanged: connected ? (v) => p.set(d: v.round()) : null,
                ),
              ),
              if (p.error != null) ...[
                const SizedBox(height: 10),
                Text(p.error!, style: const TextStyle(color: AppColors.warn)),
              ],
              const SizedBox(height: 18),
              FilledButton.icon(
                onPressed: connected
                    ? () async {
                        await _done();
                        if (context.mounted) Navigator.of(context).maybePop();
                      }
                    : null,
                icon: const Icon(Icons.check_rounded),
                label: const Text('Done'),
              ),
              const SizedBox(height: 10),
              const Text(
                  'The glasses keep this spot for everything they show, until you change it here or in the Even app. '
                  'They need about a second per move, so it follows your finger with a short delay.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ],
          );
        },
      ),
    );
  }
}

/// A sketch of your view: where the text block sits.
class _Preview extends StatelessWidget {
  const _Preview({required this.height, required this.depth});
  final int height;
  final int depth;

  @override
  Widget build(BuildContext context) => AspectRatio(
        aspectRatio: 16 / 9,
        child: Container(
          decoration: BoxDecoration(
            color: AppColors.hudBg,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.hud.withOpacity(0.2)),
          ),
          child: LayoutBuilder(builder: (context, c) {
            // Farther away looks a little smaller.
            final scale = 1.0 - (depth - 1) * 0.04;
            return AnimatedAlign(
              duration: const Duration(milliseconds: 250),
              curve: Curves.easeOut,
              alignment: Alignment(0, 0.9 - height / 8 * 1.8),
              child: AnimatedScale(
                duration: const Duration(milliseconds: 250),
                scale: scale,
                child: Container(
                  width: c.maxWidth * 0.7,
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    border: Border.all(color: AppColors.hud.withOpacity(0.5)),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Text('Your text here\nTermux-Claude listening...',
                      style: TextStyle(color: AppColors.hud, fontSize: 12, height: 1.4)),
                ),
              ),
            );
          }),
        ),
      );
}
