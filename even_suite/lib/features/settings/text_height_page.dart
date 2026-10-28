import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/height_send.dart';
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
  late final HeightSend _send;
  late final _topText = TextEditingController();
  late final _bottomText = TextEditingController();
  bool _started = false;
  bool _saved = false;
  bool _tested = false; // the test moved the glasses away from your spot
  int _homeH = 4, _homeD = 5; // your spot, put back after the test

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_started) return;
    _started = true;
    _suite = SuiteScope.of(context);
    _send = HeightSend(_suite.proto, _suite.position);
    _topText.text = _send.top.text;
    _bottomText.text = _send.bottom.text;
    _homeH = _suite.settings.displayHeight;
    _homeD = _suite.settings.displayDepth;
    // After this frame: it updates other screens, which can't happen mid-build.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final s = _suite.settings;
      _suite.display.onGlassesExit(); // the preview takes the screen
      _suite.position.begin(s.displayHeight, s.displayDepth);
    });
  }

  /// Stop the test and put the text back at your spot (preview open again).
  Future<void> _backToMine() async {
    await _send.stop();
    while (_send.busy) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
    }
    if (!_tested) return;
    _tested = false;
    await _suite.position.begin(_homeH, _homeD);
  }

  Future<void> _test(Future<void> Function() run) async {
    _tested = true;
    await run();
  }

  Future<void> _done() async {
    await _backToMine();
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
    // Left with the back button: keep your spot, close the preview.
    if (!_saved && (_suite.position.open || _tested || _send.running)) {
      final p = _suite.position;
      // Once the page is gone (other screens update when it closes).
      Future<void>(() async {
        await _backToMine();
        await p.end();
        _suite.settings
          ..set('displayHeight', p.height)
          ..set('displayDepth', p.depth);
        _send.dispose();
      });
    } else {
      _send.dispose();
    }
    _topText.dispose();
    _bottomText.dispose();
    super.dispose();
  }

  Widget _spotCard(String title, IconData icon, HeightSpot spot, TextEditingController text, bool connected) {
    final off = !connected;
    return SectionCard(
      title: title,
      icon: icon,
      iconColor: AppColors.hud,
      trailing: Text('height ${spot.height + 1} of 9', style: const TextStyle(fontWeight: FontWeight.w700)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        TextField(
          controller: text,
          minLines: 1,
          maxLines: 3,
          decoration: const InputDecoration(labelText: 'Text'),
          onChanged: (v) => spot.text = v,
        ),
        const SizedBox(height: 8),
        Row(children: [
          const SizedBox(width: 62, child: Text('Height')),
          Expanded(
            child: Slider(
              value: spot.height.toDouble(),
              min: 0,
              max: 8,
              divisions: 8,
              label: '${spot.height + 1}',
              onChanged: off ? null : (v) => setState(() => spot.height = v.round()),
            ),
          ),
        ]),
        Row(children: [
          const SizedBox(width: 62, child: Text('Stays')),
          Expanded(
            child: Slider(
              value: spot.holdMs.toDouble(),
              min: 0,
              max: 5000,
              divisions: 50,
              label: '${(spot.holdMs / 1000).toStringAsFixed(1)} s',
              onChanged: (v) => setState(() => spot.holdMs = v.round()),
            ),
          ),
          SizedBox(width: 44, child: Text('${(spot.holdMs / 1000).toStringAsFixed(1)} s')),
        ]),
        Align(
          alignment: Alignment.centerLeft,
          child: OutlinedButton(
            onPressed: off || _send.busy || _send.running ? null : () => _test(() => _send.sendOnce(spot)),
            child: const Text('Send this one now'),
          ),
        ),
      ]),
    );
  }

  Widget _testCard(bool connected) {
    final s = _send;
    return SectionCard(
      title: 'Switch between them',
      icon: Icons.swap_vert_rounded,
      iconColor: AppColors.glasses,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Text(
          s.steps == 0
              ? 'Shows the top text, then the bottom text, again and again.'
              : '${s.steps} sent · last move ${s.lastMoveMs} ms, text ${s.lastTextMs} ms'
                  '${s.cycleMs > 0 ? ' · a full turn ${s.cycleMs} ms' : ''}',
          style: const TextStyle(color: AppColors.muted, fontSize: 13),
        ),
        if (s.error != null) ...[
          const SizedBox(height: 6),
          Text(s.error!, style: const TextStyle(color: AppColors.warn, fontSize: 13)),
        ],
        const SizedBox(height: 10),
        Row(children: [
          Expanded(
            child: s.running
                ? FilledButton.icon(
                    onPressed: _backToMine,
                    icon: const Icon(Icons.stop_rounded),
                    label: const Text('Stop'),
                  )
                : FilledButton.icon(
                    onPressed: connected && !s.busy ? () => _test(s.start) : null,
                    icon: const Icon(Icons.play_arrow_rounded),
                    label: const Text('Start'),
                  ),
          ),
          const SizedBox(width: 8),
          OutlinedButton(
            onPressed: connected && !s.busy && !s.running && _tested ? _backToMine : null,
            child: const Text('Back to my spot'),
          ),
        ]),
        const SizedBox(height: 8),
        const Text(
            'Each move takes the glasses about a second, so a turn is never faster than that plus the times above. '
            'Stop (or leaving this page) puts the text back at your own height.',
            style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
      ]),
    );
  }

  @override
  Widget build(BuildContext context) {
    final p = _suite.position;
    return Scaffold(
      appBar: AppBar(title: const Text('Text height')),
      body: ListenableBuilder(
        listenable: Listenable.merge([p, _suite.ble, _send]),
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
                  onChanged: connected && !_send.running
                      ? (v) {
                          _homeH = v.round();
                          p.set(h: _homeH);
                        }
                      : null,
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
                  onChanged: connected && !_send.running
                      ? (v) {
                          _homeD = v.round();
                          p.set(d: _homeD);
                        }
                      : null,
                ),
              ),
              const SizedBox(height: 12),
              _spotCard('Top spot', Icons.vertical_align_top_rounded, _send.top, _topText, connected),
              const SizedBox(height: 12),
              _spotCard('Bottom spot', Icons.vertical_align_bottom_rounded, _send.bottom, _bottomText, connected),
              const SizedBox(height: 12),
              _testCard(connected),
              if (p.error != null) ...[
                const SizedBox(height: 10),
                Text(p.error!, style: const TextStyle(color: AppColors.warn)),
              ],
              const SizedBox(height: 18),
              FilledButton.icon(
                onPressed: connected && !_send.busy
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
