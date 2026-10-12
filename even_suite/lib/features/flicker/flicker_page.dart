import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/flicker.dart';
import '../../core/settings.dart';
import '../../widgets/common.dart';

/// Times for the spot sliders (ms), finer where it matters.
const _times = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000];

String _secs(int ms) => ms < 1000 ? '$ms ms' : '${(ms / 1000).toStringAsFixed(ms % 1000 == 0 ? 0 : 1)} s';

/// Flicker tab: text at a high and a low spot in your view, switching on a
/// timer, to try to cover more of your view than one spot.
class FlickerPage extends StatelessWidget {
  const FlickerPage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final f = suite.flicker;
    final s = suite.settings;
    return ListenableBuilder(
      listenable: Listenable.merge([f, s, suite.ble]),
      builder: (context, _) {
        final (label, color) = f.running
            ? ('${f.rate.toStringAsFixed(1)} switches/s', AppColors.ok)
            : f.busy
                ? ('Sending…', AppColors.warn)
                : ('Off', AppColors.muted);
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            Row(children: [
              Expanded(
                child: Text('Flicker',
                    style: Theme.of(context).textTheme.headlineSmall, maxLines: 1, overflow: TextOverflow.ellipsis),
              ),
              StatusPill(label, color: color, pulse: f.running || f.busy),
            ]),
            const SizedBox(height: 4),
            const Text(
              'Moves the glasses’ text up and down in your view on a timer, with different text at each spot, '
              'to try to fill more of your view.',
              style: TextStyle(color: AppColors.muted),
            ),
            const SizedBox(height: 14),
            _ViewPreview(flicker: f, settings: s),
            const SizedBox(height: 10),
            if (f.running || f.steps > 0) _Stats(flicker: f),
            if (f.error != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(f.error!, style: const TextStyle(color: AppColors.warn)),
              ),
            const SizedBox(height: 12),
            if (!f.running)
              FilledButton.icon(
                onPressed: f.busy ? null : suite.startFlicker,
                icon: const Icon(Icons.swap_vert_rounded),
                label: const Text('Start flicker'),
              )
            else
              FilledButton.icon(
                style: FilledButton.styleFrom(backgroundColor: AppColors.claude),
                onPressed: f.stop,
                icon: const Icon(Icons.stop_rounded),
                label: const Text('Stop'),
              ),
            if (f.running && f.offAt != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text('Stops by itself at ${_clock(f.offAt!)}',
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
              ),
            const SizedBox(height: 8),
            Row(children: [
              for (final (i, text, onTap) in [
                (0, 'Top once', () => f.showOnce(true)),
                (1, 'Bottom once', () => f.showOnce(false)),
                (2, 'Reset', f.reset),
              ]) ...[
                if (i > 0) const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton(
                    style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 6)),
                    onPressed: f.running || f.busy ? null : onTap,
                    child: FittedBox(fit: BoxFit.scaleDown, child: Text(text, maxLines: 1)),
                  ),
                ),
              ],
            ]),
            const SizedBox(height: 6),
            const Text(
              'Try “Top once” and “Bottom once” first to see the text move. Reset puts it back at your normal spot.',
              style: TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            const SizedBox(height: 18),
            // Keyed: rows above appear while running, and the text boxes must
            // keep their own state when the list shifts.
            SectionCard(
              key: const ValueKey('flick-top'),
              title: 'Top text',
              icon: Icons.vertical_align_top_rounded,
              iconColor: AppColors.hud,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _TextBox(settingKey: 'flickTop', initial: s.flickTop, hint: 'Text for the top spot'),
                const SizedBox(height: 6),
                _StepSlider('Height', s.flickTopPos, 0, 8, (v) => s.set('flickTopPos', v),
                    caption: '0 = lowest, 8 = highest'),
              ]),
            ),
            const SizedBox(height: 12),
            SectionCard(
              key: const ValueKey('flick-bottom'),
              title: 'Bottom text',
              icon: Icons.vertical_align_bottom_rounded,
              iconColor: AppColors.glasses,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _TextBox(settingKey: 'flickBottom', initial: s.flickBottom, hint: 'Text for the bottom spot'),
                const SizedBox(height: 6),
                _StepSlider('Height', s.flickBottomPos, 0, 8, (v) => s.set('flickBottomPos', v),
                    caption: '0 = lowest, 8 = highest'),
              ]),
            ),
            const SizedBox(height: 12),
            SectionCard(
              key: const ValueKey('flick-timing'),
              title: 'Timing',
              icon: Icons.timer_outlined,
              iconColor: AppColors.warn,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _TimeSlider('Top', s.flickTopMs, (v) => s.set('flickTopMs', v)),
                _TimeSlider('Bottom', s.flickBottomMs, (v) => s.set('flickBottomMs', v)),
                const SizedBox(height: 4),
                const Text('Both spots:', style: TextStyle(color: AppColors.muted, fontSize: 13)),
                const SizedBox(height: 6),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final ms in const [50, 100, 200, 500, 1000])
                    ChoiceChip(
                      label: Text(_secs(ms)),
                      selected: s.flickTopMs == ms && s.flickBottomMs == ms,
                      onSelected: (_) {
                        s.set('flickTopMs', ms);
                        s.set('flickBottomMs', ms);
                      },
                    ),
                ]),
                const SizedBox(height: 12),
                const Text('Stops by itself after:', style: TextStyle(color: AppColors.muted, fontSize: 13)),
                const SizedBox(height: 6),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (sec, text) in const [(15, '15 s'), (30, '30 s'), (60, '1 min'), (120, '2 min'), (300, '5 min')])
                    ChoiceChip(
                      label: Text(text),
                      selected: s.flickAutoStop == sec,
                      onSelected: (_) => s.set('flickAutoStop', sec),
                    ),
                ]),
                const SizedBox(height: 8),
                Text(
                  'Asked: ${f.askedRate.toStringAsFixed(1)} switches a second. Changes apply while it runs.',
                  style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                ),
              ]),
            ),
            const SizedBox(height: 12),
            SectionCard(
              key: const ValueKey('flick-tuning'),
              title: 'Fine tuning',
              icon: Icons.tune_rounded,
              iconColor: AppColors.claude,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _Chips('How to move', s.flickMethod,
                    const [('live', 'Live (fast)'), ('apply', 'Like the Even app')], (v) => s.set('flickMethod', v)),
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Text(
                    s.flickMethod == 'apply'
                        ? 'Each move opens the glasses’ height preview, waits, then closes it. Slower: the glasses '
                            'pause about half a second after each close.'
                        : 'Each move uses the glasses’ height preview, which moves the picture at once. The preview '
                            'is closed when you stop.',
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                  ),
                ),
                _PauseSlider('Pause after move', s.flickSettleMs, (v) => s.set('flickSettleMs', v)),
                if (s.flickMethod == 'apply')
                  _PauseSlider('Preview time', s.flickPreviewMs, (v) => s.set('flickPreviewMs', v)),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Wait for the glasses to answer'),
                  subtitle: const Text('On: steadier. Off: faster, may skip.'),
                  value: s.flickWait,
                  onChanged: (v) => s.set('flickWait', v),
                ),
                _Chips('Switch', s.flickWhat, const [('both', 'Text + spot'), ('move', 'Spot only'), ('text', 'Text only')],
                    (v) => s.set('flickWhat', v)),
                _Chips('Moves go to', s.flickArms, const [('R', 'Right arm'), ('both', 'Both arms')],
                    (v) => s.set('flickArms', v)),
                _StepSlider('Depth', s.flickDepth, 1, 9, (v) => s.set('flickDepth', v),
                    caption: 'How far away the text looks while flickering'),
                const Divider(height: 24),
                const Text('Your normal spot (put back when it stops)',
                    style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                _StepSlider('Height', s.flickHomePos, 0, 8, (v) => s.set('flickHomePos', v)),
                _StepSlider('Depth', s.flickHomeDepth, 1, 9, (v) => s.set('flickHomeDepth', v)),
                const SizedBox(height: 4),
                const Text(
                  'Set these to the display height and depth you use in the Even app.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5),
                ),
              ]),
            ),
          ],
        );
      },
    );
  }
}

String _clock(DateTime t) =>
    '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:${t.second.toString().padLeft(2, '0')}';

class _Stats extends StatelessWidget {
  const _Stats({required this.flicker});
  final Flicker flicker;

  @override
  Widget build(BuildContext context) {
    final f = flicker;
    final parts = [
      '${f.steps} switches',
      if (f.running) '${f.rate.toStringAsFixed(1)}/s of ${f.askedRate.toStringAsFixed(1)} asked',
      if (f.what != 'text') 'move ${f.lastMoveMs} ms',
      if (f.what != 'move') 'text ${f.lastTextMs} ms',
      if (f.slow > 0) '${f.slow} slow',
      if (f.failures > 0) '${f.failures} no answer',
    ];
    final notes = [
      if (f.moveSilent) 'The glasses don’t answer moves; sending them without waiting.',
      if (f.textSilent) 'The glasses don’t answer text; sending it without waiting.',
      if (f.moveReply.isNotEmpty) 'Move answer: ${f.moveReply}',
    ];
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(parts.join(' · '), style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
      for (final n in notes)
        Padding(
          padding: const EdgeInsets.only(top: 2),
          child: Text(n, style: const TextStyle(color: AppColors.muted, fontSize: 12)),
        ),
    ]);
  }
}

/// The view through the glasses: where each text sits, the live one lit.
class _ViewPreview extends StatelessWidget {
  const _ViewPreview({required this.flicker, required this.settings});
  final Flicker flicker;
  final Settings settings;

  @override
  Widget build(BuildContext context) {
    final f = flicker;
    final s = settings;
    final live = f.running || f.busy;
    Widget spot(String text, int height, bool on) => LayoutBuilder(builder: (context, c) {
          const boxH = 46.0;
          final y = (8 - height) / 8 * (c.maxHeight - boxH);
          return Stack(children: [
            Positioned(
              left: 0,
              right: 0,
              top: y,
              height: boxH,
              child: AnimatedOpacity(
                duration: const Duration(milliseconds: 60),
                opacity: !live || on ? 1 : 0.22,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    border: Border.all(color: AppColors.hud.withOpacity(on && live ? 0.9 : 0.35)),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  alignment: Alignment.centerLeft,
                  child: Text(
                    text.isEmpty ? ' ' : text,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: AppColors.hud, fontSize: 12.5, height: 1.25),
                  ),
                ),
              ),
            ),
          ]);
        });
    return MediaQuery.withNoTextScaling(
      child: Container(
        height: 190,
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppColors.hudBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.outline),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Expanded(child: Stack(fit: StackFit.expand, children: [
            spot(s.flickBottom, s.flickBottomPos, !f.top),
            spot(s.flickTop, s.flickTopPos, f.top),
          ])),
          const SizedBox(width: 10),
          SizedBox(
            width: 34,
            child: Column(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              const Text('high', style: TextStyle(color: AppColors.muted, fontSize: 10.5)),
              Icon(Icons.swap_vert_rounded, color: live ? AppColors.hud : AppColors.muted, size: 20),
              const Text('low', style: TextStyle(color: AppColors.muted, fontSize: 10.5)),
            ]),
          ),
        ]),
      ),
    );
  }
}

/// Text for one spot; saved as you type (applies on the next switch).
class _TextBox extends StatefulWidget {
  const _TextBox({required this.settingKey, required this.initial, required this.hint});
  final String settingKey;
  final String initial;
  final String hint;

  @override
  State<_TextBox> createState() => _TextBoxState();
}

class _TextBoxState extends State<_TextBox> {
  late final _ctl = TextEditingController(text: widget.initial);

  @override
  void didUpdateWidget(_TextBox old) {
    super.didUpdateWidget(old);
    if (old.settingKey != widget.settingKey) _ctl.text = widget.initial;
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => TextField(
        controller: _ctl,
        minLines: 2,
        maxLines: 4,
        maxLength: 180,
        decoration: InputDecoration(hintText: widget.hint, counterText: ''),
        onChanged: (v) => SuiteScope.of(context).settings.set(widget.settingKey, v),
      );
}

class _StepSlider extends StatelessWidget {
  const _StepSlider(this.label, this.value, this.min, this.max, this.onChanged, {this.caption});
  final String label;
  final int value, min, max;
  final ValueChanged<int> onChanged;
  final String? caption;

  @override
  Widget build(BuildContext context) => Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          SizedBox(width: 56, child: Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 13))),
          Expanded(
            child: Slider(
              value: value.clamp(min, max).toDouble(),
              min: min.toDouble(),
              max: max.toDouble(),
              divisions: max - min,
              label: '$value',
              onChanged: (v) => onChanged(v.round()),
            ),
          ),
          SizedBox(width: 28, child: Text('$value', textAlign: TextAlign.end, style: const TextStyle(fontSize: 13))),
        ]),
        if (caption != null)
          Text(caption!, style: const TextStyle(color: AppColors.muted, fontSize: 12)),
      ]);
}

class _TimeSlider extends StatelessWidget {
  const _TimeSlider(this.label, this.ms, this.onChanged);
  final String label;
  final int ms;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    var i = _times.indexWhere((t) => t >= ms);
    if (i < 0) i = _times.length - 1;
    return Row(children: [
      SizedBox(width: 56, child: Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 13))),
      Expanded(
        child: Slider(
          value: i.toDouble(),
          min: 0,
          max: (_times.length - 1).toDouble(),
          divisions: _times.length - 1,
          label: _secs(_times[i]),
          onChanged: (v) => onChanged(_times[v.round()]),
        ),
      ),
      SizedBox(width: 56, child: Text(_secs(ms), textAlign: TextAlign.end, style: const TextStyle(fontSize: 13))),
    ]);
  }
}

/// Pauses (ms) for the fine-tuning sliders, 0 = none.
const _pauses = [0, 20, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000];

class _PauseSlider extends StatelessWidget {
  const _PauseSlider(this.label, this.ms, this.onChanged);
  final String label;
  final int ms;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    var i = _pauses.indexWhere((t) => t >= ms);
    if (i < 0) i = _pauses.length - 1;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 13)),
      Row(children: [
        Expanded(
          child: Slider(
            value: i.toDouble(),
            min: 0,
            max: (_pauses.length - 1).toDouble(),
            divisions: _pauses.length - 1,
            label: ms == 0 ? 'none' : _secs(_pauses[i]),
            onChanged: (v) => onChanged(_pauses[v.round()]),
          ),
        ),
        SizedBox(
            width: 56,
            child: Text(ms == 0 ? 'none' : _secs(ms), textAlign: TextAlign.end, style: const TextStyle(fontSize: 13))),
      ]),
    ]);
  }
}

class _Chips extends StatelessWidget {
  const _Chips(this.label, this.value, this.options, this.onChanged);
  final String label;
  final String value;
  final List<(String, String)> options;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 13)),
          const SizedBox(height: 6),
          Wrap(spacing: 8, runSpacing: 8, children: [
            for (final (v, text) in options)
              ChoiceChip(label: Text(text), selected: value == v, onSelected: (_) => onChanged(v)),
          ]),
        ]),
      );
}
