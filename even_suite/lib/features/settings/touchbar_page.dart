import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../widgets/common.dart';
import '../lookup/lookup_page.dart';

/// What the glasses' gestures do: hold the left TouchBar, double tap, look up.
class TouchBarPage extends StatelessWidget {
  const TouchBarPage({super.key});

  @override
  Widget build(BuildContext context) {
    final s = SuiteScope.of(context).settings;
    return Scaffold(
      appBar: AppBar(title: const Text('TouchBar')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
          children: [
            const Text('Pick what each gesture does. Changes apply at once.',
                style: TextStyle(color: AppColors.muted)),
            const SizedBox(height: 16),
            _Gesture(
              icon: Icons.touch_app_rounded,
              color: AppColors.claude,
              title: 'Hold the left TouchBar',
              hint: 'Then talk; it sends when you pause. The glasses flash their own “Even AI” screen first '
                  '(their firmware, before the app hears anything) - double tap to talk skips it.',
              value: s.holdToTalk ? s.holdAction : 'off',
              options: const [
                ('mode', 'Whichever mode is on', 'Chat or terminal - switch by saying “terminal” or “chat”'),
                ('chat', 'Always Claude chat', 'Quick questions, answers on the glasses'),
                ('terminal', 'Always the terminal', 'Typed into Claude Code in Termux'),
                ('off', 'Nothing', 'Leave the TouchBar to the glasses'),
              ],
              onChanged: (v) {
                s.set('holdToTalk', v != 'off');
                if (v != 'off') s.set('holdAction', v);
              },
            ),
            const SizedBox(height: 12),
            _Gesture(
              icon: Icons.ads_click_rounded,
              color: AppColors.glasses,
              title: 'Double tap',
              hint: 'On a screen it closes it. With nothing showing, it does this:',
              value: s.doubleTapAction,
              options: const [
                ('talk', 'Talk (no Even AI screen)', 'Listening at once - the glasses never show “Even AI”'),
                ('close', 'Nothing', 'Just closes what is on the screen'),
                ('switch', 'Switch chat / terminal', 'Shows which one is on now'),
                ('repeat', 'Show the last answer', 'Brings Claude’s last answer back'),
              ],
              onChanged: (v) => s.set('doubleTapAction', v),
            ),
            const SizedBox(height: 12),
            _Gesture(
              icon: Icons.keyboard_double_arrow_up_rounded,
              color: AppColors.hud,
              title: 'Look up',
              hint: 'Tilt your head up to see it.',
              value: s.lookMode,
              options: const [
                ('custom', 'My layout', 'Your own screen - designed on the Look up page'),
                ('dashboard', 'Glasses dashboard', 'The glasses’ own clock and notes'),
              ],
              onChanged: (v) => s.set('lookMode', v),
              trailing: TextButton(
                onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const LookUpPage())),
                child: const Text('Design it'),
              ),
            ),
            const SizedBox(height: 12),
            SectionCard(
              title: 'While it listens',
              icon: Icons.graphic_eq_rounded,
              iconColor: AppColors.claude,
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                const Text(
                    'A pause sends what you said. A single tap sends it at once - the glasses report taps only on '
                    'Even AI style screens, so switch that on below if you want taps.',
                    style: TextStyle(color: AppColors.muted, fontSize: 13)),
                const SizedBox(height: 10),
                const Text('Send after a pause of', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (ms, label) in const [(700, '0.7 s'), (900, '0.9 s'), (1300, '1.3 s'), (2000, '2 s')])
                    ChoiceChip(
                      label: Text(label),
                      selected: s.touchPauseMs == ms,
                      onSelected: (_) => s.set('touchPauseMs', ms),
                    ),
                ]),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Show my words while I talk', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('“Termux-Claude listening” and what you say, live.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  value: s.wordsWhileHolding,
                  onChanged: (v) => s.set('wordsWhileHolding', v),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Even AI style screens', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Taps work, but the glasses fade each update in.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  value: s.aiStyle,
                  onChanged: (v) => s.set('aiStyle', v),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Faster screens (test)', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Sends each screen to both arms at once - about 50 ms quicker. Switch off '
                      'if one side ever shows a different screen.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  value: s.bleParallel,
                  onChanged: (v) => s.set('bleParallel', v),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Use the glasses microphone', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Hears you, not the room. Needs the Parakeet speech engine.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  value: s.glassesMic,
                  onChanged: (v) => s.set('glassesMic', v),
                ),
              ]),
            ),
          ],
        ),
      ),
    );
  }
}

class _Gesture extends StatelessWidget {
  const _Gesture({
    required this.icon,
    required this.color,
    required this.title,
    required this.hint,
    required this.value,
    required this.options,
    required this.onChanged,
    this.trailing,
  });

  final IconData icon;
  final Color color;
  final String title;
  final String hint;
  final String value;
  final List<(String, String, String)> options;
  final ValueChanged<String> onChanged;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) => SectionCard(
        title: title,
        icon: icon,
        iconColor: color,
        trailing: trailing,
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text(hint, style: const TextStyle(color: AppColors.muted, fontSize: 13)),
          const SizedBox(height: 8),
          for (final (v, label, sub) in options)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: _Option(
                selected: value == v,
                color: color,
                label: label,
                sub: sub,
                onTap: () => onChanged(v),
              ),
            ),
        ]),
      );
}

class _Option extends StatelessWidget {
  const _Option({required this.selected, required this.color, required this.label, required this.sub, required this.onTap});
  final bool selected;
  final Color color;
  final String label;
  final String sub;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(
        color: selected ? color.withOpacity(0.12) : AppColors.surfaceHigh,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14),
          side: BorderSide(color: selected ? color : Colors.transparent),
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
            child: Row(children: [
              Icon(selected ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded,
                  color: selected ? color : AppColors.muted, size: 20),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
                  Text(sub, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
                ]),
              ),
            ]),
          ),
        ),
      );
}
