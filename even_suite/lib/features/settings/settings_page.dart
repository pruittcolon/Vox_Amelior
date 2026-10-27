import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/glasses_text.dart';
import '../../core/settings.dart';
import '../../core/voice/wake_word.dart';
import '../../widgets/common.dart';
import '../claude/claude_settings_sheet.dart';
import '../lookup/lookup_page.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
          children: [
            _Section('Claude, voice and look up', Icons.tune_rounded, AppColors.claude, [
              const Text('Model and effort, what Claude may do, the speaking voice, follow-ups and trigger words.',
                  style: TextStyle(color: AppColors.muted)),
              const SizedBox(height: 10),
              Row(children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () => ClaudeSettingsSheet.show(context),
                    icon: const Icon(Icons.tune_rounded),
                    label: const Text('Claude and voice'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () =>
                        Navigator.of(context).push(MaterialPageRoute(builder: (_) => const LookUpPage())),
                    icon: const Icon(Icons.dashboard_customize_rounded),
                    label: const Text('Look up'),
                  ),
                ),
              ]),
            ]),
            _Section('Voice', Icons.record_voice_over_rounded, AppColors.claude, [
              const _Label('Hands-free turns itself off after'),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final m in Settings.autoOffChoices)
                  ChoiceChip(
                    label: Text('$m min'),
                    selected: s.autoOffMinutes == m,
                    onSelected: (_) => s.set('autoOffMinutes', m),
                  ),
              ]),
              const SizedBox(height: 14),
              _TextSetting(
                label: 'Wake words',
                help: 'Any of these starts a message. Separate with commas.',
                value: s.wakeWords.join(', '),
                onSave: (v) {
                  final words = v.split(',').map((w) => w.trim()).where((w) => w.isNotEmpty).toList();
                  s.set('wakeWords', words.isEmpty ? List.of(WakeWord.defaults) : words);
                },
              ),
              _Switch(
                'Quiet start chime',
                'Mutes the recognizer’s beep while waiting for “Claude” (briefly dips media volume).',
                s.quietChime,
                (v) => s.set('quietChime', v),
              ),
              _Switch(
                'Hold left TouchBar to talk',
                'Long-press on the glasses records a message; let go to send.',
                s.holdToTalk,
                (v) => s.set('holdToTalk', v),
              ),
              _Switch(
                'Use the glasses microphone',
                'When holding the TouchBar, listen through the glasses (Android 13+; otherwise the phone mic).',
                s.glassesMic,
                (v) => s.set('glassesMic', v),
              ),
              _Switch(
                'Show my words while holding (test)',
                'Writes what you say on the glasses while the TouchBar is held. Turn off if the glasses stop hearing you.',
                s.wordsWhileHolding,
                (v) => s.set('wordsWhileHolding', v),
              ),
              const SizedBox(height: 8),
              const _Label('Speech recognizer'),
              SegmentedButton<String>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: 'auto', label: Text('Auto')),
                  ButtonSegment(value: 'ondevice', label: Text('On-device')),
                  ButtonSegment(value: 'default', label: Text('Google')),
                ],
                selected: {s.recognizer},
                onSelectionChanged: (v) => s.set('recognizer', v.first),
              ),
              const SizedBox(height: 14),
              _TextSetting(
                label: 'Language',
                help: 'Leave empty for the phone’s language, or e.g. en-US.',
                value: s.language,
                onSave: (v) => s.set('language', v.trim()),
              ),
            ]),
            _Section('Glasses', Icons.visibility_rounded, AppColors.glasses, [
              _Switch('Show replies on the glasses', 'Claude’s answers and the terminal appear on the display.',
                  s.repliesOnGlasses, (v) => s.set('repliesOnGlasses', v)),
              _Switch('Volume keys page the glasses', 'While the app is open: volume up = up, down = down.',
                  s.volumeKeysScroll, (v) => s.set('volumeKeysScroll', v)),
              const SizedBox(height: 8),
              const _Label('Teleprompter: replies turn their own pages'),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final (wpm, label) in const [(0, 'Off (tap to page)'), (140, 'Slow'), (180, 'Normal'), (230, 'Fast')])
                  ChoiceChip(
                    label: Text(label),
                    selected: s.readingWpm == wpm,
                    onSelected: (_) => s.set('readingWpm', wpm),
                  ),
              ]),
              const SizedBox(height: 6),
              const Text('Right tap = next page and keep going. Left tap = back and pause.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
              const SizedBox(height: 10),
              Builder(builder: (context) {
                final size = GlassesText.screenSize();
                return Text(
                    'Claude is told your screen fits ${size.lines} lines × ~${size.charsPerLine} characters '
                    '(~${size.chars} per screen) and answers in chunks that size.',
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5));
              }),
            ]),
            _Section('Termux bridge', Icons.terminal_rounded, AppColors.ok, [
              _TextSetting(
                label: 'Bridge address',
                help: 'Agent Hub on this phone is http://127.0.0.1:8787.',
                value: s.bridgeUrl,
                keyboard: TextInputType.url,
                onSave: (v) => s.set('bridgeUrl', v.trim().isEmpty ? Settings.defaultBridgeUrl : v.trim()),
              ),
              _TextSetting(
                label: 'Start command (run by Termux)',
                help: 'Needs “allow-external-apps = true” in ~/.termux/termux.properties.',
                value: s.startCommand,
                maxLines: 3,
                onSave: (v) => s.set('startCommand', v.trim().isEmpty ? Settings.defaultStartCommand : v.trim()),
              ),
              Row(children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () async {
                      final ok = await suite.chat.check();
                      if (context.mounted) showToast(context, ok ? 'Bridge is online' : 'Bridge is not answering');
                    },
                    icon: const Icon(Icons.wifi_tethering_rounded),
                    label: const Text('Test'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () async {
                      final opened = await suite.system.openTermux();
                      if (!opened && context.mounted) showToast(context, 'Termux is not installed');
                    },
                    icon: const Icon(Icons.open_in_new_rounded),
                    label: const Text('Open Termux'),
                  ),
                ),
              ]),
            ]),
            _Section('Permissions', Icons.shield_rounded, AppColors.warn, [
              const Text(
                  'Bluetooth (“Nearby devices”), Microphone, Notifications and “Run commands in Termux”.',
                  style: TextStyle(color: AppColors.muted)),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: suite.system.openAppSettings,
                icon: const Icon(Icons.app_settings_alt_rounded),
                label: const Text('Open app permissions'),
              ),
            ]),
            const SizedBox(height: 8),
            const Center(
              child: Text('Even Suite · built on the open-source Even Realities demo (BSD-2)',
                  textAlign: TextAlign.center, style: TextStyle(color: AppColors.muted, fontSize: 12)),
            ),
          ],
        ),
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section(this.title, this.icon, this.color, this.children);
  final String title;
  final IconData icon;
  final Color color;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: SectionCard(
          title: title,
          icon: icon,
          iconColor: color,
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: children),
        ),
      );
}

class _Label extends StatelessWidget {
  const _Label(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
      );
}

class _Switch extends StatelessWidget {
  const _Switch(this.title, this.subtitle, this.value, this.onChanged);
  final String title;
  final String subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) => SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        subtitle: Text(subtitle, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
        value: value,
        onChanged: onChanged,
      );
}

/// A text field that saves when you press Done or leave it.
class _TextSetting extends StatefulWidget {
  const _TextSetting({
    required this.label,
    required this.help,
    required this.value,
    required this.onSave,
    this.maxLines = 1,
    this.keyboard,
  });

  final String label;
  final String help;
  final String value;
  final ValueChanged<String> onSave;
  final int maxLines;
  final TextInputType? keyboard;

  @override
  State<_TextSetting> createState() => _TextSettingState();
}

class _TextSettingState extends State<_TextSetting> {
  late final _c = TextEditingController(text: widget.value);
  final _focus = FocusNode();

  @override
  void initState() {
    super.initState();
    _focus.addListener(() {
      if (!_focus.hasFocus) _save();
    });
  }

  void _save() {
    if (_c.text != widget.value) widget.onSave(_c.text);
  }

  @override
  void didUpdateWidget(_TextSetting old) {
    super.didUpdateWidget(old);
    if (!_focus.hasFocus && widget.value != _c.text) _c.text = widget.value;
  }

  @override
  void dispose() {
    _c.dispose();
    _focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 14),
        child: TextField(
          controller: _c,
          focusNode: _focus,
          maxLines: widget.maxLines,
          minLines: 1,
          keyboardType: widget.keyboard,
          autocorrect: false,
          onSubmitted: (_) => _save(),
          decoration: InputDecoration(labelText: widget.label, helperText: widget.help, helperMaxLines: 2),
        ),
      );
}
