import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/suite.dart';
import '../../core/voice/speaker.dart';
import '../../core/voice/wake_word.dart';

String effortLabel(String e) => switch (e) {
      'low' => 'low effort',
      'high' => 'high effort',
      'xhigh' => 'extra-high effort',
      'max' => 'max effort',
      _ => 'medium effort',
    };

String accessLabel(String a) => switch (a) {
      'read' => 'looks only',
      'edit' => 'can edit files',
      _ => 'can do things',
    };

/// Everything about talking to Claude in one sheet: model and effort, what
/// Claude may do, answer length, the voice, back-and-forth, trigger words.
class ClaudeSettingsSheet extends StatelessWidget {
  const ClaudeSettingsSheet({super.key, this.scroll});

  final ScrollController? scroll;

  static Future<void> show(BuildContext context) => showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        backgroundColor: AppColors.surface,
        builder: (_) => DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.85,
          maxChildSize: 0.95,
          minChildSize: 0.4,
          builder: (context, scroll) => ClaudeSettingsSheet(scroll: scroll),
        ),
      );

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) => ListView(
        controller: scroll,
        padding: const EdgeInsets.fromLTRB(18, 0, 18, 28),
        children: [
          Text('Talking to Claude', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 4),
          const Text('Claude Code runs in Termux on this phone - no API key, your own Claude login.',
              style: TextStyle(color: AppColors.muted, fontSize: 13)),
          const _Head('Model'),
          _Choices(
            value: s.claudeModel,
            options: const [('opus', 'Opus'), ('sonnet', 'Sonnet'), ('haiku', 'Haiku (fastest)')],
            onChanged: (v) => s.set('claudeModel', v),
          ),
          const _Head('Effort (how hard it thinks)'),
          _Choices(
            value: s.claudeEffort,
            options: const [('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('xhigh', 'Extra'), ('max', 'Max')],
            onChanged: (v) => s.set('claudeEffort', v),
          ),
          const _Head('What Claude may do on this phone'),
          _Choices(
            value: s.claudeAccess,
            options: const [('read', 'Look only'), ('edit', 'Edit files'), ('auto', 'Do things')],
            onChanged: (v) => s.set('claudeAccess', v),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              switch (s.claudeAccess) {
                'read' => 'Reads files and searches, changes nothing.',
                'edit' => 'Also edits files; anything else that would need your OK is skipped.',
                _ => 'Runs commands and edits files, with Claude Code’s own safety checks (auto mode).',
              },
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
          ),
          const _Head('Answer length (on the glasses)'),
          _Choices(
            value: s.answerLength,
            options: const [('brief', 'Brief ~50'), ('normal', 'Normal ~150'), ('long', 'Long')],
            onChanged: (v) => s.set('answerLength', v),
          ),
          const Divider(height: 32),
          Text('Voice', style: Theme.of(context).textTheme.titleMedium),
          const _Head('Read answers out loud'),
          _Choices(
            value: s.speakReplies,
            options: const [('off', 'Off'), ('voice', 'When I talk'), ('always', 'Always')],
            onChanged: (v) => s.set('speakReplies', v),
          ),
          if (s.speakReplies != 'off') ...[
            const SizedBox(height: 8),
            _VoicePicker(suite: suite),
            _Slider(
              label: 'Speed',
              value: s.ttsRate,
              min: 60,
              max: 180,
              format: (v) => '${(v / 100).toStringAsFixed(2)}x',
              onChanged: (v) => s.set('ttsRate', v),
            ),
            _Slider(
              label: 'Pitch',
              value: s.ttsPitch,
              min: 70,
              max: 140,
              format: (v) => '${(v / 100).toStringAsFixed(2)}x',
              onChanged: (v) => s.set('ttsPitch', v),
            ),
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => suite.narrator.say('Hi, this is Claude. This is how I sound on your phone.'),
                icon: const Icon(Icons.play_arrow_rounded),
                label: const Text('Try the voice'),
              ),
            ),
          ],
          const Divider(height: 32),
          Text('Back and forth', style: Theme.of(context).textTheme.titleMedium),
          _Switch(
            'Listen for a follow-up',
            'After an answer, just keep talking - no button, no wake word. Silence or “thanks” ends it.',
            s.conversation,
            (v) => s.set('conversation', v),
          ),
          if (s.conversation)
            _Slider(
              label: 'Wait for a follow-up',
              value: s.followUpSec,
              min: 3,
              max: 20,
              format: (v) => '$v s',
              onChanged: (v) => s.set('followUpSec', v),
            ),
          _Switch(
            'Send when I pause',
            'No need to say “send” after the mic button, the TouchBar or a follow-up.',
            s.autoSend,
            (v) => s.set('autoSend', v),
          ),
          _Slider(
            label: 'Pause that ends a message',
            value: s.pauseMs ~/ 100,
            min: 7,
            max: 30,
            format: (v) => '${(v / 10).toStringAsFixed(1)} s',
            onChanged: (v) => s.set('pauseMs', v * 100),
          ),
          const Divider(height: 32),
          Text('Trigger words (hands-free)', style: Theme.of(context).textTheme.titleMedium),
          _Head('When does “${suite.voice.wake.label}” count?'),
          _Choices(
            value: s.wakeStrict,
            options: [
              (WakeStrictness.relaxed.name, 'Anywhere'),
              (WakeStrictness.normal.name, 'At the start'),
              (WakeStrictness.strict.name, '“Hey ${suite.voice.wake.label}”'),
            ],
            onChanged: (v) => s.set('wakeStrict', v),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              switch (s.wakeStrictness) {
                WakeStrictness.relaxed => 'The word anywhere starts a message. Easiest, but talk nearby can trigger it.',
                WakeStrictness.normal => 'Only when what you say starts with it (or “Hey …”). Mid-sentence it is ignored.',
                WakeStrictness.strict => 'Only “Hey …” or “OK …” at the start. Hardest to trigger by accident.',
              },
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
          ),
          _Switch(
            'Send after a pause for “${suite.voice.wake.label}…” too',
            'Off: a hands-free message only goes when you say “send” - nothing goes by accident.',
            s.wakeAutoSend,
            (v) => s.set('wakeAutoSend', v),
          ),
          Text('Words: ${s.wakeWords.join(', ')}  (change them in Settings)',
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
        ],
      ),
    );
  }
}

class _Head extends StatelessWidget {
  const _Head(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 16, bottom: 8),
        child: Text(text, style: const TextStyle(fontWeight: FontWeight.w700)),
      );
}

class _Choices extends StatelessWidget {
  const _Choices({required this.value, required this.options, required this.onChanged});
  final String value;
  final List<(String, String)> options;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) => Wrap(spacing: 8, runSpacing: 8, children: [
        for (final (v, label) in options)
          ChoiceChip(label: Text(label), selected: value == v, onSelected: (_) => onChanged(v)),
      ]);
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

class _Slider extends StatelessWidget {
  const _Slider(
      {required this.label,
      required this.value,
      required this.min,
      required this.max,
      required this.format,
      required this.onChanged});
  final String label;
  final int value;
  final int min;
  final int max;
  final String Function(int) format;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) => Row(children: [
        Expanded(flex: 4, child: Text(label, style: const TextStyle(fontSize: 13.5))),
        Expanded(
          flex: 6,
          child: Slider(
            value: value.clamp(min, max).toDouble(),
            min: min.toDouble(),
            max: max.toDouble(),
            divisions: max - min,
            label: format(value),
            onChanged: (v) => onChanged(v.round()),
          ),
        ),
        SizedBox(width: 48, child: Text(format(value), textAlign: TextAlign.end)),
      ]);
}

/// The phone's voices, best first (download more in Android's text-to-speech settings).
class _VoicePicker extends StatefulWidget {
  const _VoicePicker({required this.suite});
  final Suite suite;
  @override
  State<_VoicePicker> createState() => _VoicePickerState();
}

class _VoicePickerState extends State<_VoicePicker> {
  List<TtsVoice> _voices = const [];
  List<TtsEngineInfo> _engines = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final sp = widget.suite.speaker;
    final engines = await sp.engines();
    final voices = await sp.voices();
    if (!mounted) return;
    final lang = Localizations.localeOf(context).languageCode;
    voices.sort((a, b) {
      final la = a.locale.startsWith(lang) ? 0 : 1, lb = b.locale.startsWith(lang) ? 0 : 1;
      if (la != lb) return la - lb;
      return b.quality - a.quality;
    });
    setState(() {
      _engines = engines;
      _voices = voices;
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.suite.settings;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      if (_engines.length > 1)
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: DropdownButtonFormField<String>(
            value: _engines.any((e) => e.name == s.ttsEngine) ? s.ttsEngine : '',
            isExpanded: true,
            decoration: const InputDecoration(labelText: 'Voice app'),
            items: [
              const DropdownMenuItem(value: '', child: Text('Phone default')),
              for (final e in _engines) DropdownMenuItem(value: e.name, child: Text(e.label, overflow: TextOverflow.ellipsis)),
            ],
            onChanged: (v) {
              s.set('ttsEngine', v ?? '');
              s.set('ttsVoice', '');
              Future.delayed(const Duration(milliseconds: 800), _load);
            },
          ),
        ),
      DropdownButtonFormField<String>(
        value: _voices.any((v) => v.name == s.ttsVoice) ? s.ttsVoice : '',
        isExpanded: true,
        decoration: const InputDecoration(labelText: 'Voice'),
        items: [
          const DropdownMenuItem(value: '', child: Text('Default voice')),
          for (final v in _voices.take(60))
            DropdownMenuItem(value: v.name, child: Text(v.label, overflow: TextOverflow.ellipsis)),
        ],
        onChanged: (v) => s.set('ttsVoice', v ?? ''),
      ),
      const SizedBox(height: 4),
      const Text('“Best” voices sound most natural. Google’s are under Settings › Text-to-speech.',
          style: TextStyle(color: AppColors.muted, fontSize: 12)),
    ]);
  }
}
