import 'package:flutter/material.dart';

import '../../app/theme.dart';
import '../../core/voice/voice_commands.dart';
import '../../widgets/common.dart';

/// Everything you can say, in one place.
class HelpPage extends StatelessWidget {
  const HelpPage({super.key});

  static const _commands = [
    (VoiceCommand.scrollDown, 'Next page on the glasses'),
    (VoiceCommand.scrollUp, 'Previous page'),
    (VoiceCommand.top, 'First page'),
    (VoiceCommand.bottom, 'Last page'),
    (VoiceCommand.repeat, 'Show Claude’s last reply again'),
    (VoiceCommand.hide, 'Clear the glasses'),
    (VoiceCommand.stopReply, 'Stop Claude mid-answer'),
    (VoiceCommand.toTerminal, 'Send messages to the terminal'),
    (VoiceCommand.toChat, 'Send messages to Claude chat'),
    (VoiceCommand.enter, 'Press Enter in the terminal'),
    (VoiceCommand.escape, 'Press Escape in the terminal'),
    (VoiceCommand.choice1, 'Pick option 1 (e.g. “Yes” in Claude Code)'),
    (VoiceCommand.stopListening, 'Turn hands-free off'),
  ];

  @override
  Widget build(BuildContext context) {
    Widget step(String n, String title, String body) => Padding(
          padding: const EdgeInsets.only(bottom: 14),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            CircleAvatar(
              radius: 14,
              backgroundColor: AppColors.claude.withOpacity(0.18),
              child: Text(n, style: const TextStyle(color: AppColors.claude, fontWeight: FontWeight.w700)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
                const SizedBox(height: 2),
                Text(body, style: const TextStyle(color: AppColors.muted, height: 1.4)),
              ]),
            ),
          ]),
        );
    return Scaffold(
      appBar: AppBar(title: const Text('How to talk to it')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
        children: [
          SectionCard(
            title: 'Hands-free',
            icon: Icons.hearing_rounded,
            iconColor: AppColors.claude,
            child: Column(children: [
              step('1', 'Turn on “Hands-free”', 'On the Claude tab. It turns itself off after the minutes you pick in Settings, and the notification has a Stop button.'),
              step('2', 'Say “Claude”', '“Clod”, “Claud” and “Cloud” at the start also work. Say it more than once if it did not hear you.'),
              step('3', 'Say your message, then “send”', '“Claude, what’s on my calendar… send.” Say “cancel” or “never mind” to drop it.'),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: 'Without hands-free',
            icon: Icons.touch_app_rounded,
            iconColor: AppColors.glasses,
            child: Column(children: [
              step('•', 'Tap the mic', 'Speak, then say “send” or tap the arrow. Tap the mic again to stop and edit first.'),
              step('•', 'Hold the left TouchBar', 'Talk while holding; let go to send. (Turn off in Settings if you prefer.)'),
              step('•', 'Page through replies', 'Tap the left TouchBar for up and the right for down, or use ▲ ▼ in the app, or the volume keys (Settings).'),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: 'Say “Claude” and then…',
            icon: Icons.record_voice_over_rounded,
            iconColor: AppColors.ok,
            child: Column(children: [
              for (final (cmd, what) in _commands)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Expanded(
                      flex: 5,
                      child: Text('“${VoiceCommands.phrases[cmd]!.first}”',
                          style: const TextStyle(fontWeight: FontWeight.w700, color: AppColors.text)),
                    ),
                    Expanded(flex: 7, child: Text(what, style: const TextStyle(color: AppColors.muted))),
                  ]),
                ),
            ]),
          ),
        ],
      ),
    );
  }
}
