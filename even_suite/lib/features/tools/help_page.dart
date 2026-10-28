import 'package:flutter/material.dart';

import '../../app/shell.dart';
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
    (VoiceCommand.pause, 'Pause the teleprompter'),
    (VoiceCommand.resume, 'Keep scrolling'),
    (VoiceCommand.faster, 'Scroll faster'),
    (VoiceCommand.slower, 'Scroll slower'),
    (VoiceCommand.repeat, 'Show Claude’s last reply again'),
    (VoiceCommand.hide, 'Clear the glasses'),
    (VoiceCommand.stopReply, 'Stop Claude mid-answer'),
    (VoiceCommand.toTerminal, 'Send messages to the terminal'),
    (VoiceCommand.toChat, 'Send messages to Claude chat'),
    (VoiceCommand.newSession, 'Open another Claude Code session and talk to it'),
    (VoiceCommand.nextSession, 'Switch to the next Claude Code session'),
    (VoiceCommand.enter, 'Press Enter in the terminal'),
    (VoiceCommand.escape, 'Press Escape in the terminal'),
    (VoiceCommand.choice1, 'Pick option 1 (e.g. “Yes” in Claude Code)'),
    (VoiceCommand.stopListening, 'Turn hands-free off'),
  ];

  @override
  Widget build(BuildContext context) {
    final word = SuiteScope.of(context).voice.wake.label;
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
              step('2', 'Say “$word”', 'At the start of what you say (or “Hey $word”). Mentions mid-sentence are ignored, so talk around you does not wake it. Stricter or looser: Tools › Claude and voice.'),
              step('3', 'Say your message, then “send”', '“$word, what’s on my calendar… send.” Say “cancel” or “never mind” to drop it. Nothing is sent without “send” (unless you turn that on).'),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: 'Without hands-free',
            icon: Icons.touch_app_rounded,
            iconColor: AppColors.glasses,
            child: Column(children: [
              step('•', 'Tap the mic and talk', 'It sends when you pause. Tap again to stop early and edit. While Claude talks, tap to stop it.'),
              step('•', 'Hold the left TouchBar', 'The glasses say “Termux-Claude listening” and show what you say. It sends when you pause - no “$word”, no “send”. Tap to send at once, double tap to cancel.'),
              step('•', 'Watch it work', 'The glasses show “Got it” with your question, then what Claude is doing (“Searching the web 4s”), then the answer as it is written.'),
              step('•', 'Read replies like a teleprompter', 'Pages turn on their own at reading speed. Right tap = next and keep going; left tap = back and pause. Or use ▲ ▼, the volume keys, or your voice.'),
              step('•', 'Talk to the terminal', 'Hold and say “terminal”. Then hold, say “start”, your words, and “send” (or pause): it is typed into Claude Code in Termux and its steps and answer show like a chat answer. Say “chat” to go back.'),
              step('•', 'A tiny * means done', 'When a Claude Code job finishes (or an answer you double-tapped away is ready), a * shows for 3 seconds. Switch it off on the Work tab.'),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: 'Back and forth',
            icon: Icons.forum_rounded,
            iconColor: AppColors.claude,
            child: Column(children: [
              step('1', 'It answers out loud', 'Answers to spoken questions are read aloud on the phone while they show on the glasses. Pick the voice and speed in Tools › Claude and voice.'),
              step('2', 'Then it is your turn', 'Right after, just keep talking - no button, no “$word”. Say “thanks” or stay quiet to finish.'),
              step('3', 'Interrupt any time', 'Hold the TouchBar or tap the mic to cut in. Double-tap the glasses to stop everything.'),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: 'Say “$word” and then…',
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
