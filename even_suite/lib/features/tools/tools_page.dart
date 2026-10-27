import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../widgets/common.dart';
import '../claude/claude_settings_sheet.dart';
import '../flicker/flicker_page.dart';
import '../lookup/layout_page.dart';
import '../lookup/lookup_page.dart';
import '../screen/screen_page.dart';
import '../settings/settings_page.dart';
import '../settings/speech_page.dart';
import '../settings/touchbar_page.dart';
import '../terminal/terminal_page.dart';
import 'diagnostics_page.dart';
import 'help_page.dart';
import 'images_page.dart';
import 'notify_page.dart';
import 'text_page.dart';

/// Everything else, grouped: your glasses, Claude, labs, the app.
class ToolsPage extends StatelessWidget {
  const ToolsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    void push(Widget page) => Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
    void pushBody(String title, Widget body) => push(Scaffold(
          appBar: AppBar(title: Text(title)),
          body: SafeArea(top: false, child: body),
        ));

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
      children: [
        Text('More', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        const Text('Every tool, in one place', style: TextStyle(color: AppColors.muted)),
        const SizedBox(height: 18),
        _Group('Your glasses', [
          _Tool(Icons.dashboard_customize_rounded, AppColors.hud, 'Look up screen',
              'Your own layout when you look up - drag anything, the time too', () => push(const LookUpPage())),
          _Tool(Icons.touch_app_rounded, AppColors.glasses, 'TouchBar', 'What a hold and a double tap do',
              () => push(const TouchBarPage())),
          _Tool(Icons.view_agenda_rounded, AppColors.claude, 'Glasses layout',
              'Arrange Claude’s working screen; when screens close', () => push(const LayoutPage())),
          _Tool(Icons.notes_rounded, AppColors.ok, 'Send text', 'Type or paste text, page through it on the glasses',
              () => push(const TextPage())),
          _Tool(Icons.notifications_active_rounded, AppColors.warn, 'Notifications',
              'Allow apps and send a notification to the glasses', () => push(const NotifyPage())),
          _Tool(Icons.image_rounded, const Color(0xFFB794F6), 'Images', 'Show a 576×136 picture on the display',
              () => push(const ImagesPage())),
          _Tool(Icons.cast_rounded, AppColors.hud, 'Phone screen', 'Mirror part of your screen to the glasses',
              () => pushBody('Phone screen', const ScreenPage())),
          _Tool(Icons.layers_clear_rounded, AppColors.muted, 'Clear the glasses', 'Back to the glasses dashboard',
              () => suite.display.clear()),
        ]),
        _Group('Claude', [
          _Tool(Icons.terminal_rounded, AppColors.glasses, 'Terminal', 'Claude Code and your Termux shell',
              () => pushBody('Terminal', const TerminalPage(active: true))),
          _Tool(Icons.tune_rounded, AppColors.claude, 'Claude and voice',
              'Model, effort, the speaking voice, follow-ups, trigger words', () => ClaudeSettingsSheet.show(context)),
          _Tool(Icons.graphic_eq_rounded, AppColors.hud, 'Speech engine',
              'Android, or Parakeet on the phone (fast, accurate, offline)', () => push(const SpeechPage())),
        ]),
        _Group('Labs', [
          _Tool(Icons.swap_vert_rounded, AppColors.glasses, 'Flicker (lab)', 'Text at a high and a low spot, switching',
              () => pushBody('Flicker', const FlickerPage())),
        ]),
        _Group('App', [
          _Tool(Icons.record_voice_over_rounded, AppColors.claude, 'How to talk to it',
              'Wake word, “send”, the TouchBar and every voice command', () => push(const HelpPage())),
          _Tool(Icons.settings_rounded, AppColors.glasses, 'Settings', 'Termux bridge, voice, auto-off, glasses',
              () => push(const SettingsPage())),
          _Tool(Icons.insights_rounded, AppColors.hud, 'Diagnostics', 'Timing log of every send, saved in Termux',
              () => push(const DiagnosticsPage())),
        ]),
      ],
    );
  }
}

class _Group extends StatelessWidget {
  const _Group(this.title, this.tools);
  final String title;
  final List<_Tool> tools;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Padding(
            padding: const EdgeInsets.only(left: 4, bottom: 8),
            child: Text(title.toUpperCase(),
                style: const TextStyle(color: AppColors.muted, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.1)),
          ),
          Card(
            clipBehavior: Clip.antiAlias,
            child: Column(children: [
              for (var i = 0; i < tools.length; i++) ...[
                if (i > 0) const Divider(indent: 68),
                tools[i],
              ],
            ]),
          ),
        ]),
      );
}

class _Tool extends StatelessWidget {
  const _Tool(this.icon, this.color, this.title, this.sub, this.onTap);
  final IconData icon;
  final Color color;
  final String title;
  final String sub;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        leading: IconBadge(icon, color: color, size: 38),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
        subtitle: Text(sub, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.muted),
        onTap: onTap,
      );
}
