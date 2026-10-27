import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../widgets/common.dart';
import '../claude/claude_settings_sheet.dart';
import '../flicker/flicker_page.dart';
import '../lookup/lookup_page.dart';
import '../settings/settings_page.dart';
import 'diagnostics_page.dart';
import 'help_page.dart';
import 'images_page.dart';
import 'notify_page.dart';
import 'text_page.dart';

class ToolsPage extends StatelessWidget {
  const ToolsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    void push(Widget page) => Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
    Widget tile(IconData icon, Color color, String title, String sub, VoidCallback onTap) => Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Card(
            clipBehavior: Clip.antiAlias,
            child: ListTile(
              contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              leading: IconBadge(icon, color: color),
              title: Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
              subtitle: Text(sub, style: const TextStyle(color: AppColors.muted)),
              trailing: const Icon(Icons.chevron_right_rounded),
              onTap: onTap,
            ),
          ),
        );
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      children: [
        Text('Tools', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        const Text('Everything from the Even Realities demo app',
            style: TextStyle(color: AppColors.muted)),
        const SizedBox(height: 16),
        tile(Icons.dashboard_customize_rounded, AppColors.hud, 'Look up screen',
            'Clock, weather, notes and phone notifications when you look up', () => push(const LookUpPage())),
        tile(Icons.tune_rounded, AppColors.claude, 'Claude and voice',
            'Model, effort, the speaking voice, follow-ups, trigger words', () => ClaudeSettingsSheet.show(context)),
        tile(Icons.notes_rounded, AppColors.ok, 'Send text', 'Type or paste text, page through it on the glasses',
            () => push(const TextPage())),
        tile(Icons.notifications_active_rounded, AppColors.warn, 'Notifications',
            'Allow apps and send a notification to the glasses', () => push(const NotifyPage())),
        tile(Icons.image_rounded, const Color(0xFFB794F6), 'Images', 'Show a 576×136 picture on the display',
            () => push(const ImagesPage())),
        tile(Icons.layers_clear_rounded, AppColors.muted, 'Clear the glasses', 'Go back to the glasses dashboard',
            () => suite.display.clear()),
        tile(Icons.swap_vert_rounded, AppColors.glasses, 'Flicker (lab)', 'Text at a high and a low spot, switching',
            () => push(Scaffold(appBar: AppBar(), body: const SafeArea(top: false, child: FlickerPage())))),
        const SizedBox(height: 10),
        Text('App', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        tile(Icons.record_voice_over_rounded, AppColors.claude, 'How to talk to it',
            'Wake word, “send”, scrolling and every voice command', () => push(const HelpPage())),
        tile(Icons.settings_rounded, AppColors.glasses, 'Settings', 'Termux bridge, voice, auto-off, glasses',
            () => push(const SettingsPage())),
        tile(Icons.insights_rounded, AppColors.hud, 'Diagnostics', 'Timing log of every send, saved in Termux',
            () => push(const DiagnosticsPage())),
      ],
    );
  }
}
