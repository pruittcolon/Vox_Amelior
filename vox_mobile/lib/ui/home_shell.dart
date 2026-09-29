import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/ui/ask_screen.dart';
import 'package:vox_amelior_mobile/ui/automations_screen.dart';
import 'package:vox_amelior_mobile/ui/live_screen.dart';
import 'package:vox_amelior_mobile/ui/people_screen.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/setup_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.services});

  final AppServices services;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;

  void _openSetup() => Navigator.push(
        context,
        MaterialPageRoute<void>(builder: (_) => SetupScreen(services: widget.services)),
      );

  @override
  Widget build(BuildContext context) {
    final s = widget.services;
    final pages = [
      LiveScreen(services: s),
      AskScreen(
        assistant: s.assistant,
        requests: s.requests,
        assistantReady: () => s.gemmaReady,
        onOpenSetup: _openSetup,
      ),
      PeopleScreen(services: s),
      AutomationsScreen(services: s),
      SettingsScreen(services: s),
    ];
    return Scaffold(
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.hearing), label: 'Live'),
          NavigationDestination(icon: Icon(Icons.question_answer), label: 'Ask'),
          NavigationDestination(icon: Icon(Icons.people), label: 'People'),
          NavigationDestination(icon: Icon(Icons.bolt), label: 'Automations'),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }
}
