import 'dart:async';

import 'package:flutter/material.dart';

import '../core/log.dart';
import '../core/suite.dart';
import '../features/claude/claude_page.dart';
import '../features/home/home_page.dart';
import '../features/tools/tools_page.dart';
import '../features/work/work_page.dart';
import '../widgets/common.dart';
import '../widgets/voice_widgets.dart';

/// Gives every screen the app's [Suite].
class SuiteScope extends InheritedWidget {
  const SuiteScope({super.key, required this.suite, required super.child});

  final Suite suite;

  static Suite of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<SuiteScope>()!.suite;

  @override
  bool updateShouldNotify(SuiteScope oldWidget) => suite != oldWidget.suite;
}

/// Bottom navigation: Home (talk, your glasses), Claude, Work, More (the
/// terminal, screen sharing and every other tool).
class Shell extends StatefulWidget {
  const Shell({super.key});

  static void goTo(BuildContext context, int tab) =>
      context.findAncestorStateOfType<_ShellState>()?.select(tab);

  static const home = 0, claude = 1, work = 2, more = 3;
  static const names = {'home': home, 'claude': claude, 'work': work, 'more': more, 'tools': more};

  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> with WidgetsBindingObserver {
  int _tab = 0;
  StreamSubscription<String>? _toasts;
  StreamSubscription<String>? _tabs;
  bool _foreground = true;

  void select(int tab) {
    setState(() => _tab = tab);
    if (tab == Shell.work) SuiteScope.of(context).work.markRead();
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _toasts ??= SuiteScope.of(context).toasts.stream.listen((m) {
      if (mounted) showToast(context, m);
    });
    _tabs ??= SuiteScope.of(context).tabRequests.stream.listen((name) {
      final tab = Shell.names[name];
      if (mounted && tab != null) select(tab);
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Screens stop their live updates while the app is in the background.
    EventLog.add('app', 'lifecycle', {'state': state.name});
    final fg = state == AppLifecycleState.resumed;
    if (fg != _foreground) setState(() => _foreground = fg);
    if (fg) SuiteScope.of(context).chat.check();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _toasts?.cancel();
    _tabs?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: Column(children: [
          HandsFreeBanner(
            voice: suite.voice,
            onStop: () {
              suite.voice.stopDictation();
              suite.setHandsFree(false);
            },
          ),
          Expanded(
            child: IndexedStack(index: _tab, children: [
              HomePage(active: _foreground && _tab == Shell.home),
              ClaudePage(active: _foreground && _tab == Shell.claude),
              WorkPage(active: _foreground && _tab == Shell.work),
              const ToolsPage(),
            ]),
          ),
        ]),
      ),
      // Labels grow with the phone's text size only up to 1.2x so they stay on one line.
      bottomNavigationBar: MediaQuery.withClampedTextScaling(
        maxScaleFactor: 1.2,
        child: NavigationBar(
          selectedIndex: _tab,
          onDestinationSelected: select,
          destinations: [
            const NavigationDestination(
                icon: Icon(Icons.visibility_outlined), selectedIcon: Icon(Icons.visibility), label: 'Home'),
            const NavigationDestination(
                icon: Icon(Icons.auto_awesome_outlined), selectedIcon: Icon(Icons.auto_awesome), label: 'Claude'),
            NavigationDestination(
              icon: ListenableBuilder(
                listenable: suite.work,
                builder: (context, _) => Badge(
                  isLabelVisible: suite.work.unread > 0,
                  label: Text('${suite.work.unread}'),
                  child: const Icon(Icons.task_alt_outlined),
                ),
              ),
              selectedIcon: const Icon(Icons.task_alt),
              label: 'Work',
            ),
            const NavigationDestination(
                icon: Icon(Icons.widgets_outlined), selectedIcon: Icon(Icons.widgets), label: 'More'),
          ],
        ),
      ),
    );
  }
}
