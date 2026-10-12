import 'dart:async';

import 'package:flutter/material.dart';

import '../core/log.dart';
import '../core/suite.dart';
import '../features/claude/claude_page.dart';
import '../features/flicker/flicker_page.dart';
import '../features/home/home_page.dart';
import '../features/screen/screen_page.dart';
import '../features/terminal/terminal_page.dart';
import '../features/tools/tools_page.dart';
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

/// Bottom navigation: Home (glasses), Claude, Terminal, Screen, Flicker, Tools.
class Shell extends StatefulWidget {
  const Shell({super.key});

  static void goTo(BuildContext context, int tab) =>
      context.findAncestorStateOfType<_ShellState>()?.select(tab);

  static const home = 0, claude = 1, terminal = 2, screen = 3, flicker = 4, tools = 5;

  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> with WidgetsBindingObserver {
  int _tab = 0;
  StreamSubscription<String>? _toasts;
  bool _foreground = true;

  void select(int tab) => setState(() => _tab = tab);

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
              const HomePage(),
              ClaudePage(active: _foreground && _tab == Shell.claude),
              TerminalPage(active: _foreground && _tab == Shell.terminal),
              const ScreenPage(),
              const FlickerPage(),
              const ToolsPage(),
            ]),
          ),
        ]),
      ),
      // Six tabs share 360 dp: labels grow with the phone's text size only up
      // to 1.1x so they stay on one line.
      bottomNavigationBar: MediaQuery.withClampedTextScaling(
        maxScaleFactor: 1.1,
        child: NavigationBar(
          selectedIndex: _tab,
          onDestinationSelected: select,
          destinations: const [
            NavigationDestination(icon: Icon(Icons.visibility_outlined), selectedIcon: Icon(Icons.visibility), label: 'Glasses'),
            NavigationDestination(icon: Icon(Icons.auto_awesome_outlined), selectedIcon: Icon(Icons.auto_awesome), label: 'Claude'),
            NavigationDestination(icon: Icon(Icons.terminal_outlined), selectedIcon: Icon(Icons.terminal), label: 'Terminal'),
            NavigationDestination(icon: Icon(Icons.cast_outlined), selectedIcon: Icon(Icons.cast), label: 'Screen'),
            NavigationDestination(icon: Icon(Icons.swap_vert_outlined), selectedIcon: Icon(Icons.swap_vert), label: 'Flicker'),
            NavigationDestination(icon: Icon(Icons.widgets_outlined), selectedIcon: Icon(Icons.widgets), label: 'Tools'),
          ],
        ),
      ),
    );
  }
}
