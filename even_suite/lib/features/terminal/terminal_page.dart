import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../core/suite.dart';
import '../../widgets/session_bar.dart';
import '../claude/terminal_pane.dart';

/// A terminal on its own page (the Claude tab shows the same, next to chat):
/// pick a session, watch it live, type or talk into it.
class TerminalPage extends StatefulWidget {
  const TerminalPage({super.key, required this.active});

  final bool active;

  @override
  State<TerminalPage> createState() => _TerminalPageState();
}

class _TerminalPageState extends State<TerminalPage> {
  late Suite suite;
  final _draft = TextEditingController();
  final _focus = FocusNode();
  bool _watching = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    suite = SuiteScope.of(context);
    _updateWatch();
  }

  @override
  void didUpdateWidget(TerminalPage old) {
    super.didUpdateWidget(old);
    _updateWatch();
  }

  void _updateWatch() {
    if (widget.active && !_watching) {
      suite.terminal.watch(this);
      _watching = true;
    } else if (!widget.active && _watching) {
      suite.terminal.unwatch(this);
      _watching = false;
    }
  }

  @override
  void dispose() {
    if (_watching) suite.terminal.unwatch(this);
    _draft.dispose();
    _focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Column(children: [
        SessionBar(suite: suite, active: widget.active, padding: const EdgeInsets.fromLTRB(12, 8, 12, 6)),
        Expanded(
          child: TerminalPane(
            suite: suite,
            draft: _draft,
            focus: _focus,
            onTalk: () => suite.narrator.speaking ? suite.stopSpeaking() : suite.toggleTalk(),
          ),
        ),
      ]);
}
