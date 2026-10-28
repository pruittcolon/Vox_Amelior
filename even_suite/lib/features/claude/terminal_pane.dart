import 'package:flutter/material.dart';

import '../../app/theme.dart';
import '../../core/bridge/terminal_controller.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/terminal_view.dart';

/// The terminal you picked, live, as in Termux: its screen in colour, a
/// note when it asks you something, and (when shown) the keys a phone
/// keyboard lacks. Typing and the mic are in the dock under it.
class TerminalPane extends StatelessWidget {
  const TerminalPane({super.key, required this.suite, this.keys = false});

  final Suite suite;
  final bool keys;

  @override
  Widget build(BuildContext context) {
    final term = suite.terminal;
    return ListenableBuilder(
      listenable: term,
      builder: (context, _) => Column(children: [
        Expanded(
          child: Container(
            margin: const EdgeInsets.fromLTRB(8, 0, 8, 6),
            decoration: BoxDecoration(
              color: TerminalScreen.background,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.outline),
            ),
            clipBehavior: Clip.antiAlias,
            child: !term.online && term.text.isEmpty
                ? EmptyHint(
                    icon: Icons.terminal_rounded,
                    title: term.error == null ? 'Connecting to Termux…' : 'Termux bridge is offline',
                    lines: [
                      if (term.error != null) term.error!,
                      'Start it on the Home tab, or type even in Termux.',
                    ],
                  )
                : TerminalScreen(text: term.text, cols: term.cols, onFit: term.fit),
          ),
        ),
        _Notice(term: term),
        if (keys) _KeyBar(term: term),
      ]),
    );
  }
}

/// Claude Code asking a question, or a message that did not go in.
class _Notice extends StatelessWidget {
  const _Notice({required this.term});
  final TerminalController term;

  @override
  Widget build(BuildContext context) {
    final asking = term.current?.state == 'asking';
    final text = asking || term.lastSend == 'menu'
        ? '${term.label} is asking you something - answer with 1, 2, 3 or Esc below.'
        : term.lastSend == 'stuck'
            ? 'Your message is still in its input box - tap Enter.'
            : null;
    if (text == null) return const SizedBox.shrink();
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.claude.withOpacity(0.14),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.claude.withOpacity(0.5)),
      ),
      child: Row(children: [
        const Icon(Icons.help_outline_rounded, size: 18, color: AppColors.claude),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600))),
      ]),
    );
  }
}

/// Keys Claude Code needs that a phone keyboard lacks. A few type text
/// instead ("/" opens Claude Code's commands).
class _KeyBar extends StatelessWidget {
  const _KeyBar({required this.term});
  final TerminalController term;

  static const keys = [
    ('Esc', 'Escape', false),
    ('Enter', 'Enter', false),
    ('↑', 'Up', false),
    ('↓', 'Down', false),
    ('←', 'Left', false),
    ('→', 'Right', false),
    ('Tab', 'Tab', false),
    ('Shift+Tab', 'BTab', false),
    ('Ctrl+C', 'C-c', false),
    ('1', '1', false),
    ('2', '2', false),
    ('3', '3', false),
    ('y', 'y', false),
    ('n', 'n', false),
    ('/', '/', true),
    ('!', '!', true),
    ('@', '@', true),
    ('⌫', 'BSpace', false),
    ('Ctrl+O', 'C-o', false),
    ('Ctrl+R', 'C-r', false),
    ('Ctrl+T', 'C-t', false),
    ('PgUp', 'PageUp', false),
    ('PgDn', 'PageDown', false),
  ];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.fromLTRB(12, 2, 12, 4),
        children: [
          for (final (label, key, text) in keys)
            Padding(
              padding: const EdgeInsets.only(right: 6),
              child: ActionChip(
                visualDensity: VisualDensity.compact,
                label: Text(label, style: const TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w600)),
                onPressed: () => text ? term.sendLine(key, enter: false).catchError((Object _) => '') : term.key(key),
              ),
            ),
        ],
      ),
    );
  }
}
