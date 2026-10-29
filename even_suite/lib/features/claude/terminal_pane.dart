import 'package:flutter/material.dart';

import '../../app/theme.dart';
import '../../core/bridge/bridge_client.dart';
import '../../core/bridge/key_modifiers.dart';
import '../../core/bridge/terminal_controller.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/terminal_view.dart';

/// The terminal you picked, live, as in Termux: its screen in colour, a
/// note when it asks you something, and (when shown) the keys a phone
/// keyboard lacks. Typing and the mic are in the dock under it.
class TerminalPane extends StatelessWidget {
  const TerminalPane({
    super.key,
    required this.suite,
    required this.mods,
    this.keys = false,
    this.accent = AppColors.glasses,
  });

  final Suite suite;

  /// Ctrl, Alt and Shift as armed on the key bar (the dock's text box reads them too).
  final KeyModifiers mods;
  final bool keys;

  /// The colour of the section (Claude or Antigravity) this terminal is in.
  final Color accent;

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
            child: term.denied
                ? const EmptyHint(
                    icon: Icons.lock_outline_rounded,
                    title: 'tmux is not letting the app in',
                    lines: [
                      'In your Termux Claude, type:',
                      '! ~/agent-hub/allow-app --install',
                      'Once is enough - then this shows your sessions.',
                    ],
                  )
                : !term.online && term.text.isEmpty
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
        if (keys) ExtraKeys(term: term, mods: mods, accent: accent, hub: suite.chat.hub),
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
    final asking = term.current?.state == 'asking' || term.lastSend == 'menu';
    final stuck = !asking && term.lastSend == 'stuck';
    final text = asking
        ? '${term.label} is asking you something - answer with 1, 2, 3 or Esc below.'
        : stuck
            ? 'Your message is still in its input box - tap Enter.'
            : null;
    if (text == null) return const SizedBox.shrink();
    final quick = asking
        ? const [('1', '1'), ('2', '2'), ('3', '3'), ('y', 'y'), ('n', 'n'), ('Esc', 'Escape')]
        : const [('Enter', 'Enter')];
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.claude.withOpacity(0.14),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.claude.withOpacity(0.5)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.help_outline_rounded, size: 18, color: AppColors.claude),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600))),
        ]),
        const SizedBox(height: 6),
        Wrap(spacing: 6, runSpacing: 4, children: [
          for (final (label, key) in quick)
            ActionChip(
              visualDensity: VisualDensity.compact,
              label: Text(label, style: const TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w700)),
              onPressed: () => term.key(key),
            ),
        ]),
      ]),
    );
  }
}

/// The extra keys of Termux for a phone keyboard: Esc, Tab, sticky Ctrl, Alt
/// and Shift, Home/End, the arrows, PgUp/PgDn, Del. Tap Ctrl, Alt or Shift and
/// it is armed for the next key (a key here, or a letter from the phone
/// keyboard); tap it again to hold it, a third time to let go. Under them
/// scroll the shortcuts Claude Code and Antigravity use most.
class ExtraKeys extends StatelessWidget {
  const ExtraKeys({super.key, required this.term, required this.mods, this.accent = AppColors.glasses, this.hub});

  final TerminalController term;
  final KeyModifiers mods;
  final Color accent;

  /// What the running Agent Hub can do (null: not checked yet).
  final HubInfo? hub;

  // A (label, tmux key name), or a modifier.
  static const List<Object> _top = [
    ('Esc', 'Escape'),
    ('Tab', 'Tab'),
    Mod.ctrl,
    Mod.alt,
    Mod.shift,
    ('Home', 'Home'),
    ('End', 'End'),
    ('⌫', 'BSpace'),
  ];

  static const List<Object> _bottom = [
    ('←', 'Left'),
    ('↑', 'Up'),
    ('↓', 'Down'),
    ('→', 'Right'),
    ('PgUp', 'PageUp'),
    ('PgDn', 'PageDown'),
    ('Del', 'DC'),
    ('⏎', 'Enter'),
  ];

  /// Every tmux key name the grid can send (checked against Agent Hub's list in the tests).
  static List<String> get gridKeys => [
        for (final k in [..._top, ..._bottom])
          if (k case (String _, String name)) name,
      ];

  /// (label, key, typed as text). Their own modifiers; Ctrl/Alt/Shift do not apply.
  static const shortcuts = [
    ('1', '1', false),
    ('2', '2', false),
    ('3', '3', false),
    ('y', 'y', false),
    ('n', 'n', false),
    ('/', '/', true),
    ('!', '!', true),
    ('@', '@', true),
    ('Ctrl+C', 'C-c', false),
    ('Ctrl+D', 'C-d', false),
    ('Ctrl+L', 'C-l', false),
    ('Ctrl+O', 'C-o', false),
    ('Ctrl+R', 'C-r', false),
    ('Ctrl+T', 'C-t', false),
    ('Shift+Tab', 'BTab', false),
  ];

  void _send(String name) {
    final p = mods.named(name);
    mods.used();
    term.press(p);
  }

  static const _modLabels = {Mod.ctrl: 'Ctrl', Mod.alt: 'Alt', Mod.shift: 'Shift'};

  Widget _row(List<Object> keys) => Row(children: [
        for (final k in keys)
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(2),
              child: switch (k) {
                Mod m => _Key(
                    key: ValueKey('key-${m.name}'),
                    label: _modLabels[m]!,
                    state: mods.state(m),
                    accent: accent,
                    onTap: () => mods.tap(m),
                  ),
                (String label, String name) => _Key(
                    key: ValueKey('key-$name'),
                    label: label,
                    accent: accent,
                    onTap: () => _send(name),
                  ),
                _ => const SizedBox.shrink(),
              },
            ),
          ),
      ]);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(border: Border(top: BorderSide(color: AppColors.outline))),
      padding: const EdgeInsets.fromLTRB(6, 4, 6, 2),
      child: ListenableBuilder(
        listenable: mods,
        builder: (context, _) => Column(mainAxisSize: MainAxisSize.min, children: [
          if (hub != null && !hub!.modKeys && mods.any)
            const Padding(
              padding: EdgeInsets.fromLTRB(4, 0, 4, 2),
              child: Text(
                'Your Agent Hub is older: in Termux type  even restart  so Ctrl, Alt and Shift work with keys.',
                style: TextStyle(color: AppColors.warn, fontSize: 12),
              ),
            ),
          _row(_top),
          _row(_bottom),
          SizedBox(
            height: 44,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.fromLTRB(2, 4, 2, 0),
              children: [
                for (final (label, key, text) in shortcuts)
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: ActionChip(
                      key: ValueKey('short-$key'),
                      visualDensity: VisualDensity.compact,
                      label: Text(label, style: const TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w600)),
                      onPressed: () => term.press(KeyPress(key, text: text)),
                    ),
                  ),
              ],
            ),
          ),
        ]),
      ),
    );
  }
}

/// One key of the grid. A modifier shows whether it is armed for one key
/// (outlined) or held (filled, with a lock).
class _Key extends StatelessWidget {
  const _Key({super.key, required this.label, required this.onTap, required this.accent, this.state = ModState.off});

  final String label;
  final VoidCallback onTap;
  final Color accent;
  final ModState state;

  @override
  Widget build(BuildContext context) {
    final locked = state == ModState.locked;
    final armed = state != ModState.off;
    final fg = locked ? const Color(0xFF0B1220) : (armed ? accent : AppColors.text);
    return Semantics(
      button: true,
      selected: armed,
      label: label,
      child: Material(
        color: locked ? accent : (armed ? accent.withOpacity(0.18) : AppColors.surfaceHigh),
        borderRadius: BorderRadius.circular(9),
        child: InkWell(
          borderRadius: BorderRadius.circular(9),
          onTap: onTap,
          child: Container(
            height: 36,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(9),
              border: Border.all(color: armed ? accent : AppColors.outline),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Text(label,
                      maxLines: 1,
                      style: TextStyle(fontFamily: 'monospace', fontSize: 13, fontWeight: FontWeight.w700, color: fg)),
                ),
              ),
              if (locked) ...[
                const SizedBox(width: 2),
                Icon(Icons.lock_rounded, size: 10, color: fg),
              ],
            ]),
          ),
        ),
      ),
    );
  }
}
