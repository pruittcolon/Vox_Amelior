import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/bridge/terminal_controller.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';

class TerminalPage extends StatefulWidget {
  const TerminalPage({super.key, required this.active});

  final bool active;

  @override
  State<TerminalPage> createState() => _TerminalPageState();
}

class _TerminalPageState extends State<TerminalPage> {
  late Suite suite;
  final _input = TextEditingController();
  final _scroll = ScrollController();
  bool _watching = false;
  bool _follow = true;

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
  void initState() {
    super.initState();
    _scroll.addListener(() {
      if (!_scroll.hasClients) return;
      // The list is reversed: offset 0 is the bottom (newest output).
      _follow = _scroll.offset < 40;
    });
  }

  @override
  void dispose() {
    if (_watching) suite.terminal.unwatch(this);
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final t = _input.text;
    if (t.trim().isEmpty) {
      await suite.terminal.key('Enter');
      return;
    }
    try {
      await suite.terminal.sendLine(t);
      _input.clear();
    } catch (e) {
      if (mounted) showToast(context, '$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final term = suite.terminal;
    return ListenableBuilder(
      listenable: term,
      builder: (context, _) {
        return Column(children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
            child: Row(children: [
              Expanded(
                child: Text('Terminal',
                    style: Theme.of(context).textTheme.headlineSmall, maxLines: 1, overflow: TextOverflow.ellipsis),
              ),
              _AgentPicker(
                value: term.agent,
                onChanged: (a) => suite.settings.set('terminalAgent', a),
              ),
              const SizedBox(width: 4),
              IconButton(
                tooltip: term.mirroring ? 'Stop showing on glasses' : 'Show on glasses',
                isSelected: term.mirroring,
                selectedIcon: const Icon(Icons.visibility_rounded, color: AppColors.hud),
                icon: const Icon(Icons.visibility_outlined),
                onPressed: () => term.setMirror(!term.mirroring),
              ),
            ]),
          ),
          Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: const Color(0xFF070B14),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.outline),
              ),
              child: _screen(term),
            ),
          ),
          _KeyBar(onKey: term.key),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
            child: Row(children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  autocorrect: false,
                  enableSuggestions: false,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _send(),
                  style: const TextStyle(fontFamily: 'monospace'),
                  decoration: InputDecoration(
                    hintText: 'Type a command or message…',
                    prefixIcon: const Icon(Icons.chevron_right_rounded, color: AppColors.glasses),
                    prefixIconConstraints: const BoxConstraints(minWidth: 40),
                    fillColor: AppColors.surface,
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(color: AppColors.outline),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 52,
                height: 52,
                child: IconButton.filled(
                  tooltip: 'Send (Enter)',
                  onPressed: _send,
                  icon: const Icon(Icons.keyboard_return_rounded),
                ),
              ),
            ]),
          ),
        ]);
      },
    );
  }

  Widget _screen(TerminalController term) {
    if (!term.online && term.text.isEmpty) {
      return EmptyHint(
        icon: Icons.terminal_rounded,
        title: term.error == null ? 'Connecting to Termux…' : 'Termux bridge is offline',
        lines: [
          if (term.error != null) term.error!,
          'Start Agent Hub from the Glasses tab (or type `all` in Termux).',
        ],
      );
    }
    final lines = term.text.split('\n');
    if (_follow && _scroll.hasClients && _scroll.offset != 0) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) _scroll.jumpTo(0);
      });
    }
    return Scrollbar(
      controller: _scroll,
      child: ListView.builder(
        controller: _scroll,
        reverse: true,
        padding: const EdgeInsets.all(12),
        itemCount: lines.length,
        itemBuilder: (context, i) => Text(
          lines[lines.length - 1 - i],
          softWrap: true,
          style: const TextStyle(
              fontFamily: 'monospace', fontSize: 12.5, height: 1.35, color: Color(0xFFD7E0EE)),
        ),
      ),
    );
  }
}

class _AgentPicker extends StatelessWidget {
  const _AgentPicker({required this.value, required this.onChanged});
  final String value;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      tooltip: 'Which terminal',
      initialValue: value,
      onSelected: onChanged,
      itemBuilder: (_) => [
        for (final e in TerminalController.agents.entries)
          PopupMenuItem(value: e.key, child: Text(e.value)),
      ],
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: AppColors.surfaceHigh,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.outline),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Text(TerminalController.agents[value] ?? value, style: const TextStyle(fontWeight: FontWeight.w600)),
          const Icon(Icons.expand_more_rounded, size: 20),
        ]),
      ),
    );
  }
}

/// Keys Claude Code needs that a phone keyboard lacks.
class _KeyBar extends StatelessWidget {
  const _KeyBar({required this.onKey});
  final Future<void> Function(String key) onKey;

  static const keys = [
    ('Esc', 'Escape'),
    ('Enter', 'Enter'),
    ('Up', 'Up'),
    ('Down', 'Down'),
    ('Tab', 'Tab'),
    ('Shift-Tab', 'BTab'),
    ('Ctrl-C', 'C-c'),
    ('1', '1'),
    ('2', '2'),
    ('3', '3'),
    ('y', 'y'),
    ('n', 'n'),
  ];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 52,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
        children: [
          for (final (label, key) in keys)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: ActionChip(
                label: Text(label, style: const TextStyle(fontFamily: 'monospace')),
                onPressed: () => onKey(key),
              ),
            ),
        ],
      ),
    );
  }
}
