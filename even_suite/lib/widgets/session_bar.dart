import 'package:flutter/material.dart';

import '../app/shell.dart';
import '../app/theme.dart';
import '../core/bridge/bridge_client.dart';
import '../core/settings.dart';
import '../core/suite.dart';
import 'common.dart';

/// Where what you say (and type) goes: Claude chat, or one of the terminals
/// in Termux - the Claude Code you use there, and more you open here. Tap
/// one to talk to it, hold one for more, + opens another.
class SessionBar extends StatefulWidget {
  const SessionBar({super.key, required this.suite, required this.active, this.padding = EdgeInsets.zero});

  final Suite suite;

  /// On screen and the app in front: keep the list fresh.
  final bool active;
  final EdgeInsets padding;

  @override
  State<SessionBar> createState() => _SessionBarState();
}

class _SessionBarState extends State<SessionBar> {
  bool _watching = false;

  @override
  void initState() {
    super.initState();
    _update();
  }

  @override
  void didUpdateWidget(SessionBar old) {
    super.didUpdateWidget(old);
    _update();
  }

  void _update() {
    if (widget.active && !_watching) {
      widget.suite.terminal.watchSessions(this);
      _watching = true;
    } else if (!widget.active && _watching) {
      widget.suite.terminal.unwatchSessions(this);
      _watching = false;
    }
  }

  @override
  void dispose() {
    if (_watching) widget.suite.terminal.unwatchSessions(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final suite = widget.suite;
    return ListenableBuilder(
      listenable: Listenable.merge([suite.settings, suite.terminal]),
      builder: (context, _) {
        final term = suite.terminal;
        final chat = suite.settings.target == Target.claude;
        var list = term.sessions;
        // Not listed yet (or an older Agent Hub): at least the one picked.
        if (!list.any((s) => s.name == term.session)) list = [TermSession(term.session), ...list];
        return SizedBox(
          height: 44,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: widget.padding,
            children: [
              _Chip(
                icon: Icons.chat_bubble_rounded,
                label: 'Chat',
                color: AppColors.claude,
                selected: chat,
                onTap: suite.useChat,
              ),
              for (final s in list)
                _Chip(
                  icon: s.isClaude ? Icons.auto_awesome_rounded : Icons.terminal_rounded,
                  label: s.label,
                  color: AppColors.glasses,
                  selected: !chat && s.name == term.session,
                  dot: switch (s.state) {
                    'working' => AppColors.warn,
                    'asking' => AppColors.claude,
                    'idle' => AppColors.ok,
                    _ => null,
                  },
                  pulse: s.state == 'working' || s.state == 'asking',
                  onTap: () => suite.useSession(s.name),
                  onLongPress: () => _details(context, s),
                ),
              _Chip(
                icon: Icons.add_rounded,
                label: 'New',
                color: AppColors.muted,
                selected: false,
                onTap: () => term.sessionsSupported ? _new(context) : _old(context),
              ),
            ],
          ),
        );
      },
    );
  }

  void _old(BuildContext context) => showToast(context, 'Update Agent Hub first: in Termux type  even restart');

  Future<void> _new(BuildContext context) async {
    final suite = widget.suite;
    final pick = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      backgroundColor: AppColors.surface,
      builder: (c) => SafeArea(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          ListTile(
            leading: const Icon(Icons.auto_awesome_rounded, color: AppColors.claude),
            title: const Text('New Claude Code session', style: TextStyle(fontWeight: FontWeight.w700)),
            subtitle: const Text('A fresh conversation, next to the one in Termux'),
            onTap: () => Navigator.pop(c, 'claude'),
          ),
          ListTile(
            leading: const Icon(Icons.history_rounded, color: AppColors.glasses),
            title: const Text('Continue the last conversation', style: TextStyle(fontWeight: FontWeight.w700)),
            subtitle: const Text('Claude Code --continue in a new session'),
            onTap: () => Navigator.pop(c, 'resume'),
          ),
          ListTile(
            leading: const Icon(Icons.terminal_rounded, color: AppColors.muted),
            title: const Text('New shell', style: TextStyle(fontWeight: FontWeight.w700)),
            subtitle: const Text('A plain Termux (Ubuntu) command line'),
            onTap: () => Navigator.pop(c, 'shell'),
          ),
          const SizedBox(height: 8),
        ]),
      ),
    );
    if (pick == null) return;
    try {
      final name = await suite.terminal.openSession(kind: pick == 'shell' ? 'shell' : 'claude', resume: pick == 'resume');
      suite.useSession(name);
      if (context.mounted) {
        Shell.goTo(context, Shell.claude);
        showToast(context, 'Opened ${TermSession.labelFor(name)} - what you say goes there');
      }
    } catch (e) {
      if (context.mounted) showToast(context, 'Could not open it: $e');
    }
  }

  Future<void> _details(BuildContext context, TermSession s) async {
    final suite = widget.suite;
    final pick = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      backgroundColor: AppColors.surface,
      builder: (c) => SafeArea(
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(s.label, style: Theme.of(c).textTheme.titleLarge),
              const SizedBox(height: 2),
              Text(
                [
                  if (s.main) 'The one you use in Termux',
                  if (!s.main && s.attached > 0) 'Open in Termux too',
                  switch (s.state) {
                    'working' => 'Working',
                    'asking' => 'Asking you something',
                    'idle' => 'Waiting for you',
                    _ => '',
                  },
                ].where((x) => x.isNotEmpty).join(' · '),
                style: const TextStyle(color: AppColors.muted),
              ),
              if (s.topic.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text('Last asked: ${s.topic}', maxLines: 3, overflow: TextOverflow.ellipsis),
              ],
            ]),
          ),
          ListTile(
            leading: const Icon(Icons.record_voice_over_rounded, color: AppColors.glasses),
            title: const Text('Talk to this one'),
            onTap: () => Navigator.pop(c, 'use'),
          ),
          ListTile(
            leading: const Icon(Icons.visibility_rounded, color: AppColors.hud),
            title: const Text('Show it on the glasses'),
            onTap: () => Navigator.pop(c, 'mirror'),
          ),
          if (!s.main)
            ListTile(
              leading: const Icon(Icons.close_rounded, color: AppColors.error),
              title: const Text('End this session'),
              subtitle: const Text('Stops what it is doing and closes it'),
              onTap: () => Navigator.pop(c, 'close'),
            ),
          const SizedBox(height: 8),
        ]),
      ),
    );
    if (pick == null || !context.mounted) return;
    switch (pick) {
      case 'use':
        suite.useSession(s.name);
      case 'mirror':
        suite.useSession(s.name);
        if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
        suite.terminal.setMirror(true);
      case 'close':
        final ok = await showDialog<bool>(
          context: context,
          builder: (c) => AlertDialog(
            title: Text('End ${s.label}?'),
            content: const Text('Whatever it is doing stops, and the session closes.'),
            actions: [
              TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep')),
              FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('End it')),
            ],
          ),
        );
        if (ok != true) return;
        final error = await suite.terminal.closeSession(s.name);
        // It was the one you talked to: back to the Termux one.
        if (error == null && suite.terminal.session == s.name) suite.useSession('claude');
        if (context.mounted) showToast(context, error ?? 'Ended ${s.label}');
    }
  }
}

class _Chip extends StatelessWidget {
  const _Chip({
    required this.icon,
    required this.label,
    required this.color,
    required this.selected,
    required this.onTap,
    this.onLongPress,
    this.dot,
    this.pulse = false,
  });

  final IconData icon;
  final String label;
  final Color color;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback? onLongPress;
  final Color? dot;
  final bool pulse;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(right: 8),
        child: Material(
          color: selected ? color.withOpacity(0.18) : AppColors.surfaceHigh,
          shape: StadiumBorder(side: BorderSide(color: selected ? color : AppColors.outline)),
          child: InkWell(
            customBorder: const StadiumBorder(),
            onTap: onTap,
            onLongPress: onLongPress,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(icon, size: 16, color: selected ? color : AppColors.muted),
                const SizedBox(width: 6),
                Text(label,
                    style: TextStyle(
                        fontWeight: FontWeight.w700, fontSize: 13, color: selected ? AppColors.text : AppColors.muted)),
                if (dot != null) ...[const SizedBox(width: 7), StatusPill.dot(dot!, pulse: pulse)],
              ]),
            ),
          ),
        ),
      );
}
