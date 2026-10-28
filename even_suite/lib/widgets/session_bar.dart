import 'package:flutter/material.dart';

import '../app/shell.dart';
import '../app/theme.dart';
import '../core/bridge/bridge_client.dart';
import '../core/settings.dart';
import '../core/suite.dart';
import 'common.dart';

/// Where what you say (and type) goes: Claude chat, or one of your Claude
/// Code sessions in Termux. Tap one to talk to it, hold it for more;
/// "Sessions" lists everything open (shells too), opens new ones and picks
/// up past conversations.
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
        // Claude Code sessions here; shells and the rest under "Sessions".
        final list = [
          ...term.claudes,
          if (!chat && term.current != null && !term.current!.isClaude) term.current!,
        ];
        return SizedBox(
          height: 40,
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
                  icon: s.isClaude ? Icons.terminal_rounded : Icons.code_rounded,
                  label: s.label,
                  color: AppColors.glasses,
                  selected: !chat && s.name == term.session,
                  dot: _stateColor(s.state),
                  pulse: s.state == 'working' || s.state == 'asking',
                  onTap: () => suite.useSession(s.name),
                  onLongPress: () => SessionsSheet.details(context, suite, s),
                ),
              _Chip(
                icon: Icons.view_list_rounded,
                label: 'Sessions',
                color: AppColors.muted,
                selected: false,
                onTap: () => SessionsSheet.show(context, suite),
              ),
            ],
          ),
        );
      },
    );
  }
}

Color? _stateColor(String state) => switch (state) {
      'working' => AppColors.warn,
      'asking' => AppColors.claude,
      'idle' => AppColors.ok,
      _ => null,
    };

/// Every session in Termux, new ones, and past conversations to pick up.
class SessionsSheet extends StatefulWidget {
  const SessionsSheet({super.key, required this.suite, required this.goToClaude});
  final Suite suite;

  /// Switch to the Claude tab (the sheet sits above the tabs, so it is
  /// handed in from the screen that opened it).
  final VoidCallback goToClaude;

  static Future<void> show(BuildContext context, Suite suite) => showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        backgroundColor: AppColors.surface,
        builder: (_) => SessionsSheet(
          suite: suite,
          goToClaude: () {
            if (context.mounted) Shell.goTo(context, Shell.claude);
          },
        ),
      );

  /// One session: talk to it, show it on the glasses, end it.
  static Future<void> details(BuildContext context, Suite suite, TermSession s) async {
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
              Text(_describe(s), style: const TextStyle(color: AppColors.muted)),
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
        if (context.mounted) showToast(context, error ?? 'Ended ${s.label}');
    }
  }

  static String _describe(TermSession s) => [
        if (s.main) 'The one Termux shows',
        if (!s.main && s.attached > 0) 'Open in Termux too',
        switch (s.state) {
          'working' => 'Working',
          'asking' => 'Asking you something',
          'idle' => 'Waiting for you',
          _ => s.isClaude ? '' : 'A command line',
        },
      ].where((x) => x.isNotEmpty).join(' · ');

  @override
  State<SessionsSheet> createState() => _SessionsSheetState();
}

class _SessionsSheetState extends State<SessionsSheet> {
  late Future<List<PastConversation>> _past = _loadPast();

  Future<List<PastConversation>> _loadPast() async {
    try {
      return await widget.suite.client.conversations();
    } catch (_) {
      return const [];
    }
  }

  Future<void> _open({String kind = 'claude', String resumeId = ''}) async {
    final suite = widget.suite;
    final nav = Navigator.of(context);
    try {
      final name = await suite.terminal.openSession(kind: kind, resumeId: resumeId);
      suite.useSession(name);
      if (mounted) showToast(context, 'Opened ${TermSession.labelFor(name)} - what you say goes there');
      nav.pop();
      widget.goToClaude();
    } catch (e) {
      if (mounted) showToast(context, 'Could not open it: $e');
    }
  }

  static String _ago(DateTime t) {
    final d = DateTime.now().difference(t);
    if (d.inMinutes < 1) return 'just now';
    if (d.inHours < 1) return '${d.inMinutes} min ago';
    if (d.inDays < 1) return '${d.inHours} h ago';
    return '${d.inDays} d ago';
  }

  @override
  Widget build(BuildContext context) {
    final suite = widget.suite;
    return ListenableBuilder(
      listenable: suite.terminal,
      builder: (context, _) {
        final term = suite.terminal;
        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.75,
          maxChildSize: 0.95,
          builder: (context, scroll) => ListView(
            controller: scroll,
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
            children: [
              Text('Sessions', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 4),
              const Text('Tap one to talk to it - by voice, the TouchBar or typing.',
                  style: TextStyle(color: AppColors.muted)),
              if (!term.sessionsSupported) ...[
                const SizedBox(height: 12),
                const _Note('Agent Hub needs the update: in Termux, type  even restart'),
              ] else if (term.denied) ...[
                const SizedBox(height: 12),
                const _Note('tmux is not letting the app in yet. In your Termux Claude, type:\n'
                    '! ~/agent-hub/allow-app --install'),
              ] else if (!term.sessions.any((s) => s.main)) ...[
                const SizedBox(height: 12),
                const _Note('Your Termux Claude is not open. Type  claude  in Termux, or open one here.'),
              ],
              const SizedBox(height: 14),
              const _Heading('Open now'),
              if (term.sessions.isEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: Text('None yet.', style: TextStyle(color: AppColors.muted)),
                ),
              for (final s in term.sessions)
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(s.isClaude ? Icons.terminal_rounded : Icons.code_rounded,
                      color: s.name == term.session && suite.settings.target == Target.terminal
                          ? AppColors.glasses
                          : AppColors.muted),
                  title: Row(children: [
                    Flexible(child: Text(s.label, style: const TextStyle(fontWeight: FontWeight.w700))),
                    if (_stateColor(s.state) != null) ...[
                      const SizedBox(width: 8),
                      StatusPill.dot(_stateColor(s.state)!, pulse: s.state == 'working' || s.state == 'asking'),
                    ],
                  ]),
                  subtitle: Text(
                    [SessionsSheet._describe(s), if (s.topic.isNotEmpty) s.topic].where((x) => x.isNotEmpty).join('\n'),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                  ),
                  trailing: s.main
                      ? null
                      : IconButton(
                          tooltip: 'End it',
                          icon: const Icon(Icons.close_rounded),
                          onPressed: () => SessionsSheet.details(context, suite, s),
                        ),
                  onTap: () {
                    suite.useSession(s.name);
                    Navigator.pop(context);
                    widget.goToClaude();
                  },
                ),
              const SizedBox(height: 10),
              Row(children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: term.sessionsSupported ? () => _open() : null,
                    icon: const Icon(Icons.add_rounded),
                    label: const Text('New Claude Code'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: term.sessionsSupported ? () => _open(kind: 'shell') : null,
                  icon: const Icon(Icons.code_rounded),
                  label: const Text('Shell'),
                ),
              ]),
              const SizedBox(height: 20),
              const _Heading('Pick up a past conversation'),
              FutureBuilder<List<PastConversation>>(
                future: _past,
                builder: (context, snap) {
                  if (snap.connectionState != ConnectionState.done) {
                    return const Padding(
                      padding: EdgeInsets.all(12),
                      child: Center(child: CircularProgressIndicator()),
                    );
                  }
                  final past = snap.data ?? const [];
                  if (past.isEmpty) {
                    return const Padding(
                      padding: EdgeInsets.symmetric(vertical: 8),
                      child: Text('None found.', style: TextStyle(color: AppColors.muted)),
                    );
                  }
                  return Column(children: [
                    for (final c in past)
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: const Icon(Icons.history_rounded, color: AppColors.muted),
                        title: Text(c.title, maxLines: 2, overflow: TextOverflow.ellipsis),
                        subtitle: Text(
                          [_ago(c.time), if (c.cwd.isNotEmpty) c.cwd, if (c.active) 'open somewhere now'].join(' · '),
                          style: TextStyle(color: c.active ? AppColors.warn : AppColors.muted, fontSize: 12),
                        ),
                        onTap: term.sessionsSupported ? () => _open(resumeId: c.id) : null,
                      ),
                  ]);
                },
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Heading extends StatelessWidget {
  const _Heading(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 4),
        child: Text(text.toUpperCase(),
            style: const TextStyle(color: AppColors.muted, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.1)),
      );
}

class _Note extends StatelessWidget {
  const _Note(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppColors.warn.withOpacity(0.12),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.warn.withOpacity(0.4)),
        ),
        child: Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
      );
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
