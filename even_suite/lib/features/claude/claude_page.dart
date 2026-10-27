import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/bridge/bridge_client.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/bridge/work_controller.dart';
import '../../core/settings.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/voice_widgets.dart';
import 'claude_settings_sheet.dart';

class ClaudePage extends StatefulWidget {
  const ClaudePage({super.key, required this.active});

  /// Visible and in the foreground: keep the conversation live.
  final bool active;

  @override
  State<ClaudePage> createState() => _ClaudePageState();
}

class _ClaudePageState extends State<ClaudePage> {
  late Suite suite;
  final _draft = TextEditingController();
  final _focus = FocusNode();
  bool _watching = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    suite = SuiteScope.of(context);
    suite.voice.removeListener(_syncDraft);
    suite.voice.addListener(_syncDraft);
    _updateWatch();
  }

  @override
  void didUpdateWidget(ClaudePage old) {
    super.didUpdateWidget(old);
    _updateWatch();
  }

  void _updateWatch() {
    if (widget.active && !_watching) {
      suite.chat.watch(this);
      _watching = true;
    } else if (!widget.active && _watching) {
      suite.chat.unwatch(this);
      _watching = false;
    }
  }

  /// Dictated words appear in the message box as you speak.
  void _syncDraft() {
    final v = suite.voice;
    if ((v.dictating || v.draft.isEmpty) && _draft.text != v.draft) {
      _draft.value = TextEditingValue(
        text: v.draft,
        selection: TextSelection.collapsed(offset: v.draft.length),
      );
    }
  }

  @override
  void dispose() {
    suite.voice.removeListener(_syncDraft);
    if (_watching) suite.chat.unwatch(this);
    _draft.dispose();
    _focus.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _draft.text.trim();
    if (text.isEmpty) return;
    _focus.unfocus();
    await suite.voice.send(text);
  }

  Future<void> _talk() async {
    if (suite.narrator.speaking) {
      await suite.stopSpeaking();
      return;
    }
    await suite.toggleTalk();
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      _Header(suite: suite),
      StageStrip(talk: suite.conversation, voice: suite.voice, narrator: suite.narrator),
      Expanded(child: _Console(suite: suite)),
      _VoiceDock(suite: suite, draft: _draft, focus: _focus, onSend: _send, onTalk: _talk),
    ]);
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 0, 4),
      child: ListenableBuilder(
        listenable: suite.settings,
        builder: (context, _) => Row(children: [
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Claude',
                  style: Theme.of(context).textTheme.headlineSmall, maxLines: 1, overflow: TextOverflow.ellipsis),
              Text(
                suite.settings.target == Target.terminal
                    ? 'Typing into ${suite.terminal.agent == 'claude' ? 'Claude Code' : 'the terminal'}'
                    : '${_cap(suite.settings.claudeModel)} · ${effortLabel(suite.settings.claudeEffort)} · '
                        '${accessLabel(suite.settings.claudeAccess)}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
              ),
            ]),
          ),
          SegmentedButton<Target>(
            showSelectedIcon: false,
            style: const ButtonStyle(visualDensity: VisualDensity.compact),
            segments: const [
              ButtonSegment(value: Target.claude, label: Text('Chat')),
              ButtonSegment(value: Target.terminal, label: Text('Term')),
            ],
            selected: {suite.settings.target},
            onSelectionChanged: (s) => suite.settings.set('target', s.first),
          ),
          IconButton(
            tooltip: 'Model, voice and conversation',
            onPressed: () => ClaudeSettingsSheet.show(context),
            icon: const Icon(Icons.tune_rounded),
          ),
          PopupMenuButton<String>(
            tooltip: 'More',
            icon: const Icon(Icons.more_vert_rounded),
            onSelected: (v) async {
              switch (v) {
                case 'stop':
                  await suite.chat.stopReply();
                  await suite.stopSpeaking();
                case 'new':
                  final ok = await showDialog<bool>(
                    context: context,
                    builder: (c) => AlertDialog(
                      title: const Text('Start a new chat?'),
                      content: const Text('Claude forgets this conversation. The web page shows the new chat too.'),
                      actions: [
                        TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep')),
                        FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('New chat')),
                      ],
                    ),
                  );
                  if (ok == true) {
                    try {
                      await suite.chat.newChat();
                    } catch (e) {
                      if (context.mounted) showToast(context, '$e');
                    }
                  }
              }
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'stop', child: Text('Stop the reply')),
              PopupMenuItem(value: 'new', child: Text('New chat')),
            ],
          ),
        ]),
      ),
    );
  }

  static String _cap(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}

/// The conversation as a terminal: your questions, Claude's steps as it
/// works, and the answer typing out live.
class _Console extends StatelessWidget {
  const _Console({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([suite.chat, suite.settings]),
      builder: (context, _) {
        final chat = suite.chat;
        if (suite.settings.target == Target.terminal) {
          return EmptyHint(
            icon: Icons.terminal_rounded,
            title: 'Talking to the terminal',
            lines: [
              'What you say or type is typed into ${suite.terminal.agent == 'claude' ? 'Claude Code' : 'the terminal'} in Termux, then Enter.',
              'Its screen is mirrored to your glasses. Open the Terminal tab to watch it here.',
            ],
          );
        }
        if (chat.messages.isEmpty) {
          return EmptyHint(
            icon: Icons.auto_awesome_rounded,
            title: chat.state == BridgeState.offline ? 'Termux bridge is offline' : 'Talk to Claude',
            lines: chat.state == BridgeState.offline
                ? ['Start Agent Hub from the Glasses tab, then come back.']
                : [
                    'Tap the mic and just talk. It sends when you pause, answers on your glasses and out loud, then listens for a follow-up.',
                    'Or hold the left TouchBar, or turn on hands-free and say “Claude…”.',
                  ],
          );
        }
        final items = chat.messages.reversed.toList();
        return Container(
          margin: const EdgeInsets.fromLTRB(12, 2, 12, 8),
          decoration: BoxDecoration(
            color: const Color(0xFF070B14),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.outline),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: ListView.builder(
              reverse: true,
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
              itemCount: items.length,
              itemBuilder: (context, i) => _Entry(
                message: items[i],
                onShow: () {
                  suite.display.showText(items[i].text, title: items[i].isMine ? 'You' : 'Claude');
                  showToast(context, suite.ble.isConnected ? 'Showing on your glasses' : 'Glasses are not connected');
                },
              ),
            ),
          ),
        );
      },
    );
  }
}

class _Entry extends StatelessWidget {
  const _Entry({required this.message, required this.onShow});
  final BridgeMessage message;
  final VoidCallback onShow;

  static const _mono = TextStyle(fontFamily: 'monospace', fontSize: 13.5, height: 1.45, color: Color(0xFFD7E0EE));

  @override
  Widget build(BuildContext context) {
    final m = message;
    final Widget body;
    if (m.isMine) {
      body = Text.rich(
        TextSpan(children: [
          const TextSpan(text: '❯ ', style: TextStyle(color: AppColors.glasses, fontWeight: FontWeight.w700)),
          TextSpan(text: m.text, style: const TextStyle(color: Color(0xFFBFD3FF))),
        ]),
        style: _mono,
      );
    } else if (m.isPending) {
      body = _Working(message: m);
    } else {
      final who = m.who == 'agy' ? 'Antigravity' : 'Claude';
      body = Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        for (final s in m.steps)
          Text('  ⎿ $s', style: _mono.copyWith(color: AppColors.muted, fontSize: 12.5), maxLines: 1, overflow: TextOverflow.ellipsis),
        Text.rich(
          TextSpan(children: [
            TextSpan(
                text: '● ',
                style: TextStyle(color: m.isError ? AppColors.error : AppColors.claude, fontWeight: FontWeight.w700)),
            TextSpan(text: m.text, style: TextStyle(color: m.isError ? AppColors.error : null)),
          ]),
          style: _mono,
        ),
        if (m.ms > 0 || m.note.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text(
              [
                if (who != 'Claude') who,
                if (m.ms > 0) formatDuration(Duration(milliseconds: m.ms)),
                if (m.steps.isNotEmpty) '${m.steps.length} step${m.steps.length == 1 ? '' : 's'}',
                if (m.note.isNotEmpty) m.note,
              ].join(' · '),
              style: _mono.copyWith(color: AppColors.muted, fontSize: 11.5),
            ),
          ),
      ]);
    }
    return InkWell(
      onTap: m.isPending ? null : onShow,
      onLongPress: () {
        Clipboard.setData(ClipboardData(text: m.text));
        showToast(context, 'Copied');
      },
      child: Padding(padding: const EdgeInsets.symmetric(vertical: 6), child: body),
    );
  }
}

/// A reply on its way: the stage with a spinner and the time, the steps so
/// far, and the text as it is written with a blinking cursor.
class _Working extends StatefulWidget {
  const _Working({required this.message});
  final BridgeMessage message;
  @override
  State<_Working> createState() => _WorkingState();
}

class _WorkingState extends State<_Working> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 900))..repeat();
  Timer? _clock;

  @override
  void initState() {
    super.initState();
    _clock = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _c.dispose();
    _clock?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.message;
    final started = m.started > 0 ? DateTime.fromMillisecondsSinceEpoch((m.started * 1000).round()) : m.time;
    final secs = DateTime.now().difference(started).inSeconds.clamp(0, 99999);
    final stage = switch (m.stage) {
      'queued' => 'Waiting for the last answer',
      'starting' => 'Starting Claude',
      '' => 'Thinking',
      'thinking' => 'Thinking',
      'tool' => m.tool.isEmpty ? 'Working' : m.tool,
      'writing' => 'Writing',
      _ => m.stage,
    };
    return AnimatedBuilder(
      animation: _c,
      builder: (context, _) {
        const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        final spin = frames[(_c.value * frames.length).floor() % frames.length];
        final cursor = _c.value < 0.5 ? '▌' : ' ';
        return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          for (final s in m.steps)
            Text('  ⎿ $s',
                style: _Entry._mono.copyWith(color: AppColors.muted, fontSize: 12.5),
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
          Text('$spin $stage · ${secs}s',
              style: _Entry._mono.copyWith(color: AppColors.claude, fontWeight: FontWeight.w700)),
          if (m.live.isNotEmpty) Text('${m.live}$cursor', style: _Entry._mono),
        ]);
      },
    );
  }
}

class _VoiceDock extends StatelessWidget {
  const _VoiceDock(
      {required this.suite, required this.draft, required this.focus, required this.onSend, required this.onTalk});

  final Suite suite;
  final TextEditingController draft;
  final FocusNode focus;
  final VoidCallback onSend;
  final VoidCallback onTalk;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        border: Border(top: BorderSide(color: AppColors.outline)),
      ),
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 8),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        ListenableBuilder(
          listenable: suite.settings,
          builder: (context, _) => HandsFreeTile(
            voice: suite.voice,
            autoOffMinutes: suite.settings.autoOffMinutes,
            onChanged: suite.setHandsFree,
          ),
        ),
        const SizedBox(height: 8),
        Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Expanded(
            child: TextField(
              controller: draft,
              focusNode: focus,
              minLines: 1,
              maxLines: 4,
              textCapitalization: TextCapitalization.sentences,
              onChanged: suite.voice.setDraft,
              decoration: const InputDecoration(hintText: 'Type, or tap the mic and talk…'),
            ),
          ),
          const SizedBox(width: 8),
          ListenableBuilder(
            listenable: Listenable.merge([draft, suite.chat]),
            builder: (context, _) => SizedBox(
              width: 52,
              height: 52,
              child: IconButton.filled(
                tooltip: 'Send',
                style: IconButton.styleFrom(backgroundColor: AppColors.claude, foregroundColor: const Color(0xFF1F0E08)),
                onPressed: draft.text.trim().isEmpty ? null : onSend,
                icon: const Icon(Icons.arrow_upward_rounded),
              ),
            ),
          ),
        ]),
        const SizedBox(height: 8),
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          _DockButton(icon: Icons.keyboard_arrow_up_rounded, label: 'Up', onTap: suite.display.up),
          TalkButton(
            voice: suite.voice,
            talk: suite.conversation,
            narrator: suite.narrator,
            onTap: onTalk,
          ),
          _DockButton(icon: Icons.keyboard_arrow_down_rounded, label: 'Down', onTap: suite.display.down),
        ]),
        const SizedBox(height: 4),
        ListenableBuilder(
          listenable: suite.settings,
          builder: (context, _) => TalkStatus(
            voice: suite.voice,
            talk: suite.conversation,
            narrator: suite.narrator,
            target: suite.settings.target == Target.terminal ? 'the terminal' : 'Claude',
          ),
        ),
      ]),
    );
  }
}

class _DockButton extends StatelessWidget {
  const _DockButton({required this.icon, required this.label, required this.onTap});
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Column(mainAxisSize: MainAxisSize.min, children: [
      SizedBox(
        width: 52,
        height: 52,
        child: IconButton.filledTonal(
          tooltip: 'Glasses: page ${label.toLowerCase()}',
          onPressed: onTap,
          icon: Icon(icon, size: 30),
        ),
      ),
      const SizedBox(height: 4),
      Text('Glasses ${label.toLowerCase()}', style: const TextStyle(color: AppColors.muted, fontSize: 11.5)),
    ]);
  }
}
