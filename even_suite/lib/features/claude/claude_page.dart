import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/bridge/bridge_client.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/settings.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/voice_widgets.dart';

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

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      _Header(suite: suite),
      Expanded(child: _Messages(suite: suite)),
      _VoiceDock(suite: suite, draft: _draft, focus: _focus, onSend: _send),
    ]);
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 4, 4),
      child: ListenableBuilder(
        listenable: suite.settings,
        builder: (context, _) => Row(children: [
          Expanded(
            child: Text('Claude',
                style: Theme.of(context).textTheme.headlineSmall, maxLines: 1, overflow: TextOverflow.ellipsis),
          ),
          SegmentedButton<Target>(
            showSelectedIcon: false,
            segments: const [
              ButtonSegment(value: Target.claude, label: Text('Chat')),
              ButtonSegment(value: Target.terminal, label: Text('Terminal')),
            ],
            selected: {suite.settings.target},
            onSelectionChanged: (s) => suite.settings.set('target', s.first),
          ),
          PopupMenuButton<String>(
            tooltip: 'More',
            icon: const Icon(Icons.more_vert_rounded),
            onSelected: (v) async {
              switch (v) {
                case 'stop':
                  await suite.chat.stopReply();
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
}

class _Messages extends StatelessWidget {
  const _Messages({required this.suite});
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
            title: chat.state == BridgeState.offline ? 'Termux bridge is offline' : 'Ask Claude anything',
            lines: chat.state == BridgeState.offline
                ? ['Start Agent Hub from the Glasses tab, then come back.']
                : [
                    'Tap the mic and speak, or type below.',
                    'Hands-free: turn it on, say “Claude”, your question, then “send”.',
                    'Replies show on your glasses. Tap the TouchBar (or ▲ ▼) to page.',
                  ],
          );
        }
        final items = chat.messages.reversed.toList();
        return ListView.builder(
          reverse: true,
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
          itemCount: items.length,
          itemBuilder: (context, i) => _Bubble(
            message: items[i],
            onShow: () {
              suite.display.showText(items[i].text, title: items[i].isMine ? 'You' : 'Claude');
              showToast(context, suite.ble.isConnected ? 'Showing on your glasses' : 'Glasses are not connected');
            },
          ),
        );
      },
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.message, required this.onShow});
  final BridgeMessage message;
  final VoidCallback onShow;

  @override
  Widget build(BuildContext context) {
    final mine = message.isMine;
    final color = mine
        ? AppColors.glasses.withOpacity(0.16)
        : (message.isError ? AppColors.error.withOpacity(0.12) : AppColors.surface);
    final border = mine ? AppColors.glasses.withOpacity(0.3) : AppColors.outline;
    final who = switch (message.who) { 'you' => 'You', 'agy' => 'Antigravity', _ => 'Claude' };
    return Align(
      alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.86),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 5),
          child: Material(
            color: color,
            shape: RoundedRectangleBorder(
              side: BorderSide(color: border),
              borderRadius: BorderRadius.only(
                topLeft: const Radius.circular(18),
                topRight: const Radius.circular(18),
                bottomLeft: Radius.circular(mine ? 18 : 6),
                bottomRight: Radius.circular(mine ? 6 : 18),
              ),
            ),
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: message.isPending ? null : onShow,
              onLongPress: () {
                Clipboard.setData(ClipboardData(text: message.text));
                showToast(context, 'Copied');
              },
              child: Padding(
                padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  if (!mine)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(Icons.auto_awesome_rounded,
                            size: 14, color: message.isError ? AppColors.error : AppColors.claude),
                        const SizedBox(width: 6),
                        Text(who,
                            style: TextStyle(
                                fontSize: 12.5,
                                fontWeight: FontWeight.w700,
                                color: message.isError ? AppColors.error : AppColors.claude)),
                      ]),
                    ),
                  if (message.isPending)
                    const _Thinking()
                  else
                    SelectableText(message.text, style: const TextStyle(fontSize: 15, height: 1.42)),
                ]),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _Thinking extends StatefulWidget {
  const _Thinking();
  @override
  State<_Thinking> createState() => _ThinkingState();
}

class _ThinkingState extends State<_Thinking> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (context, _) => Row(mainAxisSize: MainAxisSize.min, children: [
        const Text('Thinking', style: TextStyle(color: AppColors.muted)),
        const SizedBox(width: 6),
        for (var i = 0; i < 3; i++)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 2),
            child: Opacity(
              opacity: ((_c.value * 3 - i) % 3) < 1 ? 1 : 0.3,
              child: Container(
                width: 6,
                height: 6,
                decoration: const BoxDecoration(color: AppColors.claude, shape: BoxShape.circle),
              ),
            ),
          ),
      ]),
    );
  }
}

class _VoiceDock extends StatelessWidget {
  const _VoiceDock({required this.suite, required this.draft, required this.focus, required this.onSend});

  final Suite suite;
  final TextEditingController draft;
  final FocusNode focus;
  final VoidCallback onSend;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        border: Border(top: BorderSide(color: AppColors.outline)),
      ),
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        ListenableBuilder(
          listenable: suite.settings,
          builder: (context, _) => HandsFreeTile(
            voice: suite.voice,
            autoOffMinutes: suite.settings.autoOffMinutes,
            onChanged: suite.setHandsFree,
          ),
        ),
        const SizedBox(height: 10),
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
        const SizedBox(height: 12),
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          _DockButton(icon: Icons.keyboard_arrow_up_rounded, label: 'Up', onTap: suite.display.up),
          MicButton(voice: suite.voice, onTap: suite.toggleTalk),
          _DockButton(icon: Icons.keyboard_arrow_down_rounded, label: 'Down', onTap: suite.display.down),
        ]),
        const SizedBox(height: 8),
        ListenableBuilder(
          listenable: suite.settings,
          builder: (context, _) => VoiceStatus(
            voice: suite.voice,
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
        width: 56,
        height: 56,
        child: IconButton.filledTonal(
          tooltip: 'Glasses: page ${label.toLowerCase()}',
          onPressed: onTap,
          icon: Icon(icon, size: 32),
        ),
      ),
      const SizedBox(height: 4),
      Text('Glasses ${label.toLowerCase()}',
          style: const TextStyle(color: AppColors.muted, fontSize: 11.5)),
    ]);
  }
}
