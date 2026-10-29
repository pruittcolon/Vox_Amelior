import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/bridge/bridge_client.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/bridge/key_modifiers.dart';
import '../../core/bridge/work_controller.dart';
import '../../core/phone_actions.dart';
import '../../core/settings.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';
import '../../widgets/session_bar.dart';
import '../../widgets/voice_widgets.dart';
import 'agent_sections.dart';
import 'claude_settings_sheet.dart';
import 'terminal_pane.dart';

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
  final _mods = KeyModifiers();
  bool _watching = false;
  bool _watchingTerm = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    suite = SuiteScope.of(context);
    suite.voice.removeListener(_syncDraft);
    suite.voice.addListener(_syncDraft);
    suite.settings.removeListener(_updateWatch);
    suite.settings.addListener(_updateWatch);
    _updateWatch();
  }

  @override
  void didUpdateWidget(ClaudePage old) {
    super.didUpdateWidget(old);
    _updateWatch();
  }

  /// Chat stays live while the tab shows; the terminal while you watch it.
  void _updateWatch() {
    if (widget.active && !_watching) {
      suite.chat.watch(this);
      _watching = true;
    } else if (!widget.active && _watching) {
      suite.chat.unwatch(this);
      _watching = false;
    }
    final term = widget.active && suite.settings.target == Target.terminal;
    if (term && !_watchingTerm) {
      suite.terminal.watch(this);
      _watchingTerm = true;
    } else if (!term && _watchingTerm) {
      suite.terminal.unwatch(this);
      _watchingTerm = false;
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
    suite.settings.removeListener(_updateWatch);
    if (_watching) suite.chat.unwatch(this);
    if (_watchingTerm) suite.terminal.unwatch(this);
    _draft.dispose();
    _focus.dispose();
    _mods.dispose();
    super.dispose();
  }

  /// Chat: sent as a message. Terminal: typed in word for word, then Enter
  /// (an empty box just presses Enter).
  Future<void> _send() async {
    final text = _draft.text;
    if (suite.settings.target == Target.terminal) {
      _draft.clear();
      suite.voice.setDraft('');
      if (text.trim().isEmpty) return suite.terminal.key('Enter');
      return suite.typeIntoTerminal(text);
    }
    if (text.trim().isEmpty) return;
    _focus.unfocus();
    await suite.voice.send(text.trim());
  }

  Future<void> _talk() async {
    if (suite.narrator.speaking) {
      await suite.stopSpeaking();
      return;
    }
    await suite.toggleTalk();
  }

  /// The keys a phone keyboard lacks (Esc, arrows, Tab ...), under the terminal.
  bool _keys = false;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: suite.settings,
      builder: (context, _) {
        final terminal = suite.settings.target == Target.terminal;
        final agent = suite.settings.agent == 'agy' ? 'agy' : 'claude';
        // While the phone keyboard is up the screen is short: the terminal gets the room.
        final typing = MediaQuery.viewInsetsOf(context).bottom > 0;
        return Column(children: [
          _Header(suite: suite),
          if (!typing) AgentSections(suite: suite),
          SessionBar(
              suite: suite, active: widget.active, agent: agent, padding: const EdgeInsets.fromLTRB(12, 2, 12, 6)),
          if (terminal)
            Expanded(
                child: TerminalPane(
                    suite: suite, mods: _mods, keys: _keys, accent: agent == 'agy' ? AppColors.agy : AppColors.glasses))
          else ...[
            StageStrip(talk: suite.conversation, voice: suite.voice, narrator: suite.narrator),
            Expanded(child: _Console(suite: suite)),
          ],
          _Dock(
            suite: suite,
            draft: _draft,
            focus: _focus,
            terminal: terminal,
            keys: _keys,
            mods: _mods,
            onKeys: () {
              // Hiding the keys lets go of a held Ctrl, Alt or Shift (it would catch letters unseen).
              if (_keys) _mods.clear();
              setState(() => _keys = !_keys);
            },
            onSend: _send,
            onTalk: _talk,
          ),
        ]);
      },
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 6, 12, 6),
      child: ListenableBuilder(
        listenable: Listenable.merge([suite.settings, suite.terminal]),
        builder: (context, _) {
          final terminal = suite.settings.target == Target.terminal;
          final term = suite.terminal;
          final state = switch (term.current?.state) {
            'working' => ' · working',
            'asking' => ' · asking you',
            'idle' => ' · ready',
            _ => '',
          };
          final agy = suite.settings.agent == 'agy';
          return Row(children: [
            Text('Terminal',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                terminal
                    ? '${term.label}$state'
                    : agy
                        ? 'Chat · ${effortLabel(suite.settings.claudeEffort)} · ${accessLabel(suite.settings.claudeAccess)}'
                        : 'Chat · ${_cap(suite.settings.claudeModel)} · ${effortLabel(suite.settings.claudeEffort)}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
              ),
            ),
            GlassesBattery(ble: suite.ble, onTap: () => suite.askBattery(now: true)),
          ]);
        },
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
      listenable: suite.chat,
      builder: (context, _) {
        final chat = suite.chat;
        final agent = chat.agent;
        // Each section keeps its own conversation: Claude's here, Antigravity's there.
        final mine = [for (final m in chat.messages) if (m.belongsTo(agent)) m];
        if (mine.isEmpty) {
          return EmptyHint(
            icon: AgentSections.iconOf(agent),
            title: chat.state == BridgeState.offline ? 'Termux bridge is offline' : 'Talk to ${chat.agentName}',
            lines: chat.state == BridgeState.offline
                ? ['Start Agent Hub from the Glasses tab, then come back.']
                : [
                    'Tap the mic and just talk. It sends when you pause and answers on your glasses and out loud. For a back and forth, pick it in ⋯.',
                    'Or hold the left TouchBar, or turn on hands-free and say “${suite.voice.wake.label}…”.',
                  ],
          );
        }
        final items = mine.reversed.toList();
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
                  suite.display.showText(PhoneAction.strip(items[i].text),
                      title: items[i].isMine ? 'You' : ChatController.agentLabel(agent));
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
          const TextSpan(text: '> ', style: TextStyle(color: AppColors.glasses, fontWeight: FontWeight.w700)),
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
          Text('   · $s', style: _mono.copyWith(color: AppColors.muted, fontSize: 12.5), maxLines: 1, overflow: TextOverflow.ellipsis),
        Text.rich(
          TextSpan(children: [
            TextSpan(
                text: '• ',
                style: TextStyle(
                    color: m.isError ? AppColors.error : AgentSections.colorOf(m.who),
                    fontWeight: FontWeight.w700)),
            TextSpan(text: PhoneAction.strip(m.text), style: TextStyle(color: m.isError ? AppColors.error : null)),
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
        // Plain characters every phone font has.
        const frames = ['|', '/', '-', r'\'];
        final spin = frames[(_c.value * frames.length).floor() % frames.length];
        final cursor = _c.value < 0.5 ? '_' : ' ';
        return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          for (final s in m.steps)
            Text('   · $s',
                style: _Entry._mono.copyWith(color: AppColors.muted, fontSize: 12.5),
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
          Text('$spin $stage · ${secs}s',
              style: _Entry._mono.copyWith(color: AppColors.claude, fontWeight: FontWeight.w700)),
          if (m.live.isNotEmpty) Text('${PhoneAction.strip(m.live)}$cursor', style: _Entry._mono),
        ]);
      },
    );
  }
}

/// One slim row under the chat or terminal: more controls, the keys
/// (terminal), what to send, and the mic.
class _Dock extends StatelessWidget {
  const _Dock({
    required this.suite,
    required this.draft,
    required this.focus,
    required this.terminal,
    required this.keys,
    required this.mods,
    required this.onKeys,
    required this.onSend,
    required this.onTalk,
  });

  final Suite suite;
  final TextEditingController draft;
  final FocusNode focus;
  final bool terminal;
  final bool keys;
  final KeyModifiers mods;
  final VoidCallback onKeys;
  final VoidCallback onSend;
  final VoidCallback onTalk;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(top: BorderSide(color: AppColors.outline)),
      ),
      padding: const EdgeInsets.fromLTRB(6, 6, 10, 8),
      child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
        IconButton(
          tooltip: 'Controls',
          onPressed: () => _Controls.show(context, suite, terminal: terminal),
          icon: const Icon(Icons.more_horiz_rounded),
        ),
        if (terminal)
          IconButton(
            tooltip: keys ? 'Hide the keys' : 'Esc, Ctrl, Alt, Shift, Tab, arrows ...',
            isSelected: keys,
            selectedIcon: const Icon(Icons.keyboard_hide_rounded, color: AppColors.glasses),
            onPressed: onKeys,
            icon: const Icon(Icons.keyboard_alt_outlined),
          ),
        Expanded(
          child: TextField(
            controller: draft,
            focusNode: focus,
            minLines: 1,
            maxLines: 4,
            autocorrect: !terminal,
            enableSuggestions: !terminal,
            textCapitalization: terminal ? TextCapitalization.none : TextCapitalization.sentences,
            textInputAction: terminal ? TextInputAction.send : TextInputAction.newline,
            onSubmitted: terminal ? (_) => onSend() : null,
            // Kept in step with the mic's draft (else that clears what you type).
            onChanged: suite.voice.setDraft,
            // Ctrl, Alt or Shift armed on the key bar: the next letter goes to the terminal as that key.
            inputFormatters: [
              if (terminal)
                ModifiedCharFormatter(mods, (ch) {
                  final p = mods.char(ch);
                  mods.used();
                  suite.terminal.press(p);
                }),
            ],
            style: terminal ? const TextStyle(fontFamily: 'monospace') : null,
            decoration: InputDecoration(
              isDense: true,
              hintText: terminal ? 'Type to ${suite.terminal.label}' : 'Type or talk',
              hintMaxLines: 1,
            ),
          ),
        ),
        const SizedBox(width: 6),
        SizedBox(
          width: 44,
          height: 44,
          child: IconButton.filled(
            tooltip: terminal ? 'Send (Enter)' : 'Send',
            style: IconButton.styleFrom(
              backgroundColor: terminal ? AppColors.glasses : AgentSections.colorOf(suite.settings.agent),
              foregroundColor: const Color(0xFF0B1220),
            ),
            onPressed: onSend,
            icon: Icon(terminal ? Icons.keyboard_return_rounded : Icons.arrow_upward_rounded, size: 20),
          ),
        ),
        const SizedBox(width: 6),
        TalkButton(voice: suite.voice, talk: suite.conversation, narrator: suite.narrator, size: 44, onTap: onTalk),
      ]),
    );
  }
}

/// Everything else, one tap away: page the glasses, hands-free, show the
/// terminal on the glasses, stop, new chat, settings.
class _Controls extends StatelessWidget {
  const _Controls({required this.suite, required this.terminal});
  final Suite suite;
  final bool terminal;

  static Future<void> show(BuildContext context, Suite suite, {required bool terminal}) =>
      showModalBottomSheet<void>(
        context: context,
        showDragHandle: true,
        isScrollControlled: true,
        backgroundColor: AppColors.surface,
        builder: (_) => SuiteScope(suite: suite, child: _Controls(suite: suite, terminal: terminal)),
      );

  @override
  Widget build(BuildContext context) {
    final term = suite.terminal;
    Widget glassesButton(IconData icon, String label, VoidCallback onTap) => Expanded(
          child: OutlinedButton.icon(
            style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 14)),
            onPressed: onTap,
            icon: Icon(icon),
            label: Text(label),
          ),
        );
    return SafeArea(
      child: ListenableBuilder(
        listenable: Listenable.merge([suite.settings, term, suite.voice]),
        builder: (context, _) => Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            const Text('On the glasses', style: TextStyle(color: AppColors.muted, fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Row(children: [
              glassesButton(Icons.keyboard_arrow_up_rounded, 'Page up', suite.display.up),
              const SizedBox(width: 8),
              glassesButton(Icons.keyboard_arrow_down_rounded, 'Page down', suite.display.down),
              const SizedBox(width: 8),
              glassesButton(Icons.layers_clear_rounded, 'Clear', suite.display.clear),
            ]),
            const SizedBox(height: 10),
            HandsFreeTile(
              voice: suite.voice,
              autoOffMinutes: suite.settings.autoOffMinutes,
              onChanged: suite.setHandsFree,
            ),
            const SizedBox(height: 8),
            TalkModeChooser(settings: suite.settings, glassesMic: suite.glassesMicInUse),
            const Divider(height: 20),
            if (terminal) ...[
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                secondary: const Icon(Icons.visibility_rounded, color: AppColors.hud),
                title: Text('Show ${term.label} on the glasses'),
                value: term.mirroring,
                onChanged: (v) {
                  if (v && !suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
                  term.setMirror(v);
                },
              ),
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.stop_circle_outlined, color: AppColors.error),
                title: const Text('Stop what it is doing (Esc)'),
                onTap: () {
                  term.key('Escape');
                  Navigator.pop(context);
                },
              ),
            ] else ...[
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.stop_circle_outlined, color: AppColors.error),
                title: const Text('Stop the reply'),
                onTap: () async {
                  Navigator.pop(context);
                  await suite.chat.stopReply();
                  await suite.stopSpeaking();
                },
              ),
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.restart_alt_rounded),
                title: const Text('New chat'),
                subtitle: const Text('Claude forgets this conversation'),
                onTap: () async {
                  Navigator.pop(context);
                  try {
                    await suite.chat.newChat();
                  } catch (e) {
                    suite.toasts.add('$e');
                  }
                },
              ),
            ],
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.tune_rounded, color: AppColors.claude),
              title: const Text('Model, voice and replies'),
              onTap: () {
                Navigator.pop(context);
                ClaudeSettingsSheet.show(context);
              },
            ),
          ]),
        ),
      ),
    );
  }
}
