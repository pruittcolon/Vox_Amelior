import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/ble/ble_manager.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/settings.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';
import '../../widgets/session_bar.dart';
import '../../widgets/voice_widgets.dart';
import '../lookup/lookup_page.dart';
import '../settings/settings_page.dart';
import '../settings/text_timing_page.dart';
import '../settings/touchbar_page.dart';
import '../tools/phone_page.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key, this.active = true});

  /// On screen and the app in front (the session list stays fresh).
  final bool active;

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return ListenableBuilder(
      listenable: Listenable.merge([suite.ble, suite.chat]),
      builder: (context, _) {
        final connected = suite.ble.isConnected;
        final online = suite.chat.state == BridgeState.online;
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
          children: [
            _Header(suite: suite),
            const SizedBox(height: 16),
            _TalkHero(suite: suite, active: active),
            if (!connected) ...[const SizedBox(height: 12), _GlassesCard(suite: suite)],
            if (!online) ...[const SizedBox(height: 12), _BridgeCard(suite: suite)],
            const SizedBox(height: 12),
            SectionCard(
              title: 'On your glasses',
              icon: Icons.crop_landscape_rounded,
              iconColor: AppColors.hud,
              child: GlassesPreview(display: suite.display, connected: connected),
            ),
            const SizedBox(height: 22),
            Text('Make it yours', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 4),
            const Text('What you see, and what the TouchBar does', style: TextStyle(color: AppColors.muted)),
            const SizedBox(height: 12),
            LayoutBuilder(builder: (context, c) {
              final cols = c.maxWidth > 560 ? 4 : 2;
              final scale = MediaQuery.textScalerOf(context).scale(1);
              return GridView(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: cols,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  mainAxisExtent: 104 + 58 * scale,
                ),
                children: [
                  _FeatureCard(
                    icon: Icons.dashboard_customize_rounded,
                    color: AppColors.hud,
                    title: 'Look up screen',
                    caption: 'Your own layout: clock, Claude, weather - drag anything',
                    onTap: () => _push(context, const LookUpPage()),
                  ),
                  _FeatureCard(
                    icon: Icons.touch_app_rounded,
                    color: AppColors.glasses,
                    title: 'TouchBar',
                    caption: 'Choose what a hold and a double tap do',
                    onTap: () => _push(context, const TouchBarPage()),
                  ),
                  _FeatureCard(
                    icon: Icons.phone_android_rounded,
                    color: AppColors.warn,
                    title: 'Phone actions',
                    caption: 'Reminders, timers, apps, calls - just say it',
                    onTap: () => _push(context, const PhonePage()),
                  ),
                  _FeatureCard(
                    icon: Icons.task_alt_rounded,
                    color: AppColors.ok,
                    title: 'Work',
                    caption: 'Told the moment Claude Code is done',
                    onTap: () => Shell.goTo(context, Shell.work),
                  ),
                  _FeatureCard(
                    icon: Icons.timer_rounded,
                    color: AppColors.claude,
                    title: 'Text timing',
                    caption: 'How long each screen of an answer stays',
                    onTap: () => _push(context, const TextTimingPage()),
                  ),
                  _FeatureCard(
                    icon: Icons.terminal_rounded,
                    color: AppColors.glasses,
                    title: 'Sessions',
                    caption: 'Claude Code as in Termux - open more, switch',
                    onTap: () {
                      suite.useSession(suite.terminal.session);
                      Shell.goTo(context, Shell.terminal);
                    },
                  ),
                ],
              );
            }),
            const SizedBox(height: 14),
            OutlinedButton.icon(
              onPressed: () => Shell.goTo(context, Shell.more),
              icon: const Icon(Icons.apps_rounded),
              label: const Text('Screen sharing, text height and more'),
            ),
          ],
        );
      },
    );
  }

  static void _push(BuildContext context, Widget page) =>
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
}

/// Title, plus one tap to see (or fix) the glasses and Claude.
class _Header extends StatelessWidget {
  const _Header({required this.suite});
  final Suite suite;

  void _sheet(BuildContext context, Widget child) => showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        backgroundColor: AppColors.bg,
        builder: (_) => SuiteScope(
          suite: suite,
          child: Padding(padding: const EdgeInsets.fromLTRB(16, 0, 16, 24), child: child),
        ),
      );

  @override
  Widget build(BuildContext context) {
    final ble = suite.ble;
    final chat = suite.chat;
    final (gLabel, gColor) = switch (ble.state) {
      LinkState.connected => ('Glasses', AppColors.ok),
      LinkState.connecting => ('Connecting', AppColors.warn),
      LinkState.scanning => ('Searching', AppColors.glasses),
      LinkState.disconnected => ('No glasses', AppColors.muted),
    };
    final (cLabel, cColor) = switch (chat.state) {
      BridgeState.online => ('Claude', AppColors.ok),
      BridgeState.starting => ('Starting', AppColors.warn),
      BridgeState.offline => ('Claude off', AppColors.error),
      BridgeState.unknown => ('Checking', AppColors.muted),
    };
    return Row(children: [
      Expanded(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          ShaderMask(
            shaderCallback: (r) => const LinearGradient(colors: [AppColors.text, AppColors.glasses]).createShader(r),
            child: Text('Even Suite',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.6,
                      color: Colors.white,
                    )),
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 6, children: [
            _StatusChip(
              icon: Icons.visibility_rounded,
              label: gLabel,
              color: gColor,
              pulse: ble.state == LinkState.connecting || ble.state == LinkState.scanning,
              onTap: () => _sheet(context, _GlassesCard(suite: suite)),
            ),
            _StatusChip(
              icon: Icons.auto_awesome_rounded,
              label: cLabel,
              color: cColor,
              pulse: chat.state == BridgeState.starting,
              onTap: () => _sheet(context, _BridgeCard(suite: suite)),
            ),
            GlassesBattery(ble: ble, compact: false, onTap: () => _sheet(context, _GlassesCard(suite: suite))),
          ]),
        ]),
      ),
      IconButton.filledTonal(
        tooltip: 'Settings',
        onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsPage())),
        icon: const Icon(Icons.settings_rounded),
      ),
    ]);
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.icon, required this.label, required this.color, required this.onTap, this.pulse = false});
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;
  final bool pulse;

  @override
  Widget build(BuildContext context) => Material(
        color: color.withOpacity(0.12),
        shape: StadiumBorder(side: BorderSide(color: color.withOpacity(0.35))),
        child: InkWell(
          customBorder: const StadiumBorder(),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(10, 5, 12, 5),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              StatusPill.dot(color, pulse: pulse),
              const SizedBox(width: 6),
              Icon(icon, size: 14, color: color),
              const SizedBox(width: 4),
              Text(label, style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 12.5)),
            ]),
          ),
        ),
      );
}

/// The main thing: talk. Where it goes (chat or one of your terminals), the
/// big button, where it is at.
class _TalkHero extends StatelessWidget {
  const _TalkHero({required this.suite, required this.active});
  final Suite suite;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final s = suite.settings;
    return ListenableBuilder(
      listenable: Listenable.merge([s, suite.terminal]),
      builder: (context, _) {
        final terminal = s.target == Target.terminal;
        final label = suite.terminal.label;
        final accent = terminal ? AppColors.glasses : AppColors.claude;
        final hold = switch (s.holdAction) {
          'chat' => 'Claude chat',
          'terminal' => label,
          _ => terminal ? label : ChatController.agentLabel(s.agent),
        };
        return Container(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(26),
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [accent.withOpacity(0.24), AppColors.surface, AppColors.surface],
              stops: const [0, 0.55, 1],
            ),
            border: Border.all(color: accent.withOpacity(0.35)),
            boxShadow: [BoxShadow(color: accent.withOpacity(0.10), blurRadius: 30, offset: const Offset(0, 10))],
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Row(children: [
              Expanded(
                child: Text(terminal ? label : 'Ask ${ChatController.agentLabel(s.agent)}',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
              ),
              if (terminal)
                TextButton.icon(
                  onPressed: () => Shell.goTo(context, Shell.terminal),
                  icon: const Icon(Icons.open_in_full_rounded, size: 18),
                  label: const Text('Watch it'),
                ),
            ]),
            const SizedBox(height: 4),
            Text(
              terminal
                  ? 'What you say is typed into $label in Termux. Its steps and answer come back here and on your glasses.'
                  : 'Ask anything. The answer shows on your glasses and can be read aloud.',
              style: const TextStyle(color: AppColors.muted, fontSize: 13.5),
            ),
            const SizedBox(height: 12),
            SessionBar(suite: suite, active: active),
            const SizedBox(height: 14),
            Center(
              child: TalkButton(
                voice: suite.voice,
                talk: suite.conversation,
                narrator: suite.narrator,
                size: 92,
                onTap: () async {
                  if (suite.narrator.speaking) return suite.stopSpeaking();
                  await suite.toggleTalk();
                },
              ),
            ),
            const SizedBox(height: 10),
            TalkStatus(
              voice: suite.voice,
              talk: suite.conversation,
              narrator: suite.narrator,
              target: terminal ? label : ChatController.agentLabel(s.agent),
            ),
            const SizedBox(height: 14),
            Row(children: [
              Icon(Icons.touch_app_rounded, size: 16, color: accent),
              const SizedBox(width: 6),
              Expanded(
                child: Text('Or hold the left TouchBar to talk to $hold',
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
              ),
            ]),
            const SizedBox(height: 10),
            HandsFreeTile(voice: suite.voice, onChanged: suite.setHandsFree, autoOffMinutes: s.autoOffMinutes),
          ]),
        );
      },
    );
  }
}

class _FeatureCard extends StatelessWidget {
  const _FeatureCard({
    required this.icon,
    required this.color,
    required this.title,
    required this.caption,
    required this.onTap,
  });

  final IconData icon;
  final Color color;
  final String title;
  final String caption;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(
        color: AppColors.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: const BorderSide(color: AppColors.outline),
        ),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: Ink(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [color.withOpacity(0.14), Colors.transparent],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  IconBadge(icon, color: color, size: 42),
                  const Spacer(),
                  Icon(Icons.north_east_rounded, size: 18, color: color.withOpacity(0.8)),
                ]),
                const Spacer(),
                Text(title,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis),
                const SizedBox(height: 2),
                Text(caption,
                    style: Theme.of(context).textTheme.bodySmall, maxLines: 2, overflow: TextOverflow.ellipsis),
              ]),
            ),
          ),
        ),
      );
}

class _GlassesCard extends StatelessWidget {
  const _GlassesCard({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    final ble = suite.ble;
    return ListenableBuilder(
      listenable: ble,
      builder: (context, _) {
        final (label, color) = switch (ble.state) {
          LinkState.connected => ('Connected', AppColors.ok),
          LinkState.connecting => ('Connecting…', AppColors.warn),
          LinkState.scanning => ('Searching…', AppColors.glasses),
          LinkState.disconnected => ('Not connected', AppColors.muted),
        };
        final busy = ble.state == LinkState.connecting || ble.state == LinkState.scanning;
        final pct = ble.battery;
        final last = suite.settings.lastGlasses;
        final name = ble.current?.label ?? (last != null ? GlassesPair.fromMap(last).label : 'Even G1');
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Row(children: [
                Container(
                  width: 72,
                  height: 56,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: AppColors.glasses.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: GlassesGlyph(
                    size: 50,
                    color: ble.isConnected ? AppColors.glasses : AppColors.muted,
                    left: ble.state != LinkState.connected || ble.leftUp,
                    right: ble.state != LinkState.connected || ble.rightUp,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(name, style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 6),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: StatusPill(label, color: color, pulse: busy),
                    ),
                  ]),
                ),
              ]),
              if (ble.lastError != null) ...[
                const SizedBox(height: 12),
                Text(ble.lastError!, style: const TextStyle(color: AppColors.warn)),
              ],
              if (ble.isConnected) ...[
                const SizedBox(height: 14),
                Row(children: [
                  Icon(
                    pct == null ? Icons.battery_unknown_rounded : GlassesBattery.icon(pct),
                    color: pct == null ? AppColors.muted : GlassesBattery.color(pct),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      pct == null ? 'Battery: asking the glasses…' : 'Battery: ${GlassesBattery.arms(ble)}',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  TextButton(onPressed: () => suite.askBattery(now: true), child: const Text('Check')),
                ]),
              ],
              const SizedBox(height: 16),
              if (ble.state == LinkState.connected)
                Row(children: [
                  const Expanded(
                    child: Text('Tap left/right on the TouchBar to page. Hold left to talk.',
                        style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton(onPressed: ble.disconnect, child: const Text('Disconnect')),
                ])
              else if (ble.state == LinkState.connecting)
                Row(children: [
                  const Expanded(
                    child: Text('Open the case lid or put the glasses on. This keeps trying on its own.',
                        style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  ),
                  TextButton(onPressed: ble.disconnect, child: const Text('Cancel')),
                ])
              else ...[
                FilledButton.icon(
                  onPressed: ble.state == LinkState.scanning ? null : suite.scan,
                  icon: const Icon(Icons.bluetooth_searching_rounded),
                  label: Text(ble.state == LinkState.scanning ? 'Searching for glasses…' : 'Find my glasses'),
                ),
                if (last != null && ble.found.isEmpty) ...[
                  const SizedBox(height: 8),
                  OutlinedButton.icon(
                    onPressed: () => ble.connect(GlassesPair.fromMap(last)),
                    icon: const Icon(Icons.link_rounded),
                    label: Text('Reconnect ${GlassesPair.fromMap(last).label}'),
                  ),
                ],
              ],
              if (ble.found.isNotEmpty && ble.state != LinkState.connected) ...[
                const SizedBox(height: 14),
                const Text('Found nearby', style: TextStyle(color: AppColors.muted, fontWeight: FontWeight.w600)),
                const SizedBox(height: 6),
                for (final pair in ble.found)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Material(
                      color: AppColors.surfaceHigh,
                      borderRadius: BorderRadius.circular(14),
                      child: ListTile(
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                        leading: const Icon(Icons.visibility_rounded, color: AppColors.glasses),
                        title: Text(pair.label, style: const TextStyle(fontWeight: FontWeight.w700)),
                        subtitle: Text('${pair.leftName}\n${pair.rightName}',
                            style: const TextStyle(color: AppColors.muted, fontSize: 12)),
                        isThreeLine: true,
                        trailing: const Icon(Icons.chevron_right_rounded),
                        onTap: () => ble.connect(pair),
                      ),
                    ),
                  ),
              ],
            ]),
          ),
        );
      },
    );
  }
}

class _BridgeCard extends StatefulWidget {
  const _BridgeCard({required this.suite});
  final Suite suite;

  @override
  State<_BridgeCard> createState() => _BridgeCardState();
}

class _BridgeCardState extends State<_BridgeCard> {
  bool _starting = false;

  Future<void> _run(Future<String> Function() action) async {
    setState(() => _starting = true);
    final msg = await action();
    if (!mounted) return;
    setState(() => _starting = false);
    showToast(context, msg);
  }

  Future<void> _start() => _run(widget.suite.startBridge);

  @override
  Widget build(BuildContext context) {
    final chat = widget.suite.chat;
    return ListenableBuilder(
      listenable: chat,
      builder: (context, _) {
        final (label, color) = switch (chat.state) {
          BridgeState.online => ('Online', AppColors.ok),
          BridgeState.starting => ('Starting…', AppColors.warn),
          BridgeState.offline => ('Offline', AppColors.error),
          BridgeState.unknown => ('Checking…', AppColors.muted),
        };
        final online = chat.state == BridgeState.online;
        return SectionCard(
          title: 'Claude via Termux',
          icon: Icons.auto_awesome_rounded,
          iconColor: AppColors.claude,
          trailing: StatusPill(label, color: color, pulse: chat.state == BridgeState.starting),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Text(
              online
                  ? 'Connected to Agent Hub on this phone. Claude Code runs in Termux with your own login - no API key.'
                  : 'Agent Hub runs Claude Code in Termux. Tap Start, or type even in Termux.',
              style: const TextStyle(color: AppColors.muted, fontSize: 13.5),
            ),
            const SizedBox(height: 14),
            Row(children: [
              if (!online)
                Expanded(
                  child: FilledButton.icon(
                    style: FilledButton.styleFrom(backgroundColor: AppColors.claude),
                    onPressed: _starting ? null : _start,
                    icon: _starting
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.play_arrow_rounded),
                    label: Text(_starting ? 'Starting…' : 'Start in Termux'),
                  ),
                )
              else
                Expanded(
                  child: FilledButton.icon(
                    style: FilledButton.styleFrom(backgroundColor: AppColors.claude),
                    onPressed: () => Shell.goTo(context, Shell.terminal),
                    icon: const Icon(Icons.chat_bubble_rounded),
                    label: const Text('Open chat'),
                  ),
                ),
              const SizedBox(width: 8),
              IconButton.outlined(
                tooltip: 'Check again',
                onPressed: chat.check,
                icon: const Icon(Icons.refresh_rounded),
              ),
              if (online)
                PopupMenuButton<String>(
                  tooltip: 'Start, restart or stop',
                  enabled: !_starting,
                  icon: _starting
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.more_vert_rounded),
                  onSelected: (v) => _run(v == 'restart' ? widget.suite.restartBridge : widget.suite.stopBridge),
                  itemBuilder: (_) => const [
                    PopupMenuItem(value: 'restart', child: Text('Restart (even restart)')),
                    PopupMenuItem(value: 'stop', child: Text('Stop (even stop)')),
                  ],
                ),
            ]),
            const SizedBox(height: 8),
            const Text('In Termux it is one word: even  ·  even stop  ·  even restart  ·  even log',
                style: TextStyle(color: AppColors.muted, fontSize: 12)),
          ]),
        );
      },
    );
  }
}
