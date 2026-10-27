import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/ble/ble_manager.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';
import '../lookup/layout_page.dart';
import '../lookup/lookup_page.dart';
import '../settings/settings_page.dart';
import '../tools/images_page.dart';
import '../tools/notify_page.dart';
import '../tools/text_page.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      children: [
        Row(children: [
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Even Suite', style: Theme.of(context).textTheme.headlineSmall),
              const SizedBox(height: 2),
              const Text('Your G1, Claude and Termux in one place',
                  style: TextStyle(color: AppColors.muted)),
            ]),
          ),
          IconButton(
            tooltip: 'Settings',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const SettingsPage()),
            ),
            icon: const Icon(Icons.settings_rounded),
          ),
        ]),
        const SizedBox(height: 16),
        _GlassesCard(suite: suite),
        const SizedBox(height: 12),
        _BridgeCard(suite: suite),
        const SizedBox(height: 12),
        SectionCard(
          title: 'On your glasses',
          icon: Icons.crop_landscape_rounded,
          iconColor: AppColors.hud,
          child: ListenableBuilder(
            listenable: suite.ble,
            builder: (context, _) => GlassesPreview(display: suite.display, connected: suite.ble.isConnected),
          ),
        ),
        const SizedBox(height: 20),
        Text('Quick actions', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        LayoutBuilder(builder: (context, c) {
          final cols = c.maxWidth > 560 ? 3 : 2;
          final tiles = [
            ActionTile(
              icon: Icons.auto_awesome_rounded,
              label: 'Ask Claude',
              caption: 'Talk or type; replies on your glasses',
              color: AppColors.claude,
              onTap: () => Shell.goTo(context, Shell.claude),
            ),
            ActionTile(
              icon: Icons.terminal_rounded,
              label: 'Terminal',
              caption: 'Claude Code and your Termux shell',
              color: AppColors.glasses,
              onTap: () => Shell.goTo(context, Shell.terminal),
            ),
            ActionTile(
              icon: Icons.dashboard_customize_rounded,
              label: 'Look up screen',
              caption: 'Clock, weather, notes, notifications',
              color: AppColors.hud,
              onTap: () => _push(context, const LookUpPage()),
            ),
            ActionTile(
              icon: Icons.drag_indicator_rounded,
              label: 'Glasses layout',
              caption: 'Drag and drop what Claude shows',
              color: AppColors.claude,
              onTap: () => _push(context, const LayoutPage()),
            ),
            ActionTile(
              icon: Icons.task_alt_rounded,
              label: 'Work',
              caption: 'Told when Claude Code is done',
              color: AppColors.claude,
              onTap: () => Shell.goTo(context, Shell.work),
            ),
            ActionTile(
              icon: Icons.notes_rounded,
              label: 'Send text',
              caption: 'Show any text on the glasses',
              color: AppColors.ok,
              onTap: () => _push(context, const TextPage()),
            ),
            ActionTile(
              icon: Icons.notifications_active_rounded,
              label: 'Notifications',
              caption: 'Send a notification to the glasses',
              color: AppColors.warn,
              onTap: () => _push(context, const NotifyPage()),
            ),
            ActionTile(
              icon: Icons.image_rounded,
              label: 'Images',
              caption: 'Show a picture on the display',
              color: const Color(0xFFB794F6),
              onTap: () => _push(context, const ImagesPage()),
            ),
            ActionTile(
              icon: Icons.cast_rounded,
              label: 'Phone screen',
              caption: 'Mirror your screen to the glasses',
              color: AppColors.hud,
              onTap: () => Shell.goTo(context, Shell.screen),
            ),
            ActionTile(
              icon: Icons.layers_clear_rounded,
              label: 'Clear glasses',
              caption: 'Back to the glasses dashboard',
              color: AppColors.muted,
              onTap: () => suite.display.clear(),
            ),
          ];
          // Fixed height that grows with the phone's text size, so nothing clips.
          final scale = MediaQuery.textScalerOf(context).scale(1);
          return GridView(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: cols,
              mainAxisSpacing: 12,
              crossAxisSpacing: 12,
              mainAxisExtent: 92 + 62 * scale,
            ),
            children: tiles,
          );
        }),
      ],
    );
  }

  static void _push(BuildContext context, Widget page) =>
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
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
                    onPressed: () => Shell.goTo(context, Shell.claude),
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
