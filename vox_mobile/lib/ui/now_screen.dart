import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Home: turn Vox on/off, see its state and what was just said.
class NowScreen extends StatefulWidget {
  const NowScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<NowScreen> createState() => _NowScreenState();
}

class _NowScreenState extends State<NowScreen> {
  StreamSubscription<Map<Object?, Object?>>? _events;
  List<SegmentView> _recent = const [];

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _load();
    s.dataVersion.addListener(_load);
    _events = s.listening.events.listen((e) {
      if (e['type'] == ServiceEvents.segment) _load();
    });
  }

  @override
  void dispose() {
    s.dataVersion.removeListener(_load);
    unawaited(_events?.cancel());
    super.dispose();
  }

  void _load() {
    if (!mounted) return;
    final today = DateTime.now();
    final start = DateTime(today.year, today.month, today.day);
    setState(() => _recent = s.transcripts.between(start, start.add(const Duration(days: 1))).reversed.take(40).toList());
  }

  Future<void> _toggle() async {
    final l = s.listening;
    if (l.isListening) {
      await l.stop();
      return;
    }
    if (!s.writeServiceConfig()) {
      await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ModelsScreen(services: s)));
      return;
    }
    final error = await l.start(useLocation: s.settings.value.locationMode != LocationMode.off);
    if (error != null && mounted) showMessage(context, error);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListenableBuilder(
          listenable: Listenable.merge([s.listening, s.downloads]),
          builder: (context, _) => CustomScrollView(
            slivers: [
              SliverToBoxAdapter(child: _header(context)),
              SliverToBoxAdapter(child: Padding(padding: const EdgeInsets.symmetric(horizontal: 16), child: _hero(context))),
              ..._notices(context),
              SliverToBoxAdapter(
                child: SectionHeader('Heard today', trailing: Text('${_recent.length}${_recent.length == 40 ? '+' : ''}')),
              ),
              if (_recent.isEmpty)
                const SliverToBoxAdapter(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: EmptyState(
                      icon: Icons.graphic_eq,
                      title: 'Nothing yet today',
                      message: 'Turn Vox on and talk normally. What is said appears here.',
                    ),
                  ),
                )
              else
                SliverList.builder(
                  itemCount: _recent.length,
                  itemBuilder: (context, i) => _line(context, _recent[i]),
                ),
              const SliverToBoxAdapter(child: SizedBox(height: 24)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header(BuildContext context) {
    final t = Theme.of(context);
    final now = DateTime.now();
    final greeting = now.hour < 12 ? 'Good morning' : (now.hour < 18 ? 'Good afternoon' : 'Good evening');
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(formatDayName(now), style: t.textTheme.labelLarge?.copyWith(color: t.colorScheme.primary)),
          Text(greeting, style: t.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }

  Widget _hero(BuildContext context) {
    final t = Theme.of(context);
    final l = s.listening;
    final on = l.isListening;
    final state = l.state;
    final (Color color, IconData icon, String title) = switch ((on, state)) {
      (false, _) => (t.colorScheme.outline, Icons.mic_off_rounded, 'Vox is off'),
      (true, ListenState.listening) => (const Color(0xFF0E9F6E), Icons.graphic_eq_rounded, 'Listening'),
      (true, ListenState.starting) => (t.colorScheme.primary, Icons.hourglass_top_rounded, 'Starting…'),
      (true, ListenState.paused) => (const Color(0xFFE8590C), Icons.pause_circle_rounded, 'Paused'),
      (true, ListenState.autoPaused) => (const Color(0xFFE8590C), Icons.location_off_rounded, 'Paused here'),
      (true, ListenState.error) => (t.colorScheme.error, Icons.error_rounded, 'Needs attention'),
    };
    final detail = !on
        ? 'Tap to start. Vox keeps listening in the background.'
        : (l.lastError ?? (l.reason.isEmpty ? 'Ready' : l.reason));
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [color.withValues(alpha: 0.18), color.withValues(alpha: 0.06)],
        ),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(color: color.withValues(alpha: 0.18), shape: BoxShape.circle),
                child: Icon(icon, color: color, size: 28),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800, color: color)),
                    const SizedBox(height: 2),
                    Text(detail, style: t.textTheme.bodyMedium, maxLines: 3, overflow: TextOverflow.ellipsis),
                  ],
                ),
              ),
            ],
          ),
          if (on) ...[
            const SizedBox(height: 14),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Pill('${l.heardToday} heard today', icon: Icons.record_voice_over_rounded),
                if (l.backlog > 0) Pill('${l.backlog} to transcribe', icon: Icons.queue_rounded, color: const Color(0xFFB08800)),
                if (l.isThinking) const Pill('Thinking', icon: Icons.auto_awesome_rounded, color: Color(0xFF9C36B5)),
                if (l.isReviewing) const Pill('Reviewing', icon: Icons.manage_search_rounded, color: Color(0xFF0B7285)),
              ],
            ),
          ],
          const SizedBox(height: 18),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: l.isBusy ? null : _toggle,
                  icon: Icon(on ? Icons.stop_rounded : Icons.mic_rounded),
                  label: Text(l.isBusy ? 'Starting…' : (on ? 'Turn off' : 'Start listening')),
                ),
              ),
              if (on && state == ListenState.error) ...[
                const SizedBox(width: 10),
                OutlinedButton(onPressed: l.retry, child: const Text('Retry')),
              ] else if (on) ...[
                const SizedBox(width: 10),
                OutlinedButton(
                  onPressed: state == ListenState.paused ? l.resume : l.pause,
                  child: Text(state == ListenState.paused ? 'Resume' : 'Pause'),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  List<Widget> _notices(BuildContext context) {
    final t = Theme.of(context);
    final l = s.listening;
    final cards = <Widget>[
      if (!s.downloads.speechReady)
        _notice(
          context,
          icon: Icons.download_rounded,
          title: 'Download the speech models',
          text: 'Needed once (about 1.2 GB) before Vox can listen.',
          action: 'Open',
          onTap: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ModelsScreen(services: s))),
        ),
      if (l.isListening && !l.batteryOk)
        _notice(
          context,
          icon: Icons.battery_alert_rounded,
          title: 'Let Vox run in the background',
          text: 'Otherwise Android may stop listening when the screen is off.',
          action: 'Allow',
          onTap: l.requestBatteryExemption,
          color: t.colorScheme.tertiaryContainer,
        ),
    ];
    return [
      for (final c in cards) SliverToBoxAdapter(child: Padding(padding: const EdgeInsets.fromLTRB(16, 12, 16, 0), child: c)),
    ];
  }

  Widget _notice(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String text,
    required String action,
    required VoidCallback onTap,
    Color? color,
  }) =>
      VoxCard(
        color: color ?? Theme.of(context).colorScheme.secondaryContainer,
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Row(
          children: [
            Icon(icon),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
                  Text(text),
                ],
              ),
            ),
            TextButton(onPressed: onTap, child: Text(action)),
          ],
        ),
      );

  Widget _line(BuildContext context, SegmentView seg) {
    final t = Theme.of(context);
    return InkWell(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute<void>(builder: (_) => ConversationScreen(services: s, conversationId: seg.conversationId)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SpeakerAvatar(label: seg.speakerLabel, known: seg.isKnownSpeaker),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(seg.speakerLabel,
                          style: TextStyle(fontWeight: FontWeight.w700, color: speakerColor(seg.speakerLabel, known: seg.isKnownSpeaker))),
                      const SizedBox(width: 8),
                      Text(formatTime(seg.startedAt), style: t.textTheme.bodySmall),
                    ],
                  ),
                  const SizedBox(height: 2),
                  Text(seg.text, style: t.textTheme.bodyLarge),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
