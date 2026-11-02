import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../widgets/common.dart';

/// Things Claude can do on the phone, the permissions they need, and the
/// reminders still to come.
class PhonePage extends StatefulWidget {
  const PhonePage({super.key});

  @override
  State<PhonePage> createState() => _PhonePageState();
}

class _PhonePageState extends State<PhonePage> with WidgetsBindingObserver {
  Map status = const {};
  List reminders = const [];
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_ready) return;
    _ready = true;
    _refresh();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Back from Android's settings: show what changed.
    if (state == AppLifecycleState.resumed) _refresh();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<void> _refresh() async {
    final system = SuiteScope.of(context).system;
    final s = await system.phone({'do': 'status'});
    final r = await system.phone({'do': 'reminders'});
    if (!mounted) return;
    setState(() {
      status = s;
      reminders = (r['list'] as List?) ?? const [];
    });
  }

  static String _when(int ms) {
    final t = DateTime.fromMillisecondsSinceEpoch(ms);
    final now = DateTime.now();
    final h = t.hour % 12 == 0 ? 12 : t.hour % 12;
    final clock = '$h:${t.minute.toString().padLeft(2, '0')} ${t.hour < 12 ? 'AM' : 'PM'}';
    final days = DateTime(t.year, t.month, t.day).difference(DateTime(now.year, now.month, now.day)).inDays;
    return days == 0 ? 'Today $clock' : days == 1 ? 'Tomorrow $clock' : '${t.month}/${t.day} $clock';
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Phone actions')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) => RefreshIndicator(
          onRefresh: _refresh,
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Let Claude use your phone', style: TextStyle(fontWeight: FontWeight.w700)),
                subtitle: const Text('Reminders, timers, alarms, apps, calls, texts, music, directions.',
                    style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                value: s.phoneActions,
                onChanged: (v) => s.set('phoneActions', v),
              ),
              const SizedBox(height: 8),
              const _Examples('Instant (no Claude needed)', Icons.bolt_rounded, AppColors.hud, [
                'Flashlight on',
                'Next song · Pause the music',
                'Volume up',
                'Set a timer for 5 minutes',
                'Remind me in 20 minutes to check the oven',
              ]),
              const SizedBox(height: 12),
              const _Examples('Ask Claude', Icons.auto_awesome_rounded, AppColors.claude, [
                'Remind me tomorrow at 9 to call the bank',
                'Wake me up at 6:30',
                'Open Spotify',
                'Call Mom · Text Sam I’m running late',
                'Directions to the nearest Costco',
              ]),
              const SizedBox(height: 12),
              SectionCard(
                title: 'So it works in your pocket',
                icon: Icons.verified_user_rounded,
                iconColor: AppColors.glasses,
                child: Column(children: [
                  _Permission(
                    title: 'Display over other apps',
                    sub: 'Lets it open apps, calls and maps while the phone is locked or in your pocket.',
                    on: status['overlay'] == true,
                    onAllow: () => suite.system.phone({'do': 'openOverlaySettings'}),
                  ),
                  _Permission(
                    title: 'Alarms and reminders',
                    sub: 'Reminders ring on the minute instead of a few minutes late.',
                    on: status['exactAlarms'] == true,
                    onAllow: () => suite.system.phone({'do': 'openAlarmSettings'}),
                  ),
                  _Permission(
                    title: 'Contacts',
                    sub: 'Call or text people by name.',
                    on: status['contacts'] == true,
                    onAllow: () async {
                      await suite.system.permissions(['contacts']);
                      await _refresh();
                    },
                  ),
                  _Permission(
                    title: 'Phone calls',
                    sub: 'Call right away instead of opening the dialer.',
                    on: status['calls'] == true,
                    onAllow: () async {
                      await suite.system.permissions(['calls']);
                      await _refresh();
                    },
                  ),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Coming up',
                icon: Icons.alarm_rounded,
                iconColor: AppColors.warn,
                trailing: IconButton(onPressed: _refresh, icon: const Icon(Icons.refresh_rounded)),
                child: reminders.isEmpty
                    ? const Text('No reminders. Say “remind me…” to the glasses.',
                        style: TextStyle(color: AppColors.muted))
                    : Column(children: [
                        for (final r in reminders)
                          ListTile(
                            contentPadding: EdgeInsets.zero,
                            leading: const Icon(Icons.notifications_active_rounded, color: AppColors.warn),
                            title: Text('${(r as Map)['text']}', style: const TextStyle(fontWeight: FontWeight.w600)),
                            subtitle: Text(_when((r['at'] as num).toInt()),
                                style: const TextStyle(color: AppColors.muted)),
                            trailing: IconButton(
                              tooltip: 'Cancel',
                              icon: const Icon(Icons.close_rounded),
                              onPressed: () async {
                                await suite.system.phone({'do': 'cancelReminder', 'id': r['id']});
                                await _refresh();
                              },
                            ),
                          ),
                      ]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Examples extends StatelessWidget {
  const _Examples(this.title, this.icon, this.color, this.lines);
  final String title;
  final IconData icon;
  final Color color;
  final List<String> lines;

  @override
  Widget build(BuildContext context) => SectionCard(
        title: title,
        icon: icon,
        iconColor: color,
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          for (final l in lines)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(children: [
                Icon(Icons.format_quote_rounded, size: 16, color: color),
                const SizedBox(width: 8),
                Expanded(child: Text(l, style: const TextStyle(fontWeight: FontWeight.w500))),
              ]),
            ),
        ]),
      );
}

class _Permission extends StatelessWidget {
  const _Permission({required this.title, required this.sub, required this.on, required this.onAllow});
  final String title;
  final String sub;
  final bool on;
  final VoidCallback onAllow;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Row(children: [
          Icon(on ? Icons.check_circle_rounded : Icons.radio_button_unchecked_rounded,
              color: on ? AppColors.ok : AppColors.muted),
          const SizedBox(width: 10),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
              Text(sub, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ]),
          ),
          if (!on) TextButton(onPressed: onAllow, child: const Text('Allow')),
        ]),
      );
}
