import 'package:flutter/material.dart';

import '../app/shell.dart';
import '../app/theme.dart';
import '../core/ble/ble_manager.dart';

/// What keeps the glasses linked while the app is out of sight, as plain
/// checks with a button where the phone must be told to allow it:
/// notifications (the service that holds the link shows one), no battery
/// restriction (Android freezes or kills restricted apps), and where the
/// link is now.
class StayConnectedCard extends StatefulWidget {
  const StayConnectedCard({super.key});

  @override
  State<StayConnectedCard> createState() => _StayConnectedCardState();
}

class _StayConnectedCardState extends State<StayConnectedCard> with WidgetsBindingObserver {
  bool? _battery;
  bool? _notifications;
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
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Back from Android's dialog or settings: show what changed.
    if (state == AppLifecycleState.resumed) _refresh();
  }

  Future<void> _refresh() async {
    final suite = SuiteScope.of(context);
    final battery = await suite.system.batteryUnrestricted();
    final notifications = await suite.system.notificationsAllowed();
    if (!mounted) return;
    setState(() {
      _battery = battery;
      _notifications = notifications;
    });
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final ble = suite.ble;
    return ListenableBuilder(
      listenable: ble,
      builder: (context, _) {
        final (link, linkColor) = switch (ble.state) {
          LinkState.connected => ('Linked', AppColors.ok),
          LinkState.connecting => ('Reconnecting', AppColors.warn),
          LinkState.scanning => ('Searching', AppColors.glasses),
          LinkState.disconnected => ('Not linked', AppColors.muted),
        };
        Widget row(IconData icon, Color color, String title, String sub, {Widget? action}) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(icon, color: color, size: 22),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
                    Text(sub, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  ]),
                ),
                if (action != null) action,
              ]),
            );
        return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Text(
              'The glasses drop the link when nothing is sent to them for about 30 seconds. While the app is in '
              'your pocket it keeps sending a heartbeat, and Android has to let it.',
              style: TextStyle(color: AppColors.muted, fontSize: 13)),
          const SizedBox(height: 6),
          row(Icons.bluetooth_connected_rounded, linkColor, 'The glasses: $link',
              ble.state == LinkState.connected
                  ? 'It stays linked after you close the app. “Disconnect glasses” in the notification lets go.'
                  : 'It keeps trying until you disconnect it.'),
          row(
            _battery == false ? Icons.battery_alert_rounded : Icons.battery_charging_full_rounded,
            _battery == false ? AppColors.warn : AppColors.ok,
            _battery == false ? 'Battery: restricted' : 'Battery: not restricted',
            _battery == false
                ? 'Android may freeze or stop the app in the background. Allow it to run without limits.'
                : 'Android will not stop the app to save battery.',
            action: _battery == false
                ? FilledButton.tonal(onPressed: suite.system.requestBatteryUnrestricted, child: const Text('Allow'))
                : null,
          ),
          row(
            _notifications == false ? Icons.notifications_off_rounded : Icons.notifications_active_rounded,
            _notifications == false ? AppColors.warn : AppColors.ok,
            _notifications == false ? 'Notifications: off' : 'Notifications: on',
            'The app keeps a small notification up while the glasses are linked - Android needs it to keep the '
                'app running.',
            action: _notifications == false
                ? FilledButton.tonal(onPressed: () => suite.system.openAppSettings(), child: const Text('Open'))
                : null,
          ),
          const SizedBox(height: 4),
          const Text(
              'Samsung: Settings › Battery › Background usage limits › Never sleeping apps › add Even Suite. '
              'Other phones: Settings › Apps › Even Suite › Battery › Unrestricted.',
              style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
        ]);
      },
    );
  }
}
