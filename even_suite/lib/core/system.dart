
import 'package:flutter/services.dart';

/// Android bits without extra packages: saved settings, permissions, the
/// foreground service, and running commands in Termux.
class SystemBridge {
  static const _channel = MethodChannel('suite/system');

  /// Notification "Stop listening" pressed.
  VoidCallback? onStopListening;

  /// Volume key pressed while the app is open ('up' / 'down').
  void Function(String dir)? onVolumeKey;

  /// A notification of ours was tapped: open this tab ('work', ...).
  void Function(String tab)? onOpenTab;

  /// A reminder went off (while the app runs).
  void Function(String id, String text)? onReminder;

  void init() {
    _channel.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'stopListening':
          onStopListening?.call();
        case 'volumeKey':
          onVolumeKey?.call('${call.arguments}');
        case 'openTab':
          onOpenTab?.call('${call.arguments}');
        case 'reminder':
          final m = call.arguments as Map? ?? const {};
          onReminder?.call('${m['id'] ?? ''}', '${m['text'] ?? ''}');
      }
    });
  }

  /// Notifications posted by other apps (needs notification access).
  Stream<Map> get phoneNotifications => const EventChannel('suite/notifications')
      .receiveBroadcastStream()
      .map((e) => e as Map)
      .handleError((_) {});

  /// Show a notification on the phone; tapping it opens [tab].
  Future<void> notify(int id, String title, String text, {String tab = 'work'}) =>
      _invoke('notify', {'id': id, 'title': title, 'text': text, 'tab': tab});

  /// Has Even Suite been given notification access?
  Future<bool> notificationAccess() async => await _invoke<bool>('notificationAccess') ?? false;

  Future<void> openNotificationAccess() => _invoke('openNotificationAccess');

  /// The phone's last known position (for the weather), or null.
  Future<(double, double)?> location() async {
    final m = await _invoke<Map>('location');
    final lat = m?['lat'], lon = m?['lon'];
    if (lat is num && lon is num) return (lat.toDouble(), lon.toDouble());
    return null;
  }

  /// The tab a tapped notification asked for when the app was started by it.
  Future<String?> launchTab() => _invoke<String>('launchTab');

  /// Do something on the phone (see PhoneAction): {ok, say} or {ok: false, error}.
  Future<Map> phone(Map<String, Object?> action) async =>
      await _invoke<Map>('phone', action) ?? const {'ok': false, 'error': 'not available here'};

  /// Where the app keeps its own files (downloaded speech models).
  Future<String?> filesDir() => _invoke<String>('filesDir');

  /// The phone's battery, 0-100 (null when unknown).
  Future<int?> battery() => _invoke<int>('battery');

  Future<T?> _invoke<T>(String method, [Object? args]) async {
    try {
      return await _channel.invokeMethod<T>(method, args);
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }

  Future<Map<String, Object?>> prefs() async {
    final m = await _invoke<Map>('prefsGetAll');
    return {for (final e in (m ?? const {}).entries) '${e.key}': e.value};
  }

  Future<void> setPref(String key, Object? value) =>
      _invoke('prefsSet', {'key': key, 'value': value});

  /// kinds: bluetooth, microphone, notifications, termux, location. Returns kind -> granted.
  Future<Map<String, bool>> permissions(List<String> kinds) async {
    final m = await _invoke<Map>('permissions', {'kinds': kinds});
    if (m == null) return {for (final k in kinds) k: true};
    return {for (final e in m.entries) '${e.key}': e.value == true};
  }

  /// Keep the app alive in the background. Returns why Android refused, or null.
  /// [mic]: may use the microphone from the background (a TouchBar hold with
  /// the phone in a pocket).
  Future<String?> service({required bool listening, required bool glasses, bool mic = false, String text = ''}) =>
      _invoke<String>('service', {'listening': listening, 'glasses': glasses, 'mic': mic, 'text': text});

  Future<bool> termuxInstalled() async => await _invoke<bool>('termuxInstalled') ?? false;

  /// 'ok', 'not-installed', 'no-permission', or an error message.
  Future<String> runTermux(String command) async =>
      await _invoke<String>('runTermux', {'command': command}) ?? 'Only available on the phone';

  Future<bool> openTermux() async => await _invoke<bool>('openTermux') ?? false;

  Future<void> openAppSettings() => _invoke('openAppSettings');

  Future<void> volumeKeys(bool enabled) => _invoke('volumeKeys', {'enabled': enabled});

  /// Native counters since the last call (Bluetooth writes, mic, recognizer, screen).
  Future<Map<String, Object?>> nativeStats() async {
    final m = await _invoke<Map>('nativeStats');
    return {for (final e in (m ?? const {}).entries) '${e.key}': e.value};
  }
}
