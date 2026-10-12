import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Android bits without extra packages: saved settings, permissions, the
/// foreground service, and running commands in Termux.
class SystemBridge {
  static const _channel = MethodChannel('suite/system');

  /// Notification "Stop listening" pressed.
  VoidCallback? onStopListening;

  /// Volume key pressed while the app is open ('up' / 'down').
  void Function(String dir)? onVolumeKey;

  void init() {
    _channel.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'stopListening':
          onStopListening?.call();
        case 'volumeKey':
          onVolumeKey?.call('${call.arguments}');
      }
    });
  }

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

  /// kinds: bluetooth, microphone, notifications, termux. Returns kind -> granted.
  Future<Map<String, bool>> permissions(List<String> kinds) async {
    final m = await _invoke<Map>('permissions', {'kinds': kinds});
    if (m == null) return {for (final k in kinds) k: true};
    return {for (final e in m.entries) '${e.key}': e.value == true};
  }

  /// Keep the app alive in the background. Returns why Android refused, or null.
  Future<String?> service({required bool listening, required bool glasses, String text = ''}) =>
      _invoke<String>('service', {'listening': listening, 'glasses': glasses, 'text': text});

  Future<bool> termuxInstalled() async => await _invoke<bool>('termuxInstalled') ?? false;

  /// 'ok', 'not-installed', 'no-permission', or an error message.
  Future<String> runTermux(String command) async =>
      await _invoke<String>('runTermux', {'command': command}) ?? 'Only available on the phone';

  Future<bool> openTermux() async => await _invoke<bool>('openTermux') ?? false;

  Future<void> openAppSettings() => _invoke('openAppSettings');

  Future<void> volumeKeys(bool enabled) => _invoke('volumeKeys', {'enabled': enabled});
}
