import 'package:flutter/foundation.dart';

import 'voice/wake_word.dart';

/// Where a spoken or typed message goes.
enum Target { claude, terminal }

/// User settings, saved on the phone (Android SharedPreferences).
class Settings extends ChangeNotifier {
  Settings([this._save]);

  final Future<void> Function(String key, Object? value)? _save;

  static const defaultBridgeUrl = 'http://127.0.0.1:8787';
  static const defaultStartCommand =
      "proot-distro login ubuntu -- env HUB_NO_BROWSER=1 bash -lc '~/agent-hub/start.sh'";

  String bridgeUrl = defaultBridgeUrl;
  String startCommand = defaultStartCommand;
  List<String> wakeWords = List.of(WakeWord.defaults);
  int autoOffMinutes = 10;
  bool quietChime = true;
  String recognizer = 'auto'; // auto, ondevice, default
  String language = ''; // '' = phone default, else e.g. en-US
  bool repliesOnGlasses = true;
  int autoScrollSeconds = 0; // 0 = page by hand
  Target target = Target.claude;
  String terminalAgent = 'claude';
  bool volumeKeysScroll = false;
  bool holdToTalk = true; // glasses long press = talk to Claude
  Map<String, String>? lastGlasses;

  static const autoOffChoices = [2, 5, 10, 15, 30, 60];

  void load(Map<String, Object?> m) {
    bridgeUrl = _s(m['bridgeUrl'], bridgeUrl);
    startCommand = _s(m['startCommand'], startCommand);
    final words = _s(m['wakeWords'], '');
    if (words.trim().isNotEmpty) {
      wakeWords = words.split(',').map((w) => w.trim()).where((w) => w.isNotEmpty).toList();
    }
    autoOffMinutes = _i(m['autoOffMinutes'], autoOffMinutes).clamp(1, 240);
    quietChime = _b(m['quietChime'], quietChime);
    recognizer = _s(m['recognizer'], recognizer);
    language = _s(m['language'], language);
    repliesOnGlasses = _b(m['repliesOnGlasses'], repliesOnGlasses);
    autoScrollSeconds = _i(m['autoScrollSeconds'], autoScrollSeconds).clamp(0, 60);
    target = _s(m['target'], target.name) == 'terminal' ? Target.terminal : Target.claude;
    terminalAgent = _s(m['terminalAgent'], terminalAgent);
    volumeKeysScroll = _b(m['volumeKeysScroll'], volumeKeysScroll);
    holdToTalk = _b(m['holdToTalk'], holdToTalk);
    final g = _s(m['lastGlasses'], '');
    if (g.isNotEmpty) {
      final parts = g.split('|');
      if (parts.length == 5) {
        lastGlasses = {
          'channelNumber': parts[0],
          'leftDeviceName': parts[1],
          'rightDeviceName': parts[2],
          'leftAddress': parts[3],
          'rightAddress': parts[4],
        };
      }
    }
    notifyListeners();
  }

  void set(String key, Object value) {
    switch (key) {
      case 'bridgeUrl':
        bridgeUrl = value as String;
      case 'startCommand':
        startCommand = value as String;
      case 'wakeWords':
        wakeWords = (value as List<String>);
        value = wakeWords.join(',');
      case 'autoOffMinutes':
        autoOffMinutes = value as int;
      case 'quietChime':
        quietChime = value as bool;
      case 'recognizer':
        recognizer = value as String;
      case 'language':
        language = value as String;
      case 'repliesOnGlasses':
        repliesOnGlasses = value as bool;
      case 'autoScrollSeconds':
        autoScrollSeconds = value as int;
      case 'target':
        target = value as Target;
        value = target.name;
      case 'terminalAgent':
        terminalAgent = value as String;
      case 'volumeKeysScroll':
        volumeKeysScroll = value as bool;
      case 'holdToTalk':
        holdToTalk = value as bool;
      case 'lastGlasses':
        final g = value as Map<String, String>;
        lastGlasses = g;
        value = [
          g['channelNumber'], g['leftDeviceName'], g['rightDeviceName'],
          g['leftAddress'], g['rightAddress'],
        ].map((s) => (s ?? '').replaceAll('|', '')).join('|');
      default:
        throw ArgumentError('Unknown setting $key');
    }
    _save?.call(key, value);
    notifyListeners();
  }

  static String _s(Object? v, String d) => v is String ? v : d;
  static int _i(Object? v, int d) => v is num ? v.toInt() : d;
  static bool _b(Object? v, bool d) => v is bool ? v : d;
}
