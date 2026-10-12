import 'package:flutter/foundation.dart';

import 'log.dart';
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
  int readingWpm = 180; // teleprompter speed for replies; 0 = page by hand
  bool glassesMic = true; // TouchBar hold listens through the glasses' mic
  bool diagnostics = true; // timing log uploaded to Agent Hub (~/.agent-hub/logs)
  int bmpGapMs = 5; // gap between picture packets per arm (tuning)
  int bmpChunk = 194; // picture bytes per packet: 194 (demo) or 236 (tuning)
  bool ble2m = true; // Bluetooth 5 2M radio mode (applies on the next connect)
  // Phone screen on the glasses
  String mirrorRegion = 'strip'; // strip (readable band), top (top half), full
  int mirrorStrip = 0; // strip position, 0 (top) .. 100 (bottom)
  int mirrorFpsTenths = 20; // capture rate (20 = 2 a second; sending paces itself to the glasses)
  String mirrorInvert = 'auto'; // auto, on, off
  String mirrorStyle = 'text'; // text (sharp) or photo (dithered)
  int mirrorX = 0, mirrorY = 20, mirrorW = 100, mirrorH = 30; // custom area, % of screen
  String mirrorFit = 'fit'; // fit, fill, stretch
  int mirrorRows = 68; // picture height: 68 (half, faster - the default), 102, 136 (full)
  bool mirrorFast = false; // experimental: verify the previous picture (one transfer per frame)
  // Flicker: text at a high and a low spot in your view, switching on a timer
  String flickTop = 'TOP\nThis text sits high in your view';
  String flickBottom = 'BOTTOM\nThis text sits low in your view';
  int flickTopMs = 100, flickBottomMs = 100; // time at each spot
  int flickTopPos = 8, flickBottomPos = 0; // display height 0 (low) .. 8 (high)
  int flickDepth = 3; // display depth 1 .. 9
  int flickHomePos = 7, flickHomeDepth = 3; // normal spot, restored at stop
  bool flickPreview = false; // send moves as "preview" instead of "keep"
  bool flickWait = true; // wait for the glasses to answer each command
  String flickWhat = 'both'; // both, move, text
  String flickOrder = 'text'; // text (text first), move (move first), together
  String flickArms = 'both'; // moves to both arms or R only
  int flickAutoStop = 60; // seconds
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
    readingWpm = _i(m['readingWpm'], readingWpm).clamp(0, 400);
    glassesMic = _b(m['glassesMic'], glassesMic);
    diagnostics = _b(m['diagnostics'], diagnostics);
    bmpGapMs = _i(m['bmpGapMs'], bmpGapMs).clamp(0, 30);
    bmpChunk = _i(m['bmpChunk'], bmpChunk).clamp(100, 236);
    ble2m = _b(m['ble2m'], ble2m);
    mirrorRegion = _s(m['mirrorRegion'], mirrorRegion);
    mirrorStrip = _i(m['mirrorStrip'], mirrorStrip).clamp(0, 100);
    mirrorFpsTenths = _i(m['mirrorFpsTenths'], mirrorFpsTenths).clamp(2, 50);
    mirrorInvert = _s(m['mirrorInvert'], mirrorInvert);
    mirrorStyle = _s(m['mirrorStyle'], mirrorStyle);
    mirrorX = _i(m['mirrorX'], mirrorX).clamp(0, 99);
    mirrorY = _i(m['mirrorY'], mirrorY).clamp(0, 99);
    mirrorW = _i(m['mirrorW'], mirrorW).clamp(1, 100);
    mirrorH = _i(m['mirrorH'], mirrorH).clamp(1, 100);
    mirrorFit = _s(m['mirrorFit'], mirrorFit);
    mirrorRows = _i(m['mirrorRows'], mirrorRows).clamp(16, 136);
    mirrorFast = _b(m['mirrorFast'], mirrorFast);
    flickTop = _s(m['flickTop'], flickTop);
    flickBottom = _s(m['flickBottom'], flickBottom);
    flickTopMs = _i(m['flickTopMs'], flickTopMs).clamp(20, 5000);
    flickBottomMs = _i(m['flickBottomMs'], flickBottomMs).clamp(20, 5000);
    flickTopPos = _i(m['flickTopPos'], flickTopPos).clamp(0, 8);
    flickBottomPos = _i(m['flickBottomPos'], flickBottomPos).clamp(0, 8);
    flickDepth = _i(m['flickDepth'], flickDepth).clamp(1, 9);
    flickHomePos = _i(m['flickHomePos'], flickHomePos).clamp(0, 8);
    flickHomeDepth = _i(m['flickHomeDepth'], flickHomeDepth).clamp(1, 9);
    flickPreview = _b(m['flickPreview'], flickPreview);
    flickWait = _b(m['flickWait'], flickWait);
    flickWhat = _s(m['flickWhat'], flickWhat);
    flickOrder = _s(m['flickOrder'], flickOrder);
    flickArms = _s(m['flickArms'], flickArms);
    flickAutoStop = _i(m['flickAutoStop'], flickAutoStop).clamp(5, 600);
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
      case 'readingWpm':
        readingWpm = value as int;
      case 'glassesMic':
        glassesMic = value as bool;
      case 'diagnostics':
        diagnostics = value as bool;
      case 'bmpGapMs':
        bmpGapMs = value as int;
      case 'bmpChunk':
        bmpChunk = value as int;
      case 'ble2m':
        ble2m = value as bool;
      case 'mirrorRegion':
        mirrorRegion = value as String;
      case 'mirrorStrip':
        mirrorStrip = value as int;
      case 'mirrorFpsTenths':
        mirrorFpsTenths = value as int;
      case 'mirrorInvert':
        mirrorInvert = value as String;
      case 'mirrorStyle':
        mirrorStyle = value as String;
      case 'mirrorX':
        mirrorX = value as int;
      case 'mirrorY':
        mirrorY = value as int;
      case 'mirrorW':
        mirrorW = value as int;
      case 'mirrorH':
        mirrorH = value as int;
      case 'mirrorFit':
        mirrorFit = value as String;
      case 'mirrorRows':
        mirrorRows = value as int;
      case 'mirrorFast':
        mirrorFast = value as bool;
      case 'flickTop':
        flickTop = value as String;
      case 'flickBottom':
        flickBottom = value as String;
      case 'flickTopMs':
        flickTopMs = value as int;
      case 'flickBottomMs':
        flickBottomMs = value as int;
      case 'flickTopPos':
        flickTopPos = value as int;
      case 'flickBottomPos':
        flickBottomPos = value as int;
      case 'flickDepth':
        flickDepth = value as int;
      case 'flickHomePos':
        flickHomePos = value as int;
      case 'flickHomeDepth':
        flickHomeDepth = value as int;
      case 'flickPreview':
        flickPreview = value as bool;
      case 'flickWait':
        flickWait = value as bool;
      case 'flickWhat':
        flickWhat = value as String;
      case 'flickOrder':
        flickOrder = value as String;
      case 'flickArms':
        flickArms = value as String;
      case 'flickAutoStop':
        flickAutoStop = value as int;
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
    // Slider and text keys change many times a second; the screen and flicker
    // logs record the result.
    const quiet = {
      'lastGlasses', 'mirrorX', 'mirrorY', 'mirrorW', 'mirrorH', 'mirrorStrip', //
      'flickTop', 'flickBottom', 'flickTopMs', 'flickBottomMs', 'flickTopPos', 'flickBottomPos',
      'flickDepth', 'flickHomePos', 'flickHomeDepth',
    };
    if (!quiet.contains(key)) EventLog.add('app', 'setting', {'key': key, 'value': '$value'});
    notifyListeners();
  }

  static String _s(Object? v, String d) => v is String ? v : d;
  static int _i(Object? v, int d) => v is num ? v.toInt() : d;
  static bool _b(Object? v, bool d) => v is bool ? v : d;
}
