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
  /// The one-word Termux command (termux/even): starts Agent Hub and Claude
  /// Code in Ubuntu in the background; `even stop`, `even restart`, `even log`.
  static const defaultStartCommand = 'even';

  /// What the app used to run before `even` existed.
  static const legacyStartCommand =
      "proot-distro login ubuntu -- env HUB_NO_BROWSER=1 bash -lc '~/agent-hub/start.sh'";

  String bridgeUrl = defaultBridgeUrl;
  String startCommand = defaultStartCommand;
  /// "Start" (and what recognizers often hear instead) begins a hands-free
  /// message; "send" sends it.
  static const defaultWakeWords = ['start', 'tart', 'starts'];

  List<String> wakeWords = List.of(defaultWakeWords);
  int autoOffMinutes = 10;
  bool quietChime = true;
  String recognizer = 'auto'; // auto, ondevice, default
  String language = ''; // '' = phone default, else e.g. en-US
  bool repliesOnGlasses = true;
  int readingWpm = 180; // teleprompter speed for replies; 0 = page by hand
  bool glassesMic = false; // TouchBar hold listens through the glasses' mic (test: the phone mic is reliable)
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
  String flickMethod = 'live'; // live (state 1 per move) or apply (state 1, then 0)
  bool flickClear = true; // clear the text before each move (the glasses ignore moves while text is up)
  int flickClearMs = 80; // pause after clearing
  int flickRedrawMs = 150; // pause after a move (the glasses redraw there), before the text
  int flickPreviewMs = 300; // apply: time between state 1 and state 0
  bool flickWait = true; // wait for the glasses to answer each command
  String flickWhat = 'both'; // both, move, text
  String flickMoveTo = 'both'; // moves to both arms or the right arm only
  int flickAutoStop = 60; // seconds
  Target target = Target.claude;
  String terminalAgent = 'claude';
  bool volumeKeysScroll = false;
  bool holdToTalk = true; // glasses long press = talk to Claude
  Map<String, String>? lastGlasses;
  // Claude (Claude Code in Termux, through Agent Hub)
  String claudeModel = 'opus'; // opus, sonnet, haiku
  String claudeEffort = 'medium'; // low, medium, high, xhigh, max
  String claudeAccess = 'auto'; // read (only looks), edit (files), auto (can run commands)
  String answerLength = 'brief'; // brief (~50 characters), normal (~150), long (screens)
  // Talking back and forth
  String speakReplies = 'voice'; // off, voice (answers to spoken questions), always
  String ttsEngine = ''; // text-to-speech app ('' = the phone's default)
  String ttsVoice = ''; // '' = the engine's default voice
  int ttsRate = 100; // percent
  int ttsPitch = 100; // percent
  bool conversation = true; // after an answer, listen for a follow-up without the wake word
  int followUpSec = 8;
  bool autoSend = true; // send when you pause (mic button, TouchBar, follow-ups)
  bool wakeAutoSend = false; // same after "Claude…" (off: say "send")
  int pauseMs = 1500; // the pause that ends what you say
  String wakeStrict = 'normal'; // relaxed, normal, strict
  bool wordsWhileHolding = true; // while the TouchBar is held: "Termux-Claude listening" and your words
  // Work: when Claude Code finishes a job
  bool workAlerts = true;
  int workMinSec = 30; // only jobs that took at least this long
  bool workPhone = true; // phone notification
  bool workGlasses = true; // short notification on the glasses
  // Look up (the glasses dashboard)
  bool dashSync = true;
  String dashMode = 'dual'; // full, dual, minimal (Claude on the right always uses dual)
  String dashRight = 'claude'; // right side: claude (live panel), notes, calendar, news, stocks, map
  String panelLayout = 'asked:1,claude,meta:0'; // Claude's panel: block order, lines for your question / the project
  bool dash24h = false;
  bool dashFahrenheit = true;
  bool dashWeather = true;
  String dashCity = ''; // '' = the phone's location
  String dashNotes = 'work,reply,custom'; // up to 4: work, reply, phone, custom, off
  String dashCustom = 'Note\nAnything you want to see when you look up.';
  // Phone notifications on the glasses
  bool notifyForward = false;
  bool notifyPrivate = false; // title only, no message text
  String notifyApps = ''; // allowed packages, comma separated ('' = all)

  static const autoOffChoices = [2, 5, 10, 15, 30, 60];

  /// Model, effort and access for Agent Hub.
  Map<String, String> get claudeOptions => {'model': claudeModel, 'effort': claudeEffort, 'access': claudeAccess};

  WakeStrictness get wakeStrictness =>
      WakeStrictness.values.firstWhere((w) => w.name == wakeStrict, orElse: () => WakeStrictness.normal);

  /// Look-up screen note slots (up to 4).
  List<String> get noteSlots {
    final list = dashNotes.split(',').map((x) => x.trim()).where((x) => x.isNotEmpty).toList();
    while (list.length < 4) {
      list.add('off');
    }
    return list.take(4).toList();
  }

  Set<String> get allowedApps => notifyApps.split(',').map((x) => x.trim()).where((x) => x.isNotEmpty).toSet();

  void load(Map<String, Object?> m) {
    bridgeUrl = _s(m['bridgeUrl'], bridgeUrl);
    startCommand = _s(m['startCommand'], startCommand);
    if (startCommand == legacyStartCommand) startCommand = defaultStartCommand;
    final words = _s(m['wakeWords'], '');
    if (words.trim().isNotEmpty && words != WakeWord.defaults.join(',')) {
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
    flickMethod = _s(m['flickMethod'], flickMethod);
    flickClear = _b(m['flickClear'], flickClear);
    flickClearMs = _i(m['flickClearMs'], flickClearMs).clamp(0, 2000);
    flickRedrawMs = _i(m['flickRedrawMs'], flickRedrawMs).clamp(0, 2000);
    flickPreviewMs = _i(m['flickPreviewMs'], flickPreviewMs).clamp(0, 3000);
    flickWait = _b(m['flickWait'], flickWait);
    flickWhat = _s(m['flickWhat'], flickWhat);
    flickMoveTo = _s(m['flickMoveTo'], flickMoveTo);
    flickAutoStop = _i(m['flickAutoStop'], flickAutoStop).clamp(5, 600);
    target = _s(m['target'], target.name) == 'terminal' ? Target.terminal : Target.claude;
    terminalAgent = _s(m['terminalAgent'], terminalAgent);
    volumeKeysScroll = _b(m['volumeKeysScroll'], volumeKeysScroll);
    holdToTalk = _b(m['holdToTalk'], holdToTalk);
    claudeModel = _s(m['claudeModel'], claudeModel);
    claudeEffort = _s(m['claudeEffort'], claudeEffort);
    claudeAccess = _s(m['claudeAccess'], claudeAccess);
    answerLength = _s(m['answerLength'], answerLength);
    speakReplies = _s(m['speakReplies'], speakReplies);
    ttsEngine = _s(m['ttsEngine'], ttsEngine);
    ttsVoice = _s(m['ttsVoice'], ttsVoice);
    ttsRate = _i(m['ttsRate'], ttsRate).clamp(50, 200);
    ttsPitch = _i(m['ttsPitch'], ttsPitch).clamp(50, 200);
    conversation = _b(m['conversation'], conversation);
    followUpSec = _i(m['followUpSec'], followUpSec).clamp(3, 30);
    autoSend = _b(m['autoSend'], autoSend);
    wakeAutoSend = _b(m['wakeAutoSend'], wakeAutoSend);
    pauseMs = _i(m['pauseMs'], pauseMs).clamp(700, 4000);
    wakeStrict = _s(m['wakeStrict'], wakeStrict);
    wordsWhileHolding = _b(m['wordsWhileHolding'], wordsWhileHolding);
    workAlerts = _b(m['workAlerts'], workAlerts);
    workMinSec = _i(m['workMinSec'], workMinSec).clamp(0, 3600);
    workPhone = _b(m['workPhone'], workPhone);
    workGlasses = _b(m['workGlasses'], workGlasses);
    dashSync = _b(m['dashSync'], dashSync);
    dashMode = _s(m['dashMode'], dashMode);
    dashRight = _s(m['dashRight'], dashRight);
    panelLayout = _s(m['panelLayout'], panelLayout);
    dash24h = _b(m['dash24h'], dash24h);
    dashFahrenheit = _b(m['dashFahrenheit'], dashFahrenheit);
    dashWeather = _b(m['dashWeather'], dashWeather);
    dashCity = _s(m['dashCity'], dashCity);
    dashNotes = _s(m['dashNotes'], dashNotes);
    dashCustom = _s(m['dashCustom'], dashCustom);
    notifyForward = _b(m['notifyForward'], notifyForward);
    notifyPrivate = _b(m['notifyPrivate'], notifyPrivate);
    notifyApps = _s(m['notifyApps'], notifyApps);
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
      case 'flickMethod':
        flickMethod = value as String;
      case 'flickClear':
        flickClear = value as bool;
      case 'flickClearMs':
        flickClearMs = value as int;
      case 'flickRedrawMs':
        flickRedrawMs = value as int;
      case 'flickPreviewMs':
        flickPreviewMs = value as int;
      case 'flickWait':
        flickWait = value as bool;
      case 'flickWhat':
        flickWhat = value as String;
      case 'flickMoveTo':
        flickMoveTo = value as String;
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
      case 'claudeModel':
        claudeModel = value as String;
      case 'claudeEffort':
        claudeEffort = value as String;
      case 'claudeAccess':
        claudeAccess = value as String;
      case 'answerLength':
        answerLength = value as String;
      case 'speakReplies':
        speakReplies = value as String;
      case 'ttsEngine':
        ttsEngine = value as String;
      case 'ttsVoice':
        ttsVoice = value as String;
      case 'ttsRate':
        ttsRate = value as int;
      case 'ttsPitch':
        ttsPitch = value as int;
      case 'conversation':
        conversation = value as bool;
      case 'followUpSec':
        followUpSec = value as int;
      case 'autoSend':
        autoSend = value as bool;
      case 'wakeAutoSend':
        wakeAutoSend = value as bool;
      case 'pauseMs':
        pauseMs = value as int;
      case 'wakeStrict':
        wakeStrict = value as String;
      case 'wordsWhileHolding':
        wordsWhileHolding = value as bool;
      case 'workAlerts':
        workAlerts = value as bool;
      case 'workMinSec':
        workMinSec = value as int;
      case 'workPhone':
        workPhone = value as bool;
      case 'workGlasses':
        workGlasses = value as bool;
      case 'dashSync':
        dashSync = value as bool;
      case 'dashMode':
        dashMode = value as String;
      case 'dashRight':
        dashRight = value as String;
      case 'panelLayout':
        panelLayout = value as String;
      case 'dash24h':
        dash24h = value as bool;
      case 'dashFahrenheit':
        dashFahrenheit = value as bool;
      case 'dashWeather':
        dashWeather = value as bool;
      case 'dashCity':
        dashCity = value as String;
      case 'dashNotes':
        dashNotes = value as String;
      case 'dashCustom':
        dashCustom = value as String;
      case 'notifyForward':
        notifyForward = value as bool;
      case 'notifyPrivate':
        notifyPrivate = value as bool;
      case 'notifyApps':
        notifyApps = value as String;
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
      'flickDepth', 'flickHomePos', 'flickHomeDepth', 'flickClearMs', 'flickRedrawMs', 'flickPreviewMs',
      'ttsRate', 'ttsPitch', 'dashCustom', 'dashCity',
    };
    if (!quiet.contains(key)) EventLog.add('app', 'setting', {'key': key, 'value': '$value'});
    notifyListeners();
  }

  static String _s(Object? v, String d) => v is String ? v : d;
  static int _i(Object? v, int d) => v is num ? v.toInt() : d;
  static bool _b(Object? v, bool d) => v is bool ? v : d;
}
