import 'package:flutter/foundation.dart';

import 'home/voice_monkey.dart';
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
  int pageHoldMs = 0; // extra time each teleprompter page stays up
  int pageSec = 0; // every page stays this many seconds (0 = by reading speed)
  int firstPageMs = 600; // extra time on an answer's first page
  int doneBlinkMs = 3000; // how long the "*" shows when Claude is done
  int termShowSec = 30; // after you send to Termux: its screen on the glasses this long after it last changed (0 = answer like chat, -1 = until you close it)
  int readingWpm = 180; // teleprompter speed for replies; 0 = page by hand
  bool glassesMic = true; // TouchBar hold and follow-ups: the glasses' own mic (with Parakeet), their Even AI screen kept
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
  String answerSpeed = 'quick'; // quick (no tools, hands off to Claude Code when needed) or full
  String quickModel = 'sonnet'; // the quick lane's model: sonnet or haiku
  String agent = 'claude'; // who answers: claude, or agy (Google Antigravity - for when Claude has run out)
  bool agyCommands = false; // Antigravity may run commands (it cannot ask first, so this is a plain yes)
  int touchPauseMs = 900; // TouchBar / double tap: this much quiet sends what you said
  bool bleParallel = false; // send text screens to both arms at once (test)
  int displayHeight = 4; // where text sits: 0 (low) .. 8 (high), as last set from the app
  int displayDepth = 5; // 1 (near) .. 9 (far)
  // Talking back and forth
  String speakReplies = 'voice'; // off, voice (answers to spoken questions), always
  String ttsEngine = ''; // text-to-speech app ('' = the phone's default)
  String ttsVoice = ''; // '' = the engine's default voice
  int ttsRate = 100; // percent
  int ttsPitch = 100; // percent
  String talkMode = 'single'; // single: one question, one answer. chat: back and forth (listens for a follow-up)

  // Lights (Voice Monkey) and the Vizio TV
  String vmToken = ''; // Voice Monkey token, or the whole trigger URL from its dashboard
  String vmDevices = VoiceMonkey.defaultDevices; // one per line: Name = device: phrase, phrase
  String tvIp = ''; // the Vizio TV on the Wi-Fi
  String tvName = '';
  String tvAuth = ''; // from pairing (the TV showed a PIN)
  String tvDeviceId = '';
  bool triggerWords = true; // hands-free: timers, lights and the TV without the wake word
  int followUpSec = 8;
  bool autoSend = true; // send when you pause (mic button, TouchBar, follow-ups)
  bool wakeAutoSend = false; // same after "Claude…" (off: say "send")
  int pauseMs = 1500; // the pause that ends what you say
  String wakeStrict = 'normal'; // relaxed, normal, strict
  bool wordsWhileHolding = false; // while the TouchBar is held: our listening screen and your words instead of Even AI's
  bool aiStyle = false; // Even AI screen states (the glasses animate every update); plain text when off
  bool doneBlink = true; // a tiny "*" on the glasses for 3 s when Claude is done
  // Look up: 'custom' = your own layout (drawn by the app), 'dashboard' = the glasses' own
  String lookMode = 'custom';
  String lookFrame = 'hud'; // HUD outline drawn around your layout's text: hud, corners, underline, off
  String lookLayout = ''; // LookLayout JSON ('' = the Split preset)
  // TouchBar: what a hold and a double tap do
  String holdAction = 'mode'; // mode (chat or terminal, whichever is on), chat, terminal
  String doubleTapAction = 'talk'; // talk (no Even AI screen), close, switch (chat <-> terminal), repeat
  String speechEngine = 'android'; // android (SpeechRecognizer) or parakeet (on the phone, downloaded)
  bool phoneActions = true; // Claude (and quick commands) may set reminders, open apps, ...
  // Work: when Claude Code finishes a job
  bool workAlerts = true;
  int workMinSec = 30; // only jobs that took at least this long
  bool workPhone = true; // phone notification
  bool workGlasses = true; // short notification on the glasses
  // Look up (the glasses dashboard)
  String dashMode = 'dual'; // full, dual, minimal (Claude on the right always uses dual)
  String dashRight = 'claude'; // right side: claude (live panel), notes, calendar, news, stocks, map
  String panelLayout = 'asked:1,claude,meta:0'; // Claude's panel: block order, lines for your question / the project
  String screenLayout = 'meta:1,asked:1,claude'; // the screen in front of you while Claude works
  int closeAfterSec = 15; // Claude's screen goes back to the dashboard after this long without changes (0 = never)
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
    pageHoldMs = _i(m['pageHoldMs'], pageHoldMs).clamp(0, 6000);
    pageSec = _i(m['pageSec'], pageSec).clamp(0, 60);
    firstPageMs = _i(m['firstPageMs'], firstPageMs).clamp(0, 8000);
    doneBlinkMs = _i(m['doneBlinkMs'], doneBlinkMs).clamp(500, 15000);
    termShowSec = _i(m['termShowSec'], termShowSec).clamp(-1, 600);
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
    target = _s(m['target'], target.name) == 'terminal' ? Target.terminal : Target.claude;
    terminalAgent = _s(m['terminalAgent'], terminalAgent);
    volumeKeysScroll = _b(m['volumeKeysScroll'], volumeKeysScroll);
    holdToTalk = _b(m['holdToTalk'], holdToTalk);
    claudeModel = _s(m['claudeModel'], claudeModel);
    claudeEffort = _s(m['claudeEffort'], claudeEffort);
    claudeAccess = _s(m['claudeAccess'], claudeAccess);
    answerLength = _s(m['answerLength'], answerLength);
    answerSpeed = _s(m['answerSpeed'], answerSpeed);
    quickModel = _s(m['quickModel'], quickModel);
    agent = _s(m['agent'], agent) == 'agy' ? 'agy' : 'claude';
    agyCommands = _b(m['agyCommands'], agyCommands);
    touchPauseMs = _i(m['touchPauseMs'], touchPauseMs).clamp(500, 3000);
    bleParallel = _b(m['bleParallel'], bleParallel);
    displayHeight = _i(m['displayHeight'], displayHeight).clamp(0, 8);
    displayDepth = _i(m['displayDepth'], displayDepth).clamp(1, 9);
    speakReplies = _s(m['speakReplies'], speakReplies);
    ttsEngine = _s(m['ttsEngine'], ttsEngine);
    ttsVoice = _s(m['ttsVoice'], ttsVoice);
    ttsRate = _i(m['ttsRate'], ttsRate).clamp(50, 200);
    ttsPitch = _i(m['ttsPitch'], ttsPitch).clamp(50, 200);
    talkMode = _s(m['talkMode'], talkMode);
    vmToken = _s(m['vmToken'], vmToken);
    vmDevices = _s(m['vmDevices'], vmDevices);
    tvIp = _s(m['tvIp'], tvIp);
    tvName = _s(m['tvName'], tvName);
    tvAuth = _s(m['tvAuth'], tvAuth);
    tvDeviceId = _s(m['tvDeviceId'], tvDeviceId);
    triggerWords = _b(m['triggerWords'], triggerWords);
    if (talkMode != 'chat') talkMode = 'single';
    followUpSec = _i(m['followUpSec'], followUpSec).clamp(3, 30);
    autoSend = _b(m['autoSend'], autoSend);
    wakeAutoSend = _b(m['wakeAutoSend'], wakeAutoSend);
    pauseMs = _i(m['pauseMs'], pauseMs).clamp(700, 4000);
    wakeStrict = _s(m['wakeStrict'], wakeStrict);
    wordsWhileHolding = _b(m['wordsWhileHolding'], wordsWhileHolding);
    aiStyle = _b(m['aiStyle'], aiStyle);
    doneBlink = _b(m['doneBlink'], doneBlink);
    lookMode = _s(m['lookMode'], lookMode);
    lookFrame = _s(m['lookFrame'], lookFrame);
    if (!const ['hud', 'corners', 'underline', 'off'].contains(lookFrame)) lookFrame = 'hud';
    lookLayout = _s(m['lookLayout'], lookLayout);
    holdAction = _s(m['holdAction'], holdAction);
    doubleTapAction = _s(m['doubleTapAction'], doubleTapAction);
    speechEngine = _s(m['speechEngine'], speechEngine);
    phoneActions = _b(m['phoneActions'], phoneActions);
    workAlerts = _b(m['workAlerts'], workAlerts);
    workMinSec = _i(m['workMinSec'], workMinSec).clamp(0, 3600);
    workPhone = _b(m['workPhone'], workPhone);
    workGlasses = _b(m['workGlasses'], workGlasses);
    dashMode = _s(m['dashMode'], dashMode);
    dashRight = _s(m['dashRight'], dashRight);
    panelLayout = _s(m['panelLayout'], panelLayout);
    screenLayout = _s(m['screenLayout'], screenLayout);
    closeAfterSec = _i(m['closeAfterSec'], closeAfterSec).clamp(0, 600);
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
      case 'pageHoldMs':
        pageHoldMs = value as int;
      case 'pageSec':
        pageSec = value as int;
      case 'firstPageMs':
        firstPageMs = value as int;
      case 'doneBlinkMs':
        doneBlinkMs = value as int;
      case 'termShowSec':
        termShowSec = value as int;
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
      case 'answerSpeed':
        answerSpeed = value as String;
      case 'quickModel':
        quickModel = value as String;
      case 'agent':
        agent = value == 'agy' ? 'agy' : 'claude';
      case 'agyCommands':
        agyCommands = value as bool;
      case 'touchPauseMs':
        touchPauseMs = value as int;
      case 'bleParallel':
        bleParallel = value as bool;
      case 'displayHeight':
        displayHeight = value as int;
      case 'displayDepth':
        displayDepth = value as int;
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
      case 'talkMode':
        talkMode = value as String;
      case 'vmToken':
        vmToken = value as String;
      case 'vmDevices':
        vmDevices = value as String;
      case 'tvIp':
        tvIp = value as String;
      case 'tvName':
        tvName = value as String;
      case 'tvAuth':
        tvAuth = value as String;
      case 'tvDeviceId':
        tvDeviceId = value as String;
      case 'triggerWords':
        triggerWords = value as bool;
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
      case 'aiStyle':
        aiStyle = value as bool;
      case 'doneBlink':
        doneBlink = value as bool;
      case 'lookMode':
        lookMode = value as String;
      case 'lookFrame':
        lookFrame = value as String;
      case 'lookLayout':
        lookLayout = value as String;
      case 'holdAction':
        holdAction = value as String;
      case 'doubleTapAction':
        doubleTapAction = value as String;
      case 'speechEngine':
        speechEngine = value as String;
      case 'phoneActions':
        phoneActions = value as bool;
      case 'workAlerts':
        workAlerts = value as bool;
      case 'workMinSec':
        workMinSec = value as int;
      case 'workPhone':
        workPhone = value as bool;
      case 'workGlasses':
        workGlasses = value as bool;
      case 'dashMode':
        dashMode = value as String;
      case 'dashRight':
        dashRight = value as String;
      case 'panelLayout':
        panelLayout = value as String;
      case 'screenLayout':
        screenLayout = value as String;
      case 'closeAfterSec':
        closeAfterSec = value as int;
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
    // Slider, drag and text keys change many times a second; the screen
    // logs record the result.
    const quiet = {
      'lastGlasses', 'mirrorX', 'mirrorY', 'mirrorW', 'mirrorH', 'mirrorStrip', //
      'ttsRate', 'ttsPitch', 'dashCustom', 'dashCity', 'lookLayout',
      'vmToken', 'tvAuth', 'vmDevices', // secrets (and a long list) stay out of the log
    };
    if (!quiet.contains(key)) EventLog.add('app', 'setting', {'key': key, 'value': '$value'});
    notifyListeners();
  }

  static String _s(Object? v, String d) => v is String ? v : d;
  static int _i(Object? v, int d) => v is num ? v.toInt() : d;
  static bool _b(Object? v, bool d) => v is bool ? v : d;
}
