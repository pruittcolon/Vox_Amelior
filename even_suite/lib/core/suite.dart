import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'ble/ble_manager.dart';
import 'ble/ble_receive.dart';
import 'ble/proto.dart';
import 'bridge/bridge_client.dart';
import 'bridge/chat_controller.dart';
import 'bridge/terminal_controller.dart';
import 'bridge/work_controller.dart';
import 'glasses/claude_panel.dart';
import 'glasses/dashboard.dart';
import 'glasses/display_position.dart';
import 'glasses/flicker.dart';
import 'glasses/glasses_display.dart';
import 'glasses/glasses_text.dart';
import 'glasses/look_layout.dart';
import 'log.dart';
import 'phone_actions.dart';
import 'phone_notifications.dart';
import 'screen/picture_lab.dart';
import 'screen/screen_mirror.dart';
import 'settings.dart';
import 'system.dart';
import 'talk/conversation.dart';
import 'voice/parakeet_model.dart';
import 'voice/parakeet_speech.dart';
import 'voice/speaker.dart';
import 'voice/speech.dart';
import 'voice/voice_commands.dart';
import 'voice/voice_controller.dart';
import 'voice/wake_word.dart';

/// Wires the app together: glasses, voice, Claude, Termux, settings.
class Suite extends ChangeNotifier {
  Suite({
    required this.system,
    required BleTransport transport,
    BleControl? control,
    required SpeechEngine speech,
    BridgeClient? client,
    ScreenSource? screen,
    Speaker? speaker,
    WeatherService? weather,
    SpeechModel? speechModel,
  }) : _speech = speech {
    settings = Settings(system.setPref);
    this.speechModel = speechModel ?? SpeechModel(system.filesDir);
    this.speechModel.addListener(_applySpeech);
    ble = BleManager(transport, control: control);
    proto = Proto(ble);
    // Screens go out as fast as the glasses take them (one in flight, the
    // newest waiting), so what you say shows up without lag.
    display = GlassesDisplay(proto, liveGap: const Duration(milliseconds: 150));
    // The glasses answer every exit we send with a "double tap" (F5 00): not you.
    proto.onExit = _expectExitEcho;
    this.client = client ?? BridgeClient(settings.bridgeUrl);
    chat = ChatController(this.client);
    work = WorkController(this.client);
    terminal = TerminalController(this.client, display);
    mirror = ScreenMirror(screen ?? PlatformScreen(), proto)..onFallback = _toast;
    // Screen sharing owns the glasses while it runs: our screen (and its
    // auto-close, which would end the sharing) steps aside.
    mirror.addListener(() {
      if (mirror.running && display.mode != DisplayMode.idle) display.onGlassesExit();
    });
    this.speaker = speaker ?? PlatformSpeaker();
    narrator = Narrator(this.speaker);
    conversation = Conversation(display)..canShow = _glassesFree;
    // Claude's screen closes by itself, but not while you or the phone are talking.
    display.holdOpen = () => narrator.speaking || voice.dictating || conversation.busy;
    dashboard = Dashboard(proto, weatherService: weather ?? WeatherService(location: system.location))
      ..paused = () => mirror.running || flicker.running || display.flashing;
    phone = PhoneNotifications(proto)..onForwarded = dashboard.addPhoneNote;
    voice = VoiceController(
      speech,
      onSubmit: submit,
      onCommand: runCommand,
      onDraft: _showDraft,
      onAutoOff: () => _toast('Hands-free is off (no use for ${settings.autoOffMinutes} min)'),
    )
      ..onFollowUpEnd = _onFollowUpEnd
      ..onNothing = _onNothingHeard
      ..onBargeIn = _onBargeIn;
    voice.spokenText = () => narrator.recent;

    ble.onConnected = (pair) {
      settings.set('lastGlasses', pair.toMap());
      _toast('Glasses connected');
    };
    ble.addListener(_onLinkChanged);
    voice.addListener(_onVoiceChanged);
    voice.level.addListener(() => conversation.setLevel(voice.level.value));
    chat.onReply = _onReply;
    chat.onProgress = _onProgress;
    work.onDone = _onWorkDone;
    work.onFinished = _onJobFinished;
    work.onActive = (jobs) {
      dashboard.jobs = jobs;
      dashboard.workChanged();
      _followTerminalJob(jobs);
    };
    conversation.addListener(_onTalkChanged);
    narrator.onDone = _onSpoken;
    _events = ble.events.listen(_onGlassesEvent);
    _phoneNotes = system.phoneNotifications.listen((m) => phone.onPosted(PhoneNote.fromMap(m)));
    system.onStopListening = () => voice.setHandsFree(false);
    system.onReminder = _onReminder;
    system.onOpenTab = tabRequests.add;
    system.onVolumeKey = (dir) {
      if (dir == 'up') {
        display.up();
      } else {
        display.down();
      }
    };
  }

  /// Real phone wiring.
  factory Suite.platform() {
    final ble = PlatformBle();
    final system = SystemBridge()..init();
    final model = SpeechModel(system.filesDir);
    return Suite(
        system: system,
        transport: ble,
        control: ble,
        speech: SwitchableSpeech(PlatformSpeech(), ParakeetSpeech(model)),
        speechModel: model,
        speaker: PlatformSpeaker());
  }

  final SystemBridge system;
  final SpeechEngine _speech;
  late final Settings settings;

  /// Parakeet's files (downloaded on the Speech engine page).
  late final SpeechModel speechModel;

  /// The Parakeet recognizer, when this phone can run it.
  ParakeetSpeech? get parakeet {
    final s = _speech;
    if (s is SwitchableSpeech && s.parakeet is ParakeetSpeech) return s.parakeet as ParakeetSpeech;
    return null;
  }

  /// Parakeet is the recognizer in use now.
  bool get usingParakeet {
    final s = _speech;
    return s is SwitchableSpeech && s.useParakeet;
  }
  late final BleManager ble;
  late final Proto proto;
  late final GlassesDisplay display;
  late final BridgeClient client;
  late final ChatController chat;
  late final WorkController work;
  late final TerminalController terminal;
  late final Speaker speaker;
  late final Narrator narrator;
  late final Conversation conversation;
  late final Dashboard dashboard;
  late final PhoneNotifications phone;
  late final VoiceController voice;
  late final ScreenMirror mirror;
  late final PictureLab lab = PictureLab(proto);
  late final PhoneActions phoneActions = PhoneActions(system);

  /// Text height (the Text height page); nothing else draws while it is open.
  late final DisplayPosition position = DisplayPosition(proto);
  late final Flicker flicker = Flicker(proto);

  /// Flicker owns the glasses while it runs: screen sharing stops and the
  /// reply screen steps aside.
  Future<void> startFlicker() async {
    if (!ble.isConnected) {
      _toast('Connect your glasses first');
      return;
    }
    if (mirror.running) await mirror.stop(clearGlasses: false);
    display.onGlassesExit();
    await flicker.start();
  }

  /// Picture test (stops screen sharing first so nothing else is sending).
  Future<void> runPictureTest() async {
    if (!ble.isConnected) {
      _toast('Connect your glasses first');
      return;
    }
    if (mirror.running) await mirror.stop(clearGlasses: false);
    await lab.run();
    _toast('Picture test done - say "check the logs"');
  }

  StreamSubscription<GlassesEvent>? _events;
  StreamSubscription<Map>? _phoneNotes;
  Timer? _heartbeat;
  Timer? _liveCleanup;
  Timer? _holdLimit;
  Timer? _mirrorTimer;
  Timer? _terminalWatch;
  Timer? _lookTimer;
  Timer? _echoTimer;
  bool _exitEcho = false;
  DateTime? _lookSince;
  int? _battery;
  LookLayout? _lookCache;
  String _lookKey = '\u0000';

  /// The question on its way went to the quick Claude (no tools) or the full one.
  String _lane = 'full';

  /// The TouchBar hold talks to this (chat or terminal) instead of the
  /// current mode, for one message.
  Target? _sessionTarget;
  Target get _target => _sessionTarget ?? settings.target;
  bool _aiSession = false; // the current question came from the TouchBar
  bool _micOpen = false;
  bool _lastConnected = false;
  bool _lastHandsFree = false;
  DateTime? _lastAutoOffAt;
  bool _spokenQuestion = false; // the question on its way was spoken (a follow-up makes sense)
  bool _speakThis = false; // read this answer aloud
  String _lastQuestion = '';
  String _voiceKey = '';
  int _workNotice = 0;

  /// A line typed into Claude Code by voice: its steps and answer come back
  /// through the Work hooks and are shown like a chat answer.
  String? _terminalAsk;

  /// "start" (and your wake words) at the front of a terminal message.
  WakeWord _startWord = WakeWord(Settings.defaultWakeWords);
  String _shownJob = ''; // finished job already shown as an answer (no second popup)
  String _blinkedJob = ''; // finished job already told with the "*" (no popup on top)

  /// Short messages for the UI (shown as a snackbar).
  final toasts = StreamController<String>.broadcast();

  /// Tabs to switch to ('work' when a "Claude finished" notification is tapped).
  final tabRequests = StreamController<String>.broadcast();

  /// The glasses are free for Claude's screens (not sharing the phone screen,
  /// not running Flicker, replies switched on).
  bool _glassesFree() => ble.isConnected && settings.repliesOnGlasses && !mirror.running && !flicker.running;

  void _toast(String msg) => toasts.add(msg);

  // ---------------------------------------------------------------- start

  Future<void> start() async {
    settings.load(await system.prefs());
    applySettings();
    unawaited(speechModel.check());
    settings.addListener(applySettings);
    final log = EventLog.instance
      ..client = client
      ..nativeStats = system.nativeStats
      ..enabled = settings.diagnostics;
    final size = GlassesText.screenSize();
    EventLog.add('app', 'start', {
      'charsPerLine': size.charsPerLine,
      'readingWpm': settings.readingWpm,
      'glassesMic': settings.glassesMic,
      'recognizer': settings.recognizer,
    });
    unawaited(log.flush());
    work.start();
    unawaited(system.launchTab().then((t) {
      if (t != null && t.isNotEmpty) tabRequests.add(t);
    }));
    // Reconnect the glasses used last time, without scanning.
    final last = settings.lastGlasses;
    if (last != null) {
      final granted = await system.permissions(['bluetooth']);
      if (granted['bluetooth'] == true) await ble.connect(GlassesPair.fromMap(last));
    }
    unawaited(chat.check());
  }

  void applySettings() {
    if (client.baseUrl != settings.bridgeUrl) client.baseUrl = settings.bridgeUrl;
    _startWord = WakeWord({...Settings.defaultWakeWords, ...settings.wakeWords});
    voice
      ..wake = WakeWord(settings.wakeWords, settings.wakeStrictness)
      ..autoOff = Duration(minutes: settings.autoOffMinutes)
      ..quiet = settings.quietChime
      ..language = settings.language
      ..engineName = settings.recognizer
      // Typing into the terminal waits for "send" (or letting go of the TouchBar).
      ..autoSend = settings.autoSend && settings.target == Target.claude
      ..wakeAutoSend = settings.wakeAutoSend && settings.target == Target.claude
      ..silenceMs = settings.pauseMs;
    chat.options = _claudeOptions();
    narrator.enabled = settings.speakReplies != 'off';
    final voiceKey = '${settings.ttsEngine}|${settings.ttsVoice}|${settings.ttsRate}|${settings.ttsPitch}';
    if (voiceKey != _voiceKey) {
      _voiceKey = voiceKey;
      unawaited(speaker.configure(
          engine: settings.ttsEngine,
          voice: settings.ttsVoice,
          rate: settings.ttsRate / 100,
          pitch: settings.ttsPitch / 100));
    }
    work.minDuration = Duration(seconds: settings.workMinSec);
    conversation.layout = PanelLayout.parse(settings.screenLayout, base: PanelLayout.working);
    display.closeAfter = settings.closeAfterSec == 0 ? null : Duration(seconds: settings.closeAfterSec);
    display.aiStates = settings.aiStyle;
    final dashKey = '${dashboard.mode}|${dashboard.pane}|${dashboard.claudePanel}|${dashboard.layout}|${dashboard.h24}|${dashboard.fahrenheit}|'
        '${dashboard.weatherOn}|${dashboard.city}|${dashboard.slots}|${dashboard.custom}|${dashboard.enabled}';
    dashboard
      ..enabled = settings.dashSync
      ..mode = const ['full', 'dual', 'minimal'].indexOf(settings.dashMode).clamp(0, 2)
      ..pane = const ['notes', 'stocks', 'news', 'calendar', 'map'].indexOf(settings.dashRight).clamp(0, 4)
      ..claudePanel = settings.dashRight == 'claude'
      ..layout = PanelLayout.parse(settings.panelLayout)
      ..h24 = settings.dash24h
      ..fahrenheit = settings.dashFahrenheit
      ..weatherOn = settings.dashWeather
      ..city = settings.dashCity
      ..slots = settings.noteSlots
      ..custom = settings.dashCustom;
    final newDashKey = '${dashboard.mode}|${dashboard.pane}|${dashboard.claudePanel}|${dashboard.layout}|${dashboard.h24}|${dashboard.fahrenheit}|'
        '${dashboard.weatherOn}|${dashboard.city}|${dashboard.slots}|${dashboard.custom}|${dashboard.enabled}';
    if (newDashKey != dashKey && ble.isConnected) dashboard.markDirty(delay: const Duration(seconds: 3));
    phone
      ..forward = settings.notifyForward
      ..private = settings.notifyPrivate
      ..allowed = settings.allowedApps;
    display.holdMs = settings.pageHoldMs;
    if (display.wpm != settings.readingWpm) display.setWpm(settings.readingWpm);
    EventLog.instance.enabled = settings.diagnostics;
    if (proto.bmpGapMs != settings.bmpGapMs) proto.bmpGapMs = settings.bmpGapMs;
    proto.bmpChunk = settings.bmpChunk;
    final ctl = ble.control;
    if (ctl is PlatformBle) ctl.phy2m = settings.ble2m;
    final mirrorChanged = mirror.region != settings.mirrorRegion ||
        mirror.strip != settings.mirrorStrip ||
        mirror.fpsTenths != settings.mirrorFpsTenths ||
        mirror.style != settings.mirrorStyle ||
        mirror.invert != settings.mirrorInvert ||
        mirror.cx != settings.mirrorX ||
        mirror.cy != settings.mirrorY ||
        mirror.cw != settings.mirrorW ||
        mirror.ch != settings.mirrorH ||
        mirror.fit != settings.mirrorFit ||
        mirror.rows != settings.mirrorRows ||
        mirror.fastMode != settings.mirrorFast;
    mirror
      ..region = settings.mirrorRegion
      ..strip = settings.mirrorStrip
      ..fpsTenths = settings.mirrorFpsTenths
      ..style = settings.mirrorStyle
      ..invert = settings.mirrorInvert
      ..cx = settings.mirrorX
      ..cy = settings.mirrorY
      ..cw = settings.mirrorW
      ..ch = settings.mirrorH
      ..fit = settings.mirrorFit
      ..rows = settings.mirrorRows
      ..fastMode = settings.mirrorFast
      ..autoOff = Duration(minutes: settings.autoOffMinutes);
    // Sliders change settings many times a second: apply once they settle.
    if (mirrorChanged) {
      _mirrorTimer?.cancel();
      _mirrorTimer = Timer(const Duration(milliseconds: 400), mirror.reconfigure);
    }
    chat.system = glassesPrimer(quick: settings.answerSpeed == 'quick');
    proto.parallelText = settings.bleParallel;
    voice.stableAfter = Duration(milliseconds: settings.touchPauseMs);
    dashboard.notesOn = settings.lookMode != 'custom';
    flicker
      ..topText = settings.flickTop
      ..bottomText = settings.flickBottom
      ..topMs = settings.flickTopMs
      ..bottomMs = settings.flickBottomMs
      ..topHeight = settings.flickTopPos
      ..bottomHeight = settings.flickBottomPos
      ..depth = settings.flickDepth
      ..homeHeight = settings.flickHomePos
      ..homeDepth = settings.flickHomeDepth
      ..method = settings.flickMethod
      ..clearFirst = settings.flickClear
      ..clearMs = settings.flickClearMs
      ..settleMs = settings.flickRedrawMs
      ..previewMs = settings.flickPreviewMs
      ..waitForReplies = settings.flickWait
      ..what = settings.flickWhat
      ..arms = settings.flickMoveTo
      ..autoStopSec = settings.flickAutoStop
      ..changed();
    terminal.mirrorIdle = Duration(minutes: settings.autoOffMinutes);
    if (terminal.agent != settings.terminalAgent) terminal.setAgent(settings.terminalAgent);
    system.volumeKeys(settings.volumeKeysScroll);
    _applySpeech();
    notifyListeners();
  }

  /// Android's recognizer or Parakeet (once downloaded and picked).
  void _applySpeech() {
    final s = _speech;
    if (s is! SwitchableSpeech) return;
    final want = settings.speechEngine == 'parakeet' && speechModel.ready;
    if (want == s.useParakeet) return;
    s.useParakeet = want;
    EventLog.add('speech', 'engine', {'parakeet': want});
    final p = parakeet;
    if (p != null) {
      if (want) {
        unawaited(p.warm().catchError((Object e) => _toast('$e')));
      } else {
        unawaited(p.unload());
      }
    }
    unawaited(voice.reconnect());
  }

  // -------------------------------------------------------------- glasses

  Future<void> scan() async {
    final granted = await system.permissions(['bluetooth']);
    if (granted['bluetooth'] != true) {
      _toast('Allow “Nearby devices” so the app can find your glasses');
      return;
    }
    await ble.startScan();
  }

  void _onLinkChanged() {
    final connected = ble.isConnected;
    if (connected == _lastConnected) return;
    _lastConnected = connected;
    _heartbeat?.cancel();
    if (!connected && flicker.running) flicker.stop(reason: 'link', restore: false);
    if (connected) {
      dashboard.onConnected();
    } else {
      dashboard.onDisconnected();
      phone.onDisconnected();
    }
    if (connected) {
      // Same 8 s heartbeat as the demo keeps the link from idling out - but
      // not while other traffic (pictures, screens) already does, so it never
      // cuts into a transfer.
      _heartbeat = Timer.periodic(const Duration(seconds: 8), (_) {
        if (DateTime.now().difference(ble.lastReply) < const Duration(seconds: 6)) return;
        proto.sendHeartBeat();
      });
    }
    _updateService();
  }

  /// For a moment after we close a screen, you look down, or their
  /// dashboard closes, an F5 00 is the glasses reporting that - not a
  /// double tap (the logs show it about 0.15 s after each).
  void _expectExitEcho() {
    _exitEcho = true;
    _echoTimer?.cancel();
    _echoTimer = Timer(const Duration(milliseconds: 800), () => _exitEcho = false);
  }

  void _onGlassesEvent(GlassesEvent e) {
    if (e.code == GlassesEvent.dashboardClosed) _expectExitEcho();
    switch (e.code) {
      case GlassesEvent.tap:
        if (voice.dictating && conversation.stage == TalkStage.listening) {
          // Tap while it listens: send it now (nothing said: "Didn't catch that").
          unawaited(voice.send());
        } else if (narrator.speaking && e.lr == 'L') {
          // Left tap while it talks: stop, and it is your turn.
          unawaited(stopSpeaking());
        } else if (conversation.stage == TalkStage.error && e.lr == 'R') {
          retry();
        } else if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else if (e.lr == 'L') {
          display.tapLeft();
        } else {
          display.tapRight();
        }
      case GlassesEvent.exit:
        if (_exitEcho) {
          // The glasses confirming a screen we closed ourselves.
          _exitEcho = false;
          return;
        }
        // Double tap: done with this - quiet, and no follow-up.
        final idle = display.mode == DisplayMode.idle && !voice.dictating && !conversation.busy && !narrator.speaking;
        final wasListening = voice.dictating && !voice.followUp;
        if (wasListening) {
          _aiSession = false;
          _sessionTarget = null;
          unawaited(voice.cancel()); // throw away what was being said
        }
        _stopLook();
        display.onGlassesExit();
        terminal.onGlassesExit();
        unawaited(narrator.stop());
        unawaited(voice.endFollowUp());
        // Nothing read aloud and no follow-up for a question still on its way.
        _speakThis = false;
        _spokenQuestion = false;
        conversation.dismiss();
        if (!wasListening) _doubleTap(idle: idle);
      case GlassesEvent.headUp:
        if (settings.lookMode == 'custom') {
          lookUp();
        } else {
          dashboard.onHeadUp();
        }
      case GlassesEvent.headDown:
        _expectExitEcho();
        dashboard.onHeadDown();
        _stopLook(close: true);
      case GlassesEvent.evenAiStart:
        if (settings.holdToTalk) {
          _glassesTalk();
        } else {
          // Never leave the glasses' own "Even AI listening" up with nothing behind it.
          unawaited(display.clear());
        }
      case GlassesEvent.evenAiEnd:
        _glassesRelease();
    }
  }

  /// Left TouchBar held (the glasses' Even AI gesture): talk to Claude.
  /// Uses the glasses' own microphone when the phone supports it.
  /// Left TouchBar held (the glasses' Even AI gesture): talk to Claude.
  ///
  /// The glasses tell us when the hold starts, but once our screen is up
  /// they never say when it is let go (firmware: the "released" event only
  /// comes while their own Even AI screen is showing with its microphone
  /// on). So what you say goes when you pause; a tap sends it at once and a
  /// double tap throws it away.
  Future<void> _glassesTalk() async {
    _aiSession = true;
    _stopLook();
    _sessionTarget = switch (settings.holdAction) {
      'chat' => Target.claude,
      'terminal' => Target.terminal,
      _ => null,
    };
    // The glasses put up their own "Even AI listening" the moment the
    // TouchBar is held (firmware, it cannot be skipped). Ours replaces it
    // first thing, before anything that takes time.
    if (settings.wordsWhileHolding) _showListening('', ai: true, touch: true);
    if (!await _micAllowed()) {
      _aiSession = false;
      conversation.listeningStopped();
      return;
    }
    // Holding the TouchBar interrupts whatever is being said.
    await narrator.stop();
    if (_target == Target.claude) unawaited(chat.warm());
    final useGlassesMic = settings.glassesMic;
    if (useGlassesMic) {
      _micOpen = true;
      unawaited(proto.micOn(true));
    }
    await voice.startDictation(
        pauseSends: true, fresh: true, source: useGlassesMic ? 'glasses' : 'phone');
    // A cap on one message, so it can never listen on and on.
    _holdLimit?.cancel();
    _holdLimit = Timer(const Duration(seconds: 45), () {
      if (voice.dictating && voice.pauseSends) unawaited(voice.send());
    });
  }

  /// The glasses said the TouchBar was let go (only while their own screen
  /// shows): send it now.
  void _glassesRelease() {
    _holdLimit?.cancel();
    _holdLimit = null;
    if (voice.dictating && voice.pauseSends) unawaited(voice.send());
    _closeGlassesMic();
  }

  void _closeGlassesMic() {
    if (!_micOpen) return;
    _micOpen = false;
    unawaited(proto.micOn(false));
  }

  /// Instructions sent with every message (as Claude's system prompt): this
  /// goes straight to the glasses, how long to be, and how to split longer
  /// answers into screens.
  String glassesPrimer({bool quick = false}) => buildPrimer(
        size: GlassesText.screenSize(),
        length: settings.answerLength,
        spoken: settings.speakReplies != 'off',
        phone: settings.phoneActions,
        quick: quick,
      );

  /// Model and options for the next question: the quick Claude (no tools,
  /// answers in about a second) or full Claude Code.
  Map<String, String> _claudeOptions({bool full = false}) => settings.answerSpeed == 'quick' && !full
      ? {'model': settings.quickModel, 'effort': 'low', 'access': settings.claudeAccess, 'lane': 'quick'}
      : {...settings.claudeOptions, 'lane': 'full'};

  /// "Think harder", "use Claude Code": straight to the full Claude.
  static bool wantsFull(String q) =>
      RegExp(r'\b(think (harder|hard|deeply|carefully)|use (full )?claude code|with (your )?tools)\b',
              caseSensitive: false)
          .hasMatch(q);

  static String buildPrimer(
      {required ScreenSize size, String length = 'brief', bool spoken = false, bool phone = false, bool quick = false}) {
    final chars = size.chars;
    final aim = switch (length) {
      'normal' => 'Aim for about 150 characters. Go up to $chars (one screen) only if it needs it.',
      'long' => 'Answer fully, but keep each screen-sized chunk tight.',
      _ => 'Aim for about 50 characters (one short line). Use up to 100-150 characters only when '
          'the answer needs that to make sense - never more unless they ask for detail.',
    };
    return 'Your reply goes straight to Even Realities G1 smart glasses, where the user reads it in '
        'front of their eyes (the Even Suite app). The user usually speaks; speech-to-text may contain '
        'mistakes, so read generously.\n'
        'The display shows ${size.lines} lines of about ${size.charsPerLine} characters: one screen '
        'holds about $chars characters, shown like a teleprompter.\n'
        'For every reply:\n'
        '- Be concise and brief. $aim\n'
        '- Answer first. No preamble, no filler, no restating the question.\n'
        '- Plain text only: no Markdown, headings, tables, code blocks, links or emoji. Use "-" for lists.\n'
        '- If more is truly needed (or asked for), split it into chunks of at most $chars characters '
        'separated by a blank line; each chunk becomes one screen.\n'
        '${spoken ? '- Your reply is also read aloud: write it to sound natural spoken, no symbols or abbreviations that read badly.\n' : ''}'
        '${quick ? '- Quick mode: you have no tools here. If the request needs them (the user\'s files, running '
            'commands, Termux, git, the web or anything current), reply with exactly [[full]] and nothing else - '
            'Claude Code with its tools takes over.' : '- You run as Claude Code on the user\'s phone (Termux). You can use your tools to do what '
            'they ask; then report the result in one short line.'}'
        '${phone ? '\n${PhoneAction.primer}' : ''}';
  }

  void _showLastReply({bool speak = false}) {
    final r = chat.lastReply;
    if (r == null) return;
    display.showText(PhoneAction.strip(r.text), title: 'Claude');
    if (speak && narrator.enabled) {
      unawaited(voice.suspend());
      _speakThis = true;
      narrator.finish('again-${DateTime.now().millisecondsSinceEpoch}', 0, r.text);
    }
  }

  // ---------------------------------------------------------------- voice

  /// Talk button / glasses long press. Interrupts the phone if it is talking.
  Future<void> startTalking({bool hold = false}) async {
    if (!await _micAllowed()) return;
    await narrator.stop();
    unawaited(chat.warm());
    await voice.startDictation(hold: hold);
  }

  Future<void> toggleTalk() async {
    if (voice.dictating && !voice.followUp) return voice.stopDictation();
    await startTalking();
  }

  /// Stop the phone talking (the answer stays on the screen).
  Future<void> stopSpeaking() async {
    await narrator.stop();
    _onSpoken();
  }

  /// Ask the last question again (after an error).
  Future<void> retry() async {
    if (_lastQuestion.isNotEmpty) await submit(_lastQuestion, spoken: _spokenQuestion);
  }

  Future<void> setHandsFree(bool on) async {
    if (on) {
      if (!await _micAllowed()) return;
      await system.permissions(['notifications']);
    }
    await voice.setHandsFree(on);
  }

  Future<bool> _micAllowed() async {
    final granted = await system.permissions(['microphone']);
    if (granted['microphone'] == true) return true;
    _toast('Allow the microphone so Even Suite can hear you');
    return false;
  }

  void _showDraft(String draft) {
    // While the TouchBar is held the glasses show their own listening screen;
    // writing to the display then could interrupt the microphone (unless
    // the test setting asks for it).
    final touch = _aiSession && voice.pauseSends;
    if (touch && !settings.wordsWhileHolding) return;
    _showListening(draft, ai: _aiSession, touch: touch);
  }

  /// The listening screen: "Termux-Claude listening" and your words, or in
  /// terminal mode "Terminal - say "start"" / "Terminal [start]" with
  /// "send" as the way to type it.
  void _showListening(String draft, {bool ai = false, bool touch = false}) {
    final followUp = !touch && voice.followUp;
    if (_target == Target.terminal) {
      final line = terminalWords(draft, _startWord);
      conversation.listening(line.text,
          level: voice.level.value,
          followUp: followUp,
          ai: ai,
          header: line.started || voice.wakeStarted ? 'Terminal [start]' : 'Terminal - say "start"',
          footer: touch ? 'say "send", pause or tap' : 'say "send" to type it',
          prompt: '');
      return;
    }
    conversation.listening(draft,
        level: voice.level.value, followUp: followUp, ai: ai, footer: touch ? 'pause or tap to send' : '');
  }

  /// What goes to the terminal: without a leading "start" (the word that
  /// opens a terminal message) and a trailing "send".
  static ({bool started, String text}) terminalWords(String said, WakeWord wake) {
    var text = said.trim();
    var started = false;
    final m = wake.find(text);
    if (m != null && m.start == 0) {
      started = true;
      text = m.rest;
    }
    final ending = VoiceCommands.splitEnding(text);
    if (ending.command == VoiceCommand.send) text = ending.body;
    return (started: started, text: text.trim());
  }

  /// Let go without saying anything: never leave the glasses on a listening screen.
  void _onNothingHeard() {
    final ai = _aiSession;
    _aiSession = false;
    _sessionTarget = null;
    conversation.nothingHeard(ai: ai);
  }

  void _onVoiceChanged() {
    final changed = voice.handsFree != _lastHandsFree || voice.autoOffAt != _lastAutoOffAt;
    _lastHandsFree = voice.handsFree;
    _lastAutoOffAt = voice.autoOffAt;
    if (changed) _updateService();
    if (!voice.dictating) {
      _closeGlassesMic();
      _holdLimit?.cancel();
    }
    // Left dictation without sending: take "Listening" off the glasses.
    // Checked a moment later, because a send replaces it with "Got it".
    if (!voice.dictating && (display.mode == DisplayMode.live || conversation.stage == TalkStage.listening)) {
      _liveCleanup?.cancel();
      _liveCleanup = Timer(const Duration(milliseconds: 600), () {
        // Still taking it down, or a message is on its way ("Got it" replaces it).
        if (voice.dictating || voice.sending) return;
        if (display.mode == DisplayMode.live) display.clear();
        conversation.listeningStopped();
      });
    }
  }

  Future<void> _updateService() async {
    final off = voice.autoOffAt;
    String text = '';
    if (voice.handsFree && off != null) {
      final hh = off.hour.toString().padLeft(2, '0');
      final mm = off.minute.toString().padLeft(2, '0');
      text = 'Say “${voice.wake.label}”, then your message and “send”. Turns off at $hh:$mm.';
    }
    final error = await system.service(
        listening: voice.handsFree, glasses: ble.isConnected, mic: ble.isConnected && settings.holdToTalk, text: text);
    if (error != null && voice.handsFree) {
      _toast('Android blocked background listening: $error');
    }
  }

  // ------------------------------------------------------------- messages

  /// Send a message to Claude, or type it into the terminal. [spoken]: it
  /// was said (the answer is read aloud and a follow-up is listened for).
  Future<void> submit(String text, {bool? spoken}) async {
    final t = text.trim();
    if (t.isEmpty) return;
    HapticFeedback.lightImpact();
    // "terminal" / "chat" said on its own switches where messages go.
    final mode = VoiceCommands.parse(t);
    if (mode == VoiceCommand.toTerminal || mode == VoiceCommand.toChat) {
      _aiSession = false;
      runCommand(mode!);
      return;
    }
    _spokenQuestion = spoken ?? voice.lastSpoken;
    _speakThis = settings.speakReplies == 'always' || (settings.speakReplies == 'voice' && _spokenQuestion);
    final target = _target;
    _sessionTarget = null;
    // "Flashlight on", "next song", "set a timer for 5 minutes": done at once, no Claude needed.
    if (target == Target.claude && settings.phoneActions) {
      final quick = PhoneAction.parseLocal(t, DateTime.now());
      if (quick != null) {
        await _doPhone(quick, quick: true);
        return;
      }
    }
    try {
      if (target == Target.terminal) {
        await _toTerminal(t);
      } else {
        _lastQuestion = t;
        _terminalAsk = null;
        _lane = settings.answerSpeed == 'quick' && !wantsFull(t) ? 'quick' : 'full';
        chat.system = glassesPrimer(quick: _lane == 'quick');
        chat.options = _claudeOptions(full: _lane == 'full');
        // "Got it" first: the glasses answer at once, the rest can take a moment.
        conversation.asked(t, ai: _aiSession);
        unawaited(narrator.stop());
        await chat.send(t);
      }
    } catch (e) {
      _toast('$e');
      conversation.failed('$e');
      _aiSession = false;
      _terminalAsk = null;
    }
  }

  /// Type a line into the terminal. With Claude Code there and its hooks set
  /// up, its steps and answer show on the glasses like a chat answer;
  /// otherwise the terminal screen is mirrored.
  Future<void> _toTerminal(String said) async {
    final line = terminalWords(said, _startWord).text;
    if (line.isEmpty) {
      _onNothingHeard();
      return;
    }
    // The hooks report Claude Code's steps and answer (Work tab).
    final follow = terminal.agent == 'claude' && (chat.hub?.hook == true || work.items.any((i) => !i.isMessage));
    if (follow) {
      _terminalAsk = line;
      conversation.asked(line,
          ai: _aiSession, who: 'Claude Code', gotIt: '[send] typed into Claude Code', firstStep: _typedStep);
    } else {
      conversation.ended();
    }
    unawaited(narrator.stop());
    await terminal.sendLine(line);
    _aiSession = false;
    if (!follow) {
      if (settings.repliesOnGlasses && ble.isConnected) terminal.setMirror(true);
      return;
    }
    // Claude Code never picked it up (not at its prompt, hooks off): show
    // the terminal itself instead of waiting.
    _terminalWatch?.cancel();
    _terminalWatch = Timer(const Duration(seconds: 20), () {
      if (_terminalAsk != line || !conversation.busy || conversation.step != _typedStep) return;
      _terminalAsk = null;
      conversation.ended();
      if (_glassesFree()) terminal.setMirror(true);
    });
  }

  static const _typedStep = '[send] typed it in';

  /// The Claude Code job started by [_terminalAsk] (matched by its prompt).
  static bool sameAsk(String asked, String prompt) {
    String norm(String s) => s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), ' ').trim();
    final a = norm(asked), p = norm(prompt);
    if (a.isEmpty || p.isEmpty) return false;
    final n = a.length < 40 ? a.length : 40;
    return p.startsWith(a.substring(0, n)) || a.startsWith(p.substring(0, p.length < n ? p.length : n));
  }

  void _followTerminalJob(List<WorkJob> jobs) {
    final ask = _terminalAsk;
    if (ask == null || conversation.who != 'Claude Code' || !conversation.busy) return;
    for (final j in jobs) {
      if (sameAsk(ask, j.prompt)) {
        conversation.externalProgress(j.step, j.recent);
        return;
      }
    }
  }

  /// Every finished Claude Code job: the answer you are waiting for, or a
  /// tiny "*" on the glasses so you know it is done.
  void _onJobFinished(WorkItem item) {
    if (item.isMessage) return;
    final ask = _terminalAsk;
    if (ask != null && sameAsk(ask, item.prompt)) {
      _terminalAsk = null;
      _shownJob = item.id;
      final text = item.text.isEmpty ? item.summary : item.text;
      HapticFeedback.mediumImpact();
      if (conversation.externalAnswer(text)) {
        dashboard.lastReply = text;
        if (_speakThis && narrator.enabled && text.isNotEmpty) {
          unawaited(voice.suspend());
          conversation.speaking();
          narrator.finish('job-${item.id}', 0, text);
        } else {
          _onSpoken();
        }
        return;
      }
    }
    if (settings.workAlerts && _doneBlink()) _blinkedJob = item.id;
  }

  /// "*" for 3 seconds: barely there, but you see Claude is done. Only when
  /// the glasses show nothing of ours.
  bool _doneBlink() {
    if (!settings.doneBlink || mirror.running || flicker.running || position.open) return false;
    if (voice.dictating || conversation.busy) return false;
    return display.flash('*');
  }

  /// Claude's reply without its phone-action lines.
  static BridgeMessage _clean(BridgeMessage m) {
    if (!m.text.contains('[[') && !m.live.contains('[[')) return m;
    return BridgeMessage(
        id: m.id,
        who: m.who,
        to: m.to,
        text: PhoneAction.strip(m.text),
        status: m.status,
        ts: m.ts,
        stage: m.stage,
        tool: m.tool,
        live: PhoneAction.strip(m.live),
        block: m.block,
        steps: m.steps,
        started: m.started,
        ms: m.ms,
        note: m.note);
  }

  /// Do it and say so: a notice on the glasses, a toast on the phone.
  Future<void> _doPhone(PhoneAction a, {bool quick = false}) async {
    if (quick) {
      _aiSession = false;
      conversation.ended();
    }
    final said = await phoneActions.run(a);
    EventLog.add('phone', 'action', {'do': a.kind, 'quick': quick, 'result': said});
    _toast(said);
    // After Claude's answer only a problem is worth a screen of its own.
    if ((quick || said.startsWith('Could not')) && _glassesFree() && !conversation.busy && !voice.dictating) {
      display.notice(said, duration: const Duration(seconds: 3));
    }
  }

  /// A reminder (or timer) went off: on the glasses, with a buzz on the phone.
  void _onReminder(String id, String text) {
    HapticFeedback.heavyImpact();
    EventLog.add('phone', 'reminder', {'id': id});
    if (ble.isConnected && !mirror.running && !flicker.running && !voice.dictating && !conversation.busy) {
      display.showText(text.startsWith('Timer done') ? text : 'Reminder\n$text', title: 'Reminder');
    }
  }

  void _onProgress(BridgeMessage raw) {
    final m = _clean(raw);
    conversation.progress(m);
    if (_speakThis && narrator.enabled && m.stage == 'writing' && m.live.isNotEmpty) {
      unawaited(voice.suspend());
      narrator.feed(m.id, m.block, m.live);
      if (narrator.speaking) conversation.speaking();
    }
  }

  /// The quick Claude needs tools for this one: the same question to Claude Code.
  void _escalate() {
    _lane = 'full';
    EventLog.add('chat', 'escalate', {});
    conversation.externalProgress('Handing to Claude Code', const []);
    chat.system = glassesPrimer();
    chat.options = _claudeOptions(full: true);
    unawaited(chat.send(_lastQuestion).catchError((Object e) => conversation.failed('$e')));
  }

  void _onReply(BridgeMessage raw) {
    if (_lane == 'quick' && !raw.isError && raw.text.contains('[[full]]')) {
      _escalate();
      return;
    }
    final actions = settings.phoneActions && !raw.isError ? PhoneAction.extract(raw.text) : const <PhoneAction>[];
    final m = _clean(raw);
    for (final a in actions) {
      unawaited(_doPhone(a));
    }
    HapticFeedback.mediumImpact();
    // Sent away with a double tap while it worked: just a "*" to say it is done.
    if (!conversation.answered(m) && !m.isError) _doneBlink();
    if (!m.isError) {
      dashboard.lastReply = m.text;
      if (settings.noteSlots.contains('reply')) dashboard.markDirty();
    }
    _aiSession = false;
    if (_speakThis && narrator.enabled && !m.isError) {
      unawaited(voice.suspend());
      conversation.speaking();
      narrator.finish(m.id, m.block, m.text);
    } else {
      _onSpoken();
    }
  }

  /// The answer has been said (or was not to be said): listen for a follow-up.
  void _onSpoken() {
    unawaited(voice.resume());
    if (conversation.stage == TalkStage.error) return;
    if (settings.conversation && _spokenQuestion && !voice.dictating) {
      _spokenQuestion = false; // one follow-up window per answer
      conversation.followingUp();
      unawaited(voice.startFollowUp(Duration(seconds: settings.followUpSec)));
    } else {
      conversation.ended();
    }
  }

  void _onFollowUpEnd() => conversation.ended();

  /// You said the wake word while the phone was talking: it stops at once.
  void _onBargeIn() {
    unawaited(narrator.stop());
    _speakThis = false;
  }

  /// A question from this app is part of "Claude's work" on the look-up screen too.
  void _onTalkChanged() {
    final c = conversation;
    final next = c.busy ? (c.step.isEmpty ? 'Thinking' : c.step, c.question, c.reply?.live ?? '') : null;
    if (next == dashboard.asking) return;
    dashboard.asking = next;
    dashboard.workChanged();
  }

  /// Claude Code finished a job (Work tab): say so briefly, only now.
  void _onWorkDone(WorkItem item) {
    dashboard.lastWork = item;
    if (settings.noteSlots.contains('work') || dashboard.claudePanel) dashboard.markDirty();
    if (!settings.workAlerts) return;
    if (item.id == _shownJob) return; // already on the glasses as the answer
    final where = item.project.isEmpty ? '' : ' · ${item.project}';
    if (item.isMessage) {
      // A Claude Code session in /even mode talking to you: read it on the glasses.
      if (settings.workPhone) unawaited(system.notify(_workNotice++, 'Claude Code$where', item.summary));
      if (_glassesFree() && !conversation.busy && !voice.dictating) {
        display.showText(item.text.isEmpty ? item.summary : item.text, title: 'Claude Code');
      }
      return;
    }
    if (settings.workPhone) {
      unawaited(system.notify(_workNotice++, 'Claude finished$where', item.summary));
    }
    if (settings.workGlasses && ble.isConnected && item.id != _blinkedJob && !conversation.busy && !voice.dictating) {
      unawaited(phone.send(PhoneNotifications.selfId, 'Claude', 'Claude finished$where', item.summary));
    }
  }

  /// Spoken commands ("Claude, scroll down").
  void runCommand(VoiceCommand cmd) {
    switch (cmd) {
      case VoiceCommand.scrollUp:
        if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else {
          display.up();
        }
      case VoiceCommand.scrollDown:
        if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else {
          display.down();
        }
      case VoiceCommand.top:
        display.top();
      case VoiceCommand.bottom:
        display.bottom();
      case VoiceCommand.repeat:
        _showLastReply(speak: true);
      case VoiceCommand.hide:
        display.clear();
        terminal.setMirror(false);
        unawaited(narrator.stop());
      case VoiceCommand.stop:
        if (narrator.speaking) {
          unawaited(stopSpeaking());
        } else if (chat.busy) {
          chat.stopReply();
        } else if (voice.followUp) {
          unawaited(voice.endFollowUp());
        } else {
          voice.setHandsFree(false);
        }
      case VoiceCommand.stopReply:
        chat.stopReply();
      case VoiceCommand.toTerminal:
        settings.set('target', Target.terminal);
        _toast('Messages now go to the terminal');
        _modeNotice('Terminal mode\nSay "start", your words, then "send"\nSay "chat" to go back');
      case VoiceCommand.toChat:
        settings.set('target', Target.claude);
        _toast('Messages now go to Claude chat');
        _modeNotice('Chat mode\nTalking to Claude again');
      case VoiceCommand.enter:
        terminal.key('Enter');
      case VoiceCommand.escape:
        terminal.key('Escape');
      case VoiceCommand.choice1:
        terminal.key('1');
      case VoiceCommand.choice2:
        terminal.key('2');
      case VoiceCommand.choice3:
        terminal.key('3');
      case VoiceCommand.pause:
        display.pause();
      case VoiceCommand.resume:
        display.resume();
      case VoiceCommand.faster:
        display.faster();
        settings.set('readingWpm', display.wpm);
      case VoiceCommand.slower:
        display.slower();
        settings.set('readingWpm', display.wpm);
      case VoiceCommand.send:
      case VoiceCommand.cancel:
      case VoiceCommand.stopListening:
        break; // handled by the voice controller
    }
  }

  /// Double tap on the glasses (when nothing was being said): whatever you
  /// chose. "talk" starts listening without the glasses' Even AI screen
  /// (only a long press brings that up) - when nothing of ours was showing;
  /// otherwise the double tap just closed it.
  void _doubleTap({required bool idle}) {
    switch (settings.doubleTapAction) {
      case 'talk':
        if (idle) unawaited(_glassesTalk());
      case 'switch':
        runCommand(settings.target == Target.terminal ? VoiceCommand.toChat : VoiceCommand.toTerminal);
      case 'repeat':
        _showLastReply();
      default:
        break; // the glasses already closed the screen
    }
  }

  /// Where messages go now, shown briefly on the glasses.
  void _modeNotice(String text) {
    conversation.ended();
    if (_glassesFree()) display.notice(text, duration: const Duration(seconds: 4));
  }

  // -------------------------------------------------------------- look up

  static const lookTitle = 'Look up';

  /// Your layout (from Settings), parsed once per change.
  LookLayout get lookLayout {
    if (settings.lookLayout != _lookKey) {
      _lookKey = settings.lookLayout;
      _lookCache = LookLayout.parse(settings.lookLayout);
    }
    return _lookCache!;
  }

  /// What the look-up screen shows right now.
  LookData lookData({DateTime? now}) {
    final panel = dashboard.panel;
    final work = dashboard.lastWork;
    String ago(double ts) {
      final d = DateTime.now().difference(DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()));
      if (d.inMinutes < 1) return 'just now';
      if (d.inHours < 1) return '${d.inMinutes} min ago';
      return '${d.inHours} h ago';
    }

    return LookData(
      now: now ?? DateTime.now(),
      h24: settings.dash24h,
      fahrenheit: settings.dashFahrenheit,
      weather: settings.dashWeather ? dashboard.weather : null,
      claude: [
        if (panel.title.trim().isNotEmpty) panel.title,
        ...panel.lines.where((l) => l.trim().isNotEmpty),
      ],
      work: work == null ? const [] : ['Done ${ago(work.ts)}: ${work.summary}'],
      phone: List.of(dashboard.phone),
      battery: _battery,
    );
  }

  bool get _lookShowing => display.mode == DisplayMode.status && display.title == lookTitle;

  /// You looked up: your own layout replaces the glasses' dashboard, and
  /// keeps the clock and Claude's work fresh while you look.
  void lookUp({Duration lasts = const Duration(seconds: 45)}) {
    if (!ble.isConnected || mirror.running || flicker.running || position.open) return;
    // Never over something you are reading, saying or waiting for.
    if (display.mode != DisplayMode.idle && !_lookShowing) return;
    if (voice.dictating || conversation.busy) return;
    _lookSince = DateTime.now();
    _drawLook(immediate: true);
    _lookTimer?.cancel();
    _lookTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (!_lookShowing || DateTime.now().difference(_lookSince!) > lasts) {
        _stopLook(close: true);
        return;
      }
      _drawLook();
    });
    unawaited(system.battery().then((b) => _battery = b, onError: (_) => null));
  }

  /// Show your look-up screen now for a few seconds (from the phone).
  void showLookNow() => lookUp(lasts: const Duration(seconds: 8));

  void _drawLook({bool immediate = false}) {
    final rows = LookRenderer.render(lookLayout, lookData());
    display.showStatus(rows.join('\n'), title: lookTitle, raw: true, immediate: immediate);
  }

  void _stopLook({bool close = false}) {
    _lookTimer?.cancel();
    _lookTimer = null;
    if (close && _lookShowing) unawaited(display.clear());
  }

  // --------------------------------------------------------------- termux

  /// Start Agent Hub in Termux (runs the one-word `even`, the same as typing
  /// it in Termux). Returns a message for the user.
  Future<String> startBridge() => _termux(settings.startCommand, 'Claude is ready.');

  /// Stop and start Agent Hub again (`even restart`), e.g. after an update.
  Future<String> restartBridge() async {
    final cmd = settings.startCommand == Settings.defaultStartCommand ? 'even restart' : settings.startCommand;
    if (cmd != 'even restart') {
      try {
        await client.quit();
        await Future.delayed(const Duration(seconds: 2));
      } catch (_) {}
    }
    return _termux(cmd, 'Restarted. Claude is ready.');
  }

  /// Turn Agent Hub off (`even stop`).
  Future<String> stopBridge() async {
    try {
      await client.quit();
    } catch (e) {
      return '$e';
    }
    await Future.delayed(const Duration(milliseconds: 600));
    await chat.check();
    return 'Stopped. Tap Start (or type even in Termux) to turn it on again.';
  }

  Future<String> _termux(String command, String done) async {
    if (!await system.termuxInstalled()) return 'Termux is not installed on this phone.';
    final granted = await system.permissions(['termux']);
    if (granted['termux'] != true) {
      // Android only asks twice; after that it has to be switched on by hand.
      await system.openAppSettings();
      return 'Tap Permissions › Additional permissions › “Run commands in Termux environment” › Allow. '
          'Or just type even in Termux.';
    }
    final r = await system.runTermux(command);
    if (r == 'ok') {
      chat.markStarting();
      // Ubuntu and Claude Code take a while to start on a phone.
      for (var i = 0; i < 60; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (await chat.check()) return done;
      }
      chat.state = BridgeState.offline;
      return 'Started it, but Claude is not answering yet. In Termux, type: even log';
    }
    if (r == 'no-permission') {
      await system.openAppSettings();
      return 'Tap Permissions › Additional permissions › “Run commands in Termux environment” › Allow.';
    }
    if (r == 'not-installed') return 'Termux is not installed on this phone.';
    return 'Could not start Termux: $r. In Termux, type: even';
  }

  @override
  void dispose() {
    _events?.cancel();
    _phoneNotes?.cancel();
    work.dispose();
    narrator.dispose();
    conversation.removeListener(_onTalkChanged);
    conversation.dispose();
    dashboard.dispose();
    phone.dispose();
    tabRequests.close();
    _heartbeat?.cancel();
    _liveCleanup?.cancel();
    _holdLimit?.cancel();
    _mirrorTimer?.cancel();
    _terminalWatch?.cancel();
    _lookTimer?.cancel();
    _echoTimer?.cancel();
    speechModel.removeListener(_applySpeech);
    settings.removeListener(applySettings);
    voice.dispose();
    mirror.dispose();
    flicker.dispose();
    chat.dispose();
    terminal.dispose();
    display.dispose();
    ble.dispose();
    client.close();
    toasts.close();
    super.dispose();
  }
}
