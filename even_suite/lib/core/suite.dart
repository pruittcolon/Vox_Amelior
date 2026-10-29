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
import 'glasses/glasses_display.dart';
import 'glasses/glasses_text.dart';
import 'glasses/hud_frame.dart';
import 'home/home_commands.dart';
import 'home/vizio.dart';
import 'home/voice_monkey.dart';
import 'glasses/look_layout.dart';
import 'log.dart';
import 'phone_actions.dart';
import 'phone_notifications.dart';
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
    terminal = TerminalController(this.client, display)
      ..onSessionGone = () {
        // The one picked was closed: the Termux Claude (or the first one left).
        final next = terminal.claudes.isNotEmpty ? terminal.claudes.first.name : terminal.sessions.first.name;
        settings.set('terminalAgent', next);
        terminal.setSession(next);
      };
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
      ..paused = () => mirror.running || display.flashing;
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
      ..onBargeIn = _onBargeIn
      ..onTrigger = _onTrigger;
    voice.spokenText = () => narrator.recent;

    ble.onConnected = (pair) {
      settings.set('lastGlasses', pair.toMap());
      _toast('Glasses connected');
    };
    ble.onUnsolicited = _onUnsolicited;
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
  late final PhoneActions phoneActions = PhoneActions(system);

  /// The lights (Alexa routines through Voice Monkey) and the Vizio TV.
  VoiceMonkey voiceMonkey = VoiceMonkey();
  VizioTv tv = VizioTv();

  /// Text height (the Text height page); nothing else draws while it is open.
  late final DisplayPosition position = DisplayPosition(proto);

  StreamSubscription<GlassesEvent>? _events;
  StreamSubscription<Map>? _phoneNotes;
  Timer? _heartbeat;
  Timer? _liveCleanup;
  Timer? _holdLimit;
  Timer? _mirrorTimer;
  Timer? _terminalWatch;
  Timer? _batteryTimer;
  Timer? _batterySoon;
  Timer? _redrawTimer;
  Timer? _foreignTimer;
  DateTime _foreignFixed = DateTime.fromMillisecondsSinceEpoch(0);
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

  /// Talking with the TouchBar held: the glasses' own Even AI screen is up.
  bool _holdTalk = false;
  bool _lastConnected = false;
  bool _lastWantsLink = false;
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
  /// replies switched on).
  bool _glassesFree() => ble.isConnected && settings.repliesOnGlasses && !mirror.running;

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
    unawaited(_logLastExit());
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

  /// Why Android ended the app's process the last few times (killed for
  /// memory, frozen, swiped away...) - the answer when the link went down
  /// while the app was out of sight.
  Future<void> _logLastExit() async {
    try {
      final list = await system.exitReasons();
      if (list.isNotEmpty) EventLog.add('app', 'exitReasons', {'last': list});
    } catch (_) {}
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
      // A pause sends - to the terminal too ("send" sends at once).
      ..autoSend = settings.autoSend
      ..wakeAutoSend = settings.wakeAutoSend
      ..silenceMs = settings.pauseMs;
    chat.agent = settings.agent;
    conversation.agent = dashboard.agent = chat.agentName;
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
      // Always kept filled: left alone, the Even app's last write shows.
      ..enabled = true
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
    display
      ..holdMs = settings.pageHoldMs
      ..fixedMs = settings.pageSec * 1000
      ..firstExtraMs = settings.firstPageMs;
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
    chat.system = glassesPrimer(quick: _quickLane);
    proto.parallelText = settings.bleParallel;
    voice.stableAfter = Duration(milliseconds: settings.touchPauseMs);
    dashboard.notesOn = settings.lookMode != 'custom';
    // Termux on the glasses: this long after its screen last changed.
    terminal.mirrorIdle = switch (settings.termShowSec) {
      < 0 => null,
      0 => Duration(minutes: settings.autoOffMinutes),
      final s => Duration(seconds: s),
    };
    if (terminal.session != settings.terminalAgent) terminal.setSession(settings.terminalAgent);
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
    // The background service follows what is WANTED (up, or being rebuilt),
    // not what is up: a dropped link must not let Android freeze the app.
    if (ble.wantsLink != _lastWantsLink) {
      _lastWantsLink = ble.wantsLink;
      _updateService();
    }
    final connected = ble.isConnected;
    if (connected == _lastConnected) return;
    _lastConnected = connected;
    _heartbeat?.cancel();
    if (connected) {
      dashboard.onConnected();
    } else {
      dashboard.onDisconnected();
      phone.onDisconnected();
    }
    _batteryTimer?.cancel();
    _batterySoon?.cancel();
    if (connected) {
      _leftFixTried = false;
      _modeCheck?.cancel();
      _modeCheck = Timer(const Duration(seconds: 6), () async {
        if (!ble.isConnected || mirror.running || settings.lookMode != 'dashboard') return;
        final l = await proto.dashStartupMode('L'), r = await proto.dashStartupMode('R');
        EventLog.add('dash', 'startupMode', {'L': l, 'R': r});
      });
      // The glasses' battery, now and every 5 minutes (shown in the app).
      _askBatterySoon(const Duration(seconds: 3));
      _batteryTimer = Timer.periodic(const Duration(minutes: 5), (_) => _askBattery());
    }
    if (connected) {
      // Same 8 s heartbeat as the demo keeps the link from idling out - but
      // not right after we wrote something ourselves (pictures, screens), so
      // it never cuts into a transfer. What the glasses SEND does not count:
      // looking up and down makes them chatter, and a heartbeat skipped for
      // that left them 30 s without a word from us, so they hung up (four of
      // six drops in the log). The native side backs this up (BleManager).
      _heartbeat = Timer.periodic(const Duration(seconds: 8), (_) {
        if (ble.clock().difference(ble.lastWrite) < const Duration(seconds: 4)) return;
        proto.sendHeartBeat();
      });
    }
    _updateService();
  }

  DateTime _batteryAsked = DateTime.fromMillisecondsSinceEpoch(0);

  void _askBatterySoon(Duration after) {
    _batterySoon?.cancel();
    _batterySoon = Timer(after, _askBattery);
  }

  /// Read the glasses' battery again (the app shows it). [now] skips the
  /// 20 s spacing - a tap on the battery.
  Future<void> askBattery({bool now = false}) {
    if (now) _batteryAsked = DateTime.fromMillisecondsSinceEpoch(0);
    return _askBattery();
  }

  Future<void> _askBattery() async {
    if (!ble.isConnected || mirror.running) return;
    if (DateTime.now().difference(_batteryAsked) < const Duration(seconds: 20)) return;
    _batteryAsked = DateTime.now();
    try {
      await proto.queryBattery();
    } catch (_) {}
    EventLog.add('glasses', 'battery', {'L': ble.batteryL, 'R': ble.batteryR});
    // No answer (busy link): try again soon rather than in 5 minutes.
    if (ble.isConnected && ble.battery == null) {
      _askBatterySoon(const Duration(seconds: 25));
    }
  }

  /// Packets the glasses sent for someone else. Dashboard acks there mean
  /// another app on the phone (the Even app) just rewrote the dashboard -
  /// it does that every minute. With the built-in look-up screen, ours goes
  /// back right after, so their update never replaces your choice.
  void _onUnsolicited(BleReceive r) {
    if (r.data.length < 4 || r.data[0] != 0x06 || r.lr != 'R') return;
    // A late answer to our own packet (same sequence number) is not someone else's.
    if (proto.isOurDashSeq(r.data[3])) return;
    dashboard.noteForeign();
    if (settings.lookMode != 'dashboard') return;
    _foreignTimer?.cancel();
    _foreignTimer = Timer(const Duration(milliseconds: 900), () {
      // At most every 20 s, whatever the other app does.
      if (DateTime.now().difference(_foreignFixed) < const Duration(seconds: 20)) return;
      _foreignFixed = DateTime.now();
      EventLog.add('dash', 'foreign', {'writes': dashboard.foreignWrites});
      unawaited(dashboard.sync(force: true));
    });
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
    // Their own dashboard came up (or went) over your look-up layout: draw
    // ours again (an unchanged screen is otherwise never resent).
    if ((e.code == GlassesEvent.dashboardShown || e.code == GlassesEvent.dashboardClosed) && _lookShowing) {
      _redrawTimer?.cancel();
      _redrawTimer = Timer(const Duration(milliseconds: 150), () {
        if (!_lookShowing) return;
        if (settings.lookFrame == 'off') {
          display.redraw();
        } else if (!_painting || _pictureLanded) {
          // Picture and text again (a picture still on its way covers theirs anyway).
          unawaited(_paintLook(soon: true));
        }
      });
    }
    // Both arms report their own dashboard opening; note when the left one
    // stays dark (the right arm tells it you looked up).
    if (e.code == GlassesEvent.dashboardShown) {
      _dashShownAt[e.lr] = DateTime.now();
      if (e.lr == 'R') {
        final at = _dashShownAt['R']!;
        _leftCheck?.cancel();
        _leftCheck = Timer(const Duration(milliseconds: 800), () {
          final l = _dashShownAt['L'];
          if (l == null || at.difference(l).inMilliseconds > 800) {
            EventLog.add('dash', 'leftMissing', {'look': settings.lookMode});
            // Built-in look-up with the left eye dark: repair it once per connection.
            if (settings.lookMode == 'dashboard' && !_leftFixTried) {
              _leftFixTried = true;
              unawaited(fixLeftEye());
            }
          }
        });
      }
    }
    // Charging or battery news from the glasses: read the battery again.
    if (e.code == 9 || e.code == 10) _askBatterySoon(const Duration(seconds: 1));
    switch (e.code) {
      case GlassesEvent.tap:
        if (voice.dictating && (conversation.stage == TalkStage.listening || (_holdTalk && voice.pauseSends))) {
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
  ///
  /// Their own "Even AI listening" stays up (it is instant, and only while
  /// it shows do they say when the TouchBar is let go - so it sends the
  /// moment you let go), and their microphone starts right away: it hears
  /// you, not the room. With Android's recognizer (which hears nothing from
  /// the glasses' mic) the phone listens. A pause sends too; a tap sends at
  /// once, a double tap throws it away.
  Future<void> _glassesTalk({bool hold = true}) async {
    _aiSession = true;
    _holdTalk = hold;
    _stopLook();
    _sessionTarget = switch (settings.holdAction) {
      'chat' => Target.claude,
      'terminal' => Target.terminal,
      _ => null,
    };
    // The microphone first, before anything that takes time (the sound
    // that comes meanwhile is kept and heard).
    final glasses = holdMicReady;
    if (glasses) _openGlassesMic();
    // A double tap has no Even AI screen of its own: ours shows it listens.
    if (!hold || settings.wordsWhileHolding) _showListening('', ai: true, touch: true);
    if (!glasses && !await _micAllowed()) {
      _aiSession = false;
      conversation.listeningStopped();
      return;
    }
    // Holding the TouchBar interrupts whatever is being said.
    unawaited(narrator.stop());
    if (_target == Target.claude) unawaited(chat.warm());
    await _dictate(pauseSends: true, fresh: true, glasses: glasses);
    // A cap on one message, so it can never listen on and on.
    _holdLimit?.cancel();
    _holdLimit = Timer(const Duration(seconds: 45), () {
      if (voice.dictating && voice.pauseSends) unawaited(voice.send());
    });
  }

  /// The glasses said the TouchBar was let go (while their own screen
  /// shows): the recognizer finishes the last words and it is sent at once.
  void _glassesRelease() {
    _holdLimit?.cancel();
    _holdLimit = null;
    if (voice.dictating && voice.pauseSends) voice.releaseHold();
    _closeGlassesMic();
  }

  void _closeGlassesMic() {
    if (!_micOpen) return;
    _micOpen = false;
    unawaited(proto.micOn(false));
  }

  void _openGlassesMic() {
    if (_micOpen) return;
    _micOpen = true;
    unawaited(proto.micOn(true));
  }

  /// Listen through the glasses' own microphone - it hears you, not the TV
  /// or the people around - while they are connected and Parakeet is the
  /// speech engine (Android's recognizer hears nothing from it on many phones).
  bool get glassesMicInUse => holdMicReady;

  /// The glasses' microphone can be used: switched on (the default),
  /// connected, and Parakeet listening.
  bool get holdMicReady =>
      settings.glassesMic && ble.isConnected && settings.speechEngine == 'parakeet' && speechModel.ready;

  /// Start taking a message down, from the glasses' microphone when it can be
  /// used ([followUp]: the window after an answer).
  Future<void> _dictate(
      {bool hold = false, bool pauseSends = false, bool fresh = false, Duration? followUp, bool? glasses}) async {
    final useGlasses = glasses ?? glassesMicInUse;
    conversation.calmListening = useGlasses;
    if (useGlasses) _openGlassesMic();
    final source = useGlasses ? 'glasses' : 'phone';
    if (followUp != null) {
      await voice.startFollowUp(followUp, source: source);
    } else {
      await voice.startDictation(hold: hold, pauseSends: pauseSends, fresh: fresh, source: source);
    }
    // (Switching away from the phone's mic can close it on the way.)
    if (useGlasses && voice.dictating) _openGlassesMic();
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
        agent: settings.agent,
      );

  /// Model and options for the next question: the quick Claude (no tools,
  /// answers in about a second) or full Claude Code.
  Map<String, String> _claudeOptions({bool full = false}) {
    // Antigravity has no quick lane and its own model names: effort and what
    // it may do carry over, running commands is a separate yes.
    if (settings.agent == 'agy') {
      return {
        'effort': settings.claudeEffort,
        'access': settings.claudeAccess,
        if (settings.agyCommands) 'agy_commands': 'true',
        'lane': 'full',
      };
    }
    return _quickLane && !full
        ? {'model': settings.quickModel, 'effort': 'low', 'access': settings.claudeAccess, 'lane': 'quick'}
        : {...settings.claudeOptions, 'lane': 'full'};
  }

  /// The quick Claude (no tools) answers first - only Claude has one.
  bool get _quickLane => settings.answerSpeed == 'quick' && settings.agent == 'claude';

  /// "Think harder", "use Claude Code": straight to the full Claude.
  static bool wantsFull(String q) =>
      RegExp(r'\b(think (harder|hard|deeply|carefully)|use (full )?claude code|with (your )?tools)\b',
              caseSensitive: false)
          .hasMatch(q);

  static String buildPrimer(
      {required ScreenSize size,
      String length = 'brief',
      bool spoken = false,
      bool phone = false,
      bool quick = false,
      String agent = 'claude'}) {
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
            'Claude Code with its tools takes over.' : '- You run as ${agent == 'agy' ? 'Google Antigravity (agy)' : 'Claude Code'} on the user\'s phone (Termux). You can use your tools to do what '
            'they ask; then report the result in one short line.'}'
        '${phone ? '\n${PhoneAction.primer}' : ''}';
  }

  void _showLastReply({bool speak = false}) {
    final r = chat.lastReply;
    if (r == null) return;
    display.showText(PhoneAction.strip(r.text), title: chat.agentName);
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
    await _dictate(hold: hold);
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
    if (touch && _holdTalk && !settings.wordsWhileHolding) return;
    _showListening(draft, ai: _aiSession, touch: touch);
  }

  /// The listening screen: "Listening" and your words, or in
  /// terminal mode which terminal it goes to ("> Claude 2", "[start]" once
  /// said) and that a pause or "send" types it in.
  void _showListening(String draft, {bool ai = false, bool touch = false}) {
    final followUp = !touch && voice.followUp;
    if (_target == Target.terminal) {
      final line = terminalWords(draft, _startWord);
      conversation.listening(line.text,
          level: voice.level.value,
          followUp: followUp,
          ai: ai,
          header: '> ${terminal.label}${line.started || voice.wakeStarted ? ' [start]' : ''}',
          footer: touch ? 'pause, say "send" or tap' : 'pause or say "send"',
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
      _holdTalk = false;
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
        listening: voice.handsFree, glasses: ble.wantsLink, mic: ble.isConnected && settings.holdToTalk, text: text);
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
    if (mode == VoiceCommand.toTerminal ||
        mode == VoiceCommand.toChat ||
        mode == VoiceCommand.newSession ||
        mode == VoiceCommand.nextSession ||
        mode == VoiceCommand.useAgy ||
        mode == VoiceCommand.useClaude) {
      _aiSession = false;
      runCommand(mode!);
      return;
    }
    _spokenQuestion = spoken ?? voice.lastSpoken;
    // A fresh question (not said in the window after an answer) starts the count again.
    if (spoken != null || !voice.lastFollowUp) _followUps = 0;
    _speakThis = settings.speakReplies == 'always' || (settings.speakReplies == 'voice' && _spokenQuestion);
    final target = _target;
    _sessionTarget = null;
    // "Flashlight on", "next song", "set a timer for 5 minutes": done at once,
    // no Claude needed - in chat or terminal mode alike (never typed in).
    // The lights and the TV the same way (and the TV remote, once started).
    final home = _homeCommand(t);
    if (home != null) {
      await _doHome(home);
      return;
    }
    if (settings.phoneActions) {
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
        _lane = _quickLane && !wantsFull(t) ? 'quick' : 'full';
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

  /// Type a line into the terminal you picked. With Claude Code there and
  /// its hooks set up, its steps and answer show on the glasses like a chat
  /// answer; otherwise the terminal screen is mirrored.
  Future<void> _toTerminal(String said, {bool typed = false}) async {
    // Spoken: without "start" in front and "send" at the end. Typed: as is.
    final line = typed ? said.trim() : terminalWords(said, _startWord).text;
    if (line.isEmpty) {
      _onNothingHeard();
      return;
    }
    // The session picked was closed elsewhere: another one instead.
    if (terminal.sessions.isNotEmpty && terminal.current == null) terminal.onSessionGone?.call();
    final label = terminal.label;
    // The hooks report Claude Code's steps and answer (Work tab).
    // With "Terminal on glasses" set, the terminal itself shows instead.
    final follow = settings.termShowSec == 0 &&
        terminal.isClaude &&
        (chat.hub?.hook == true || work.items.any((i) => !i.isMessage));
    if (follow) {
      _terminalAsk = line;
      conversation.asked(line, ai: _aiSession, who: label, gotIt: '[send] typed into $label', firstStep: _typedStep);
    } else {
      conversation.ended();
    }
    unawaited(narrator.stop());
    final String result;
    try {
      result = await terminal.sendLine(line);
    } finally {
      _aiSession = false;
    }
    if (result == 'menu' || result == 'stuck') {
      // Not sent: say so on the glasses instead of waiting for an answer.
      _terminalAsk = null;
      conversation.ended();
      final text = result == 'menu'
          ? '$label is asking you something\nSay "one", "two" or "escape"'
          : 'Typed into $label, not sent yet\nSay "enter" to send it';
      _toast(text.split('\n').first);
      if (_glassesFree()) {
        display.notice(text, duration: const Duration(seconds: 6));
      }
      return;
    }
    if (!follow) {
      if (settings.repliesOnGlasses && ble.isConnected) {
        // Already showing: start its time again from now.
        terminal.setMirror(false);
        terminal.setMirror(true);
      }
      return;
    }
    // Claude Code never picked it up (hooks off): show the terminal itself
    // instead of waiting.
    _terminalWatch?.cancel();
    _terminalWatch = Timer(const Duration(seconds: 20), () {
      if (_terminalAsk != line || !conversation.busy || conversation.step != _typedStep) return;
      _terminalAsk = null;
      conversation.ended();
      if (_glassesFree()) terminal.setMirror(true);
    });
  }

  /// Typed on the phone into the terminal you picked (word for word, then
  /// Enter); its answer comes to the glasses like a spoken one.
  Future<void> typeIntoTerminal(String text) async {
    if (text.trim().isEmpty) return;
    _spokenQuestion = false;
    _speakThis = settings.speakReplies == 'always';
    try {
      await _toTerminal(text, typed: true);
    } catch (e) {
      _toast('$e');
      conversation.failed('$e');
      _terminalAsk = null;
    }
  }

  /// Talk to Claude chat from now on (the TouchBar hold too).
  void useChat() => _setTarget(Target.claude);

  /// Who answers from now on: 'claude', or 'agy' (Google Antigravity - for
  /// when Claude has run out). The same voice, glasses and screens; each one
  /// keeps its own conversation in Agent Hub. [toChat]: said by voice, so
  /// messages go to chat, not the terminal.
  void switchAgent(String agent, {bool toChat = false}) {
    final next = agent == 'agy' ? 'agy' : 'claude';
    final name = ChatController.agentLabel(next);
    if (toChat && settings.target != Target.claude) useChat();
    if (settings.agent == next) {
      _toast('Already talking to $name');
      if (toChat) _modeNotice('$name\nAlready answering');
      return;
    }
    settings.set('agent', next);
    EventLog.add('chat', 'agent', {'agent': next});
    _toast('Now talking to $name');
    _modeNotice('$name\nAnswers from now on');
    if (settings.target == Target.claude) unawaited(chat.warm());
  }

  /// Talk to terminal [name] from now on ("claude" is the one Termux shows).
  void useSession(String name) {
    settings.set('terminalAgent', name);
    terminal.setSession(name);
    _setTarget(Target.terminal);
  }

  void _setTarget(Target target) {
    if (settings.target != target) settings.set('target', target);
    // A hold fixed to the other one would quietly send it somewhere else.
    final hold = settings.holdAction;
    if (hold != 'mode' && hold != (target == Target.terminal ? 'terminal' : 'chat')) {
      settings.set('holdAction', 'mode');
    }
  }

  /// Open another Claude Code session and talk to it.
  Future<void> newSession() async {
    try {
      final name = await terminal.openSession();
      useSession(name);
      _modeNotice('New session: ${TermSession.labelFor(name)}\nWhat you say goes there');
    } catch (e) {
      _toast('Could not open a session: $e');
    }
  }

  /// The next Claude Code session (by voice: "next session").
  Future<void> nextSession() async {
    await terminal.refreshSessions();
    final list = terminal.claudes;
    if (list.isEmpty) return newSession();
    final at = list.indexWhere((s) => s.name == terminal.session);
    final next = settings.target == Target.terminal ? list[(at + 1) % list.length] : list[at < 0 ? 0 : at];
    useSession(next.name);
    _modeNotice('> ${next.label}${next.topic.isEmpty ? '' : '\n${next.topic}'}');
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
    // Only while that terminal question is still waiting (any session's label).
    if (ask == null || !conversation.busy) return;
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
    if (!settings.doneBlink || mirror.running || position.open) return false;
    if (voice.dictating || conversation.busy) return false;
    return display.flash('*', duration: Duration(milliseconds: settings.doneBlinkMs));
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
    if (ble.isConnected && !mirror.running && !voice.dictating && !conversation.busy) {
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

  /// Back-and-forth turns in a row without a fresh question: a TV or other
  /// people in the room can never keep it going on its own.
  static const maxFollowUps = 6;
  int _followUps = 0;

  /// The answer has been said (or was not to be said). One question (the
  /// standard): that is it - the answer stays for its reading time, then
  /// goes. Back and forth: listen for a follow-up.
  void _onSpoken() {
    unawaited(voice.resume());
    if (conversation.stage == TalkStage.error) return;
    if (settings.talkMode == 'chat' && _spokenQuestion && !voice.dictating && _followUps < maxFollowUps) {
      _spokenQuestion = false; // one follow-up window per answer
      _followUps++;
      conversation.followingUp();
      unawaited(_dictate(followUp: Duration(seconds: settings.followUpSec)));
    } else {
      if (_followUps >= maxFollowUps) EventLog.add('voice', 'followUpLimit', {'turns': _followUps});
      _followUps = 0;
      conversation.ended();
    }
  }

  void _onFollowUpEnd() {
    if (tvRemote) _endTvRemote();
    conversation.ended();
  }

  // ---------------------------------------------------------- lights, TV

  /// The TV remote is on: "up", "OK", "back" ... go to the TV until "done".
  bool tvRemote = false;
  Timer? _tvRemoteOff;
  static const tvRemoteIdle = Duration(seconds: 30);

  HomeCommand? _homeCommand(String text, {bool trigger = false}) {
    if (tvRemote && !trigger) return HomeCommands.remote(text);
    return HomeCommands.parse(text, VoiceMonkey.parse(settings.vmDevices), trigger: trigger);
  }

  /// Hands-free, heard without the wake word: a timer, a reminder, the
  /// lights or the TV are done at once when that is all that was said
  /// ("set a timer for 5 minutes", "all lights off", "TV off"). Anything
  /// else is ignored, as before.
  bool _onTrigger(String text) {
    if (!settings.triggerWords) return false;
    final home = _homeCommand(text, trigger: true);
    final phone = settings.phoneActions ? PhoneAction.parseLocal(text, DateTime.now()) : null;
    if (home == null && phone?.kind != 'remind') return false;
    EventLog.add('voice', 'trigger', {'what': home?.kind.name ?? phone!.kind});
    unawaited(submit(text, spoken: true));
    return true;
  }

  Future<void> _doHome(HomeCommand c) async {
    _aiSession = false;
    conversation.ended();
    String said;
    switch (c.kind) {
      case HomeKind.lights:
        final d = VoiceMonkey.parse(settings.vmDevices).firstWhere((d) => d.device == c.device);
        said = await voiceMonkey.trigger(settings.vmToken, d);
      case HomeKind.tvRemote:
        _startTvRemote();
        return;
      case HomeKind.tvDone:
        _endTvRemote();
        if (voice.followUp) unawaited(voice.endFollowUp());
        return;
      case HomeKind.tvKeys:
      case HomeKind.tvInput:
        said = await _tvDo(c);
    }
    EventLog.add('home', c.kind.name, {'result': said});
    _toast(said);
    if (tvRemote) {
      _armTvRemote();
      _homeNotice('TV remote: $said\nup down left right  OK  back\nhome  volume up/down  done', remote: true);
      unawaited(_dictate(followUp: tvRemoteIdle)); // the next one, without the wake word
    } else {
      _homeNotice(said);
    }
  }

  Future<String> _tvDo(HomeCommand c) async {
    if (settings.tvIp.isEmpty || settings.tvAuth.isEmpty) return 'Pair your TV first: More › Lights and TV';
    try {
      if (c.kind == HomeKind.tvInput) {
        await tv.setInput(settings.tvIp, settings.tvAuth, c.input);
      } else {
        await tv.keys(settings.tvIp, settings.tvAuth, c.keys);
      }
      return '${c.label} ✓';
    } on VizioException catch (e) {
      return '${c.label}: $e';
    } catch (e) {
      return 'Could not reach the TV (${settings.tvName.isEmpty ? settings.tvIp : settings.tvName})';
    }
  }

  void _startTvRemote() {
    tvRemote = true;
    voice.plainWords = true; // up and down are the TV's keys now, not scrolling
    EventLog.add('home', 'tvRemote', {'on': true});
    _armTvRemote();
    _homeNotice('TV remote\nup down left right  OK  back\nhome  volume up/down  mute\npause  play  off  - say "done"', remote: true);
    unawaited(_dictate(followUp: tvRemoteIdle));
    notifyListeners();
  }

  void _armTvRemote() {
    _tvRemoteOff?.cancel();
    _tvRemoteOff = Timer(tvRemoteIdle, _endTvRemote);
  }

  void _endTvRemote() {
    if (!tvRemote) return;
    tvRemote = false;
    voice.plainWords = false;
    _tvRemoteOff?.cancel();
    EventLog.add('home', 'tvRemote', {'on': false});
    if (_glassesFree() && !conversation.busy) display.notice('TV remote off', duration: const Duration(seconds: 2));
    notifyListeners();
  }

  /// What happened, briefly on the glasses (longer while the remote is on).
  void _homeNotice(String text, {bool remote = false}) {
    if (!_glassesFree() || conversation.busy) return;
    display.notice(text, title: 'Home', raw: true, duration: remote ? tvRemoteIdle : const Duration(seconds: 3));
  }

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
        _setTarget(Target.terminal);
        _toast('Messages now go to ${terminal.label}');
        _modeNotice('> ${terminal.label}\nTalk, then pause or say "send"\n"next session" switches, "chat" goes back');
      case VoiceCommand.toChat:
        useChat();
        _toast('Messages now go to Claude chat');
        _modeNotice('Chat mode\nTalking to Claude again');
      case VoiceCommand.newSession:
        unawaited(newSession());
      case VoiceCommand.nextSession:
        unawaited(nextSession());
      case VoiceCommand.useAgy:
        switchAgent('agy', toChat: true);
      case VoiceCommand.useClaude:
        switchAgent('claude', toChat: true);
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
        if (idle) unawaited(_glassesTalk(hold: false));
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
    if (!ble.isConnected || mirror.running || position.open) return;
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
    final data = lookData();
    final rows = LookRenderer.render(lookLayout, data);
    final text = rows.join('\n');
    _lookText = text;
    if (settings.lookFrame == 'off') {
      display.showStatus(text, title: lookTitle, raw: true, immediate: immediate);
      return;
    }
    // With the HUD outline. A picture replaces whatever the glasses show, so
    // it goes first and the text after it, over it (_paintLook). Just looked
    // up: the text at once, so nothing waits on the picture.
    if (immediate && !_lookShowing) display.showStatus(text, title: lookTitle, raw: true, immediate: true);
    final key = '${settings.lookFrame}|$text';
    if (key == _frameKey) return;
    _frameKey = key;
    _framePieces = LookRenderer.pieces(lookLayout, data);
    unawaited(_paintLook(soon: immediate));
  }

  String _lookText = '';
  String _frameKey = '';
  List<LookPiece> _framePieces = const [];
  DateTime _frameSentAt = DateTime.fromMillisecondsSinceEpoch(0);
  bool _painting = false;
  bool _paintAgain = false;
  bool _paintSoon = false;
  bool _pictureLanded = false;
  final _dashShownAt = <String, DateTime>{};
  Timer? _leftCheck;
  bool _leftFixTried = false;
  Timer? _modeCheck;

  /// The built-in look-up screen only on the right eye: the right arm tells
  /// the left one to open its dashboard only while the dashboard startup
  /// mode both keep in flash is 0 (firmware imu_fusion_thread), and the left
  /// one ignores the right one while their screens differ
  /// (local_esbs_ipc_service_recv "screen id is not same"). So: read the
  /// mode on both arms, put it back to 0 if it is not, close whatever is on
  /// either screen, and send the dashboard again. Returns what it found.
  Future<String> fixLeftEye() async {
    if (!ble.isConnected) return 'Connect your glasses first';
    final before = {'L': await proto.dashStartupMode('L'), 'R': await proto.dashStartupMode('R')};
    var reset = false;
    if (before.values.any((m) => m != null && m != 0)) {
      reset = await proto.setDashStartupMode(0);
    }
    final after = reset ? {'L': await proto.dashStartupMode('L'), 'R': await proto.dashStartupMode('R')} : before;
    await proto.exit(); // both screens idle, the same on both arms
    await dashboard.sync(force: true);
    EventLog.add('dash', 'fixLeft', {'before': '${before['L']}/${before['R']}', 'reset': reset, 'after': '${after['L']}/${after['R']}'});
    String show(int? m) => m == null ? 'no answer' : '$m';
    return reset
        ? 'Dashboard startup mode was ${show(before['L'])} (left) / ${show(before['R'])} (right), now ${show(after['L'])} / ${show(after['R'])}. Look up to try.'
        : 'Startup mode ${show(before['L'])} (left) / ${show(before['R'])} (right). Both screens reset. Look up to try.';
  }
  /// Your layout with its HUD outline: the outline picture first (it
  /// replaces the screen), then the text over it - each time either changes.
  /// One picture at a time and at most one every 3 s ([soon]: now - you just
  /// looked up, or the glasses drew over ours), as each takes them about a
  /// second. If something else took the screen while the picture was on its
  /// way, that is put back.
  Future<void> _paintLook({bool soon = false}) async {
    if (_painting) {
      _paintAgain = true;
      _paintSoon = _paintSoon || soon;
      return;
    }
    _painting = true;
    _paintSoon = soon;
    try {
      do {
        _paintAgain = false;
        _pictureLanded = false;
        final wait = _paintSoon ? 0 : 3000 - DateTime.now().difference(_frameSentAt).inMilliseconds;
        _paintSoon = false;
        if (wait > 0) await Future<void>.delayed(Duration(milliseconds: wait));
        await display.idle(); // whatever text is on its way lands first
        if (!_lookShowing || !ble.isConnected) return;
        final bmp = HudFrame.bmp(settings.lookFrame, lookLayout, _framePieces);
        if (bmp != null) {
          _frameSentAt = DateTime.now();
          final ok = await proto.sendBmp(bmp, requireCrc: false);
          _pictureLanded = true;
          EventLog.add('look', 'frame',
              {'style': settings.lookFrame, 'ok': ok, 'ms': DateTime.now().difference(_frameSentAt).inMilliseconds});
        }
        if (!_lookShowing) {
          // Looked down (or something else came up) meanwhile: don't leave
          // the outline there.
          if (display.mode == DisplayMode.idle) {
            await proto.exit();
          } else {
            display.redraw();
          }
          return;
        }
        // The text, over the picture.
        display.showStatus(_lookText, title: lookTitle, raw: true, immediate: true);
        display.redraw();
        await display.idle();
      } while (_paintAgain && _lookShowing);
    } finally {
      _painting = false;
    }
  }

  void _stopLook({bool close = false}) {
    _lookTimer?.cancel();
    _lookTimer = null;
    _frameKey = '';
    _paintAgain = false;
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
    EventLog.add('termux', 'run', {'command': command, 'result': r});
    if (r == 'ok') {
      chat.markStarting();
      // Ubuntu and Claude Code take a while to start on a phone.
      for (var i = 0; i < 60; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (await chat.check()) return done;
      }
      chat.state = BridgeState.offline;
      EventLog.add('termux', 'noAnswer', {'command': command});
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
    _batteryTimer?.cancel();
    _batterySoon?.cancel();
    _redrawTimer?.cancel();
    _foreignTimer?.cancel();
    _lookTimer?.cancel();
    _leftCheck?.cancel();
    _modeCheck?.cancel();
    _tvRemoteOff?.cancel();
    _echoTimer?.cancel();
    speechModel.removeListener(_applySpeech);
    settings.removeListener(applySettings);
    voice.dispose();
    mirror.dispose();
    chat.dispose();
    terminal.dispose();
    display.dispose();
    ble.dispose();
    client.close();
    toasts.close();
    super.dispose();
  }
}
