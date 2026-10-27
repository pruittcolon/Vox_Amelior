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
import 'glasses/dashboard.dart';
import 'glasses/flicker.dart';
import 'glasses/glasses_display.dart';
import 'glasses/glasses_text.dart';
import 'log.dart';
import 'phone_notifications.dart';
import 'screen/picture_lab.dart';
import 'screen/screen_mirror.dart';
import 'settings.dart';
import 'system.dart';
import 'talk/conversation.dart';
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
  }) {
    settings = Settings(system.setPref);
    ble = BleManager(transport, control: control);
    proto = Proto(ble);
    display = GlassesDisplay(proto);
    this.client = client ?? BridgeClient(settings.bridgeUrl);
    chat = ChatController(this.client);
    work = WorkController(this.client);
    terminal = TerminalController(this.client, display);
    mirror = ScreenMirror(screen ?? PlatformScreen(), proto)..onFallback = _toast;
    this.speaker = speaker ?? PlatformSpeaker();
    narrator = Narrator(this.speaker);
    conversation = Conversation(display)..canShow = _glassesFree;
    dashboard = Dashboard(proto, weatherService: weather ?? WeatherService(location: system.location))
      ..paused = () => mirror.running || flicker.running;
    phone = PhoneNotifications(proto)..onForwarded = dashboard.addPhoneNote;
    voice = VoiceController(
      speech,
      onSubmit: submit,
      onCommand: runCommand,
      onDraft: _showDraft,
      onAutoOff: () => _toast('Hands-free is off (no use for ${settings.autoOffMinutes} min)'),
    )..onFollowUpEnd = _onFollowUpEnd;

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
    work.onActive = (jobs) {
      dashboard.jobs = jobs;
      dashboard.workChanged();
    };
    conversation.addListener(_onTalkChanged);
    narrator.onDone = _onSpoken;
    _events = ble.events.listen(_onGlassesEvent);
    _phoneNotes = system.phoneNotifications.listen((m) => phone.onPosted(PhoneNote.fromMap(m)));
    system.onStopListening = () => voice.setHandsFree(false);
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
    return Suite(
        system: SystemBridge()..init(),
        transport: ble,
        control: ble,
        speech: PlatformSpeech(),
        speaker: PlatformSpeaker());
  }

  final SystemBridge system;
  late final Settings settings;
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
    voice
      ..wake = WakeWord(settings.wakeWords, settings.wakeStrictness)
      ..autoOff = Duration(minutes: settings.autoOffMinutes)
      ..quiet = settings.quietChime
      ..language = settings.language
      ..engineName = settings.recognizer
      ..autoSend = settings.autoSend
      ..wakeAutoSend = settings.wakeAutoSend
      ..silenceMs = settings.pauseMs;
    chat.options = settings.claudeOptions;
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
    final dashKey = '${dashboard.mode}|${dashboard.pane}|${dashboard.h24}|${dashboard.fahrenheit}|'
        '${dashboard.weatherOn}|${dashboard.city}|${dashboard.slots}|${dashboard.custom}|${dashboard.enabled}';
    dashboard
      ..enabled = settings.dashSync
      ..mode = const ['full', 'dual', 'minimal'].indexOf(settings.dashMode).clamp(0, 2)
      ..pane = const ['notes', 'stocks', 'news', 'calendar', 'map'].indexOf(settings.dashPane).clamp(0, 4)
      ..h24 = settings.dash24h
      ..fahrenheit = settings.dashFahrenheit
      ..weatherOn = settings.dashWeather
      ..city = settings.dashCity
      ..slots = settings.noteSlots
      ..custom = settings.dashCustom;
    final newDashKey = '${dashboard.mode}|${dashboard.pane}|${dashboard.h24}|${dashboard.fahrenheit}|'
        '${dashboard.weatherOn}|${dashboard.city}|${dashboard.slots}|${dashboard.custom}|${dashboard.enabled}';
    if (newDashKey != dashKey && ble.isConnected) dashboard.markDirty(delay: const Duration(seconds: 3));
    phone
      ..forward = settings.notifyForward
      ..private = settings.notifyPrivate
      ..allowed = settings.allowedApps;
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
    chat.system = glassesPrimer();
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
    notifyListeners();
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

  void _onGlassesEvent(GlassesEvent e) {
    switch (e.code) {
      case GlassesEvent.tap:
        if (conversation.stage == TalkStage.error && e.lr == 'R') {
          retry();
        } else if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else if (e.lr == 'L') {
          display.tapLeft();
        } else {
          display.tapRight();
        }
      case GlassesEvent.exit:
        // Double tap: done with this - quiet, and no follow-up.
        display.onGlassesExit();
        terminal.onGlassesExit();
        unawaited(narrator.stop());
        unawaited(voice.endFollowUp());
        conversation.ended();
      case GlassesEvent.headUp:
        dashboard.onHeadUp();
      case GlassesEvent.headDown:
        dashboard.onHeadDown();
      case GlassesEvent.evenAiStart:
        if (settings.holdToTalk) _glassesTalk();
      case GlassesEvent.evenAiEnd:
        _glassesRelease();
    }
  }

  /// Left TouchBar held (the glasses' Even AI gesture): talk to Claude.
  /// Uses the glasses' own microphone when the phone supports it.
  Future<void> _glassesTalk() async {
    if (!await _micAllowed()) return;
    // Holding the TouchBar interrupts whatever is being said.
    await narrator.stop();
    unawaited(chat.warm());
    _aiSession = true;
    final useGlassesMic = settings.glassesMic;
    if (useGlassesMic) {
      _micOpen = true;
      unawaited(proto.micOn(true));
    }
    await voice.startDictation(hold: true, source: useGlassesMic ? 'glasses' : 'phone');
    // Same 30 s cap as the demo, in case the release never arrives.
    _holdLimit?.cancel();
    _holdLimit = Timer(const Duration(seconds: 30), _glassesRelease);
  }

  void _glassesRelease() {
    _holdLimit?.cancel();
    _holdLimit = null;
    if (voice.holding) voice.releaseHold();
    if (_micOpen) {
      _micOpen = false;
      unawaited(proto.micOn(false));
    }
  }

  /// Instructions sent with every message (as Claude's system prompt): this
  /// goes straight to the glasses, how long to be, and how to split longer
  /// answers into screens.
  String glassesPrimer() => buildPrimer(
        size: GlassesText.screenSize(),
        length: settings.answerLength,
        spoken: settings.speakReplies != 'off',
      );

  static String buildPrimer({required ScreenSize size, String length = 'brief', bool spoken = false}) {
    final chars = size.chars;
    final aim = switch (length) {
      'normal' => 'Aim for about 150 characters. Go up to $chars (one screen) only if it needs it.',
      'long' => 'Answer fully, but keep each screen-sized chunk tight.',
      _ => 'Aim for about 50 characters (one short line). Use up to 100-150 characters only when '
          'the answer needs that to make sense.',
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
        '- You run as Claude Code on the user\'s phone (Termux). You can use your tools to do what '
        'they ask; then report the result in one short line.';
  }

  void _showLastReply({bool speak = false}) {
    final r = chat.lastReply;
    if (r == null) return;
    display.showText(r.text, title: 'Claude');
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
    if (voice.source == 'glasses' && !settings.wordsWhileHolding) return;
    if (settings.target == Target.terminal) {
      if (!ble.isConnected && display.mode == DisplayMode.idle) return;
      display.showLive(draft.isEmpty ? 'Listening...\nSay your message, then "send".' : draft,
          title: 'To Terminal');
      return;
    }
    conversation.listening(draft, level: voice.level.value, followUp: voice.followUp);
  }

  void _onVoiceChanged() {
    final changed = voice.handsFree != _lastHandsFree || voice.autoOffAt != _lastAutoOffAt;
    _lastHandsFree = voice.handsFree;
    _lastAutoOffAt = voice.autoOffAt;
    if (changed) _updateService();
    // Left dictation without sending: take "Listening" off the glasses.
    // Checked a moment later, because a send replaces it with "Got it".
    if (!voice.dictating && (display.mode == DisplayMode.live || conversation.stage == TalkStage.listening)) {
      _liveCleanup?.cancel();
      _liveCleanup = Timer(const Duration(milliseconds: 600), () {
        if (voice.dictating) return;
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
      text = 'Say “Claude”, then your message and “send”. Turns off at $hh:$mm.';
    }
    final error = await system.service(listening: voice.handsFree, glasses: ble.isConnected, text: text);
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
    await narrator.stop();
    try {
      if (settings.target == Target.terminal) {
        await terminal.sendLine(t);
        if (settings.repliesOnGlasses && ble.isConnected) terminal.setMirror(true);
      } else {
        _lastQuestion = t;
        _spokenQuestion = spoken ?? voice.lastSpoken;
        _speakThis = settings.speakReplies == 'always' || (settings.speakReplies == 'voice' && _spokenQuestion);
        chat.system = glassesPrimer();
        chat.options = settings.claudeOptions;
        conversation.asked(t, ai: _aiSession);
        await chat.send(t);
      }
    } catch (e) {
      _toast('$e');
      conversation.failed('$e');
      _aiSession = false;
    }
  }

  void _onProgress(BridgeMessage m) {
    conversation.progress(m);
    if (_speakThis && narrator.enabled && m.stage == 'writing' && m.live.isNotEmpty) {
      unawaited(voice.suspend());
      narrator.feed(m.id, m.block, m.live);
      if (narrator.speaking) conversation.speaking();
    }
  }

  void _onReply(BridgeMessage m) {
    HapticFeedback.mediumImpact();
    conversation.answered(m);
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
    if (settings.conversation && _spokenQuestion && settings.target == Target.claude && !voice.dictating) {
      _spokenQuestion = false; // one follow-up window per answer
      conversation.followingUp();
      unawaited(voice.startFollowUp(Duration(seconds: settings.followUpSec)));
    } else {
      conversation.ended();
    }
  }

  void _onFollowUpEnd() => conversation.ended();

  /// A question from this app is part of "Claude's work" on the look-up screen too.
  void _onTalkChanged() {
    final c = conversation;
    final next = c.busy ? (c.step.isEmpty ? 'Thinking' : c.step, c.question) : null;
    if (next == dashboard.asking) return;
    dashboard.asking = next;
    dashboard.workChanged();
  }

  /// Claude Code finished a job (Work tab): say so briefly, only now.
  void _onWorkDone(WorkItem item) {
    dashboard.lastWork = item;
    if (settings.noteSlots.contains('work')) dashboard.markDirty();
    if (!settings.workAlerts) return;
    final where = item.project.isEmpty ? '' : ' · ${item.project}';
    if (settings.workPhone) {
      unawaited(system.notify(_workNotice++, 'Claude finished$where', item.summary));
    }
    if (settings.workGlasses && ble.isConnected) {
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
      case VoiceCommand.toChat:
        settings.set('target', Target.claude);
        _toast('Messages now go to Claude chat');
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

  // --------------------------------------------------------------- termux

  /// Start Agent Hub in Termux. Returns a message for the user.
  Future<String> startBridge() async {
    if (!await system.termuxInstalled()) return 'Termux is not installed on this phone.';
    final granted = await system.permissions(['termux']);
    if (granted['termux'] != true) {
      return 'Allow “Run commands in Termux” for Even Suite (Settings › Apps › Even Suite › Permissions), then try again.';
    }
    final r = await system.runTermux(settings.startCommand);
    if (r == 'ok') {
      chat.markStarting();
      for (var i = 0; i < 20; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (await chat.check()) return 'Termux bridge is running.';
      }
      chat.state = BridgeState.offline;
      return 'Started the command, but the bridge did not answer. In Termux, make sure '
          '~/.termux/termux.properties has allow-external-apps = true.';
    }
    if (r == 'no-permission') return 'Allow “Run commands in Termux” for Even Suite first.';
    if (r == 'not-installed') return 'Termux is not installed on this phone.';
    return 'Could not start Termux: $r';
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
