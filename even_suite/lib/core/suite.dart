import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'ble/ble_manager.dart';
import 'ble/ble_receive.dart';
import 'ble/proto.dart';
import 'bridge/bridge_client.dart';
import 'bridge/chat_controller.dart';
import 'bridge/terminal_controller.dart';
import 'glasses/flicker.dart';
import 'glasses/glasses_display.dart';
import 'glasses/glasses_text.dart';
import 'log.dart';
import 'screen/picture_lab.dart';
import 'screen/screen_mirror.dart';
import 'settings.dart';
import 'system.dart';
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
  }) {
    settings = Settings(system.setPref);
    ble = BleManager(transport, control: control);
    proto = Proto(ble);
    display = GlassesDisplay(proto);
    this.client = client ?? BridgeClient(settings.bridgeUrl);
    chat = ChatController(this.client);
    terminal = TerminalController(this.client, display);
    mirror = ScreenMirror(screen ?? PlatformScreen(), proto)..onFallback = _toast;
    voice = VoiceController(
      speech,
      onSubmit: submit,
      onCommand: runCommand,
      onDraft: _showDraft,
      onAutoOff: () => _toast('Hands-free is off (no use for ${settings.autoOffMinutes} min)'),
    );

    ble.onConnected = (pair) {
      settings.set('lastGlasses', pair.toMap());
      _toast('Glasses connected');
    };
    ble.addListener(_onLinkChanged);
    voice.addListener(_onVoiceChanged);
    chat.onReply = _onReply;
    _events = ble.events.listen(_onGlassesEvent);
    system.onStopListening = () => voice.setHandsFree(false);
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
    return Suite(system: SystemBridge()..init(), transport: ble, control: ble, speech: PlatformSpeech());
  }

  final SystemBridge system;
  late final Settings settings;
  late final BleManager ble;
  late final Proto proto;
  late final GlassesDisplay display;
  late final BridgeClient client;
  late final ChatController chat;
  late final TerminalController terminal;
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
  Timer? _heartbeat;
  Timer? _liveCleanup;
  Timer? _holdLimit;
  Timer? _mirrorTimer;
  bool _aiSession = false; // the current question came from the TouchBar
  bool _micOpen = false;
  bool _lastConnected = false;
  bool _lastHandsFree = false;
  DateTime? _lastAutoOffAt;

  /// Short messages for the UI (shown as a snackbar).
  final toasts = StreamController<String>.broadcast();

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
      ..wake = WakeWord(settings.wakeWords)
      ..autoOff = Duration(minutes: settings.autoOffMinutes)
      ..quiet = settings.quietChime
      ..language = settings.language
      ..engineName = settings.recognizer;
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
        if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else if (e.lr == 'L') {
          display.tapLeft();
        } else {
          display.tapRight();
        }
      case GlassesEvent.exit:
        display.onGlassesExit();
        terminal.onGlassesExit();
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

  /// Instructions sent with every message (as Claude's system prompt) so
  /// answers come in chunks that fit the glasses screen.
  String glassesPrimer() {
    final size = GlassesText.screenSize();
    final chars = size.chars;
    return 'You are talking to the user through Even Realities G1 smart glasses '
        '(the Even Suite app, built on Even\'s open-source demo). The user speaks; the '
        'speech-to-text may contain mistakes, so read generously.\n'
        'The glasses show ${size.lines} lines of about ${size.charsPerLine} characters: one '
        'screen holds about $chars characters. The app shows your answer one screen at a '
        'time, like a teleprompter.\n'
        'For every reply:\n'
        '- Plain text only: no Markdown, headings, tables, code blocks or emoji. Use "-" for lists.\n'
        '- Put the answer first. Fit it in one screen (under $chars characters) when you can.\n'
        '- If it needs more, write chunks of at most $chars characters separated by a blank '
        'line; each chunk becomes one screen. At most 3 chunks unless asked for more.\n'
        '- Short sentences, no filler, no preamble.';
  }

  void _showLastReply() {
    final r = chat.lastReply;
    if (r != null) display.showText(r.text, title: 'Claude');
  }

  // ---------------------------------------------------------------- voice

  /// Talk button / glasses long press.
  Future<void> startTalking({bool hold = false}) async {
    if (!await _micAllowed()) return;
    await voice.startDictation(hold: hold);
  }

  Future<void> toggleTalk() async {
    if (voice.dictating) return voice.stopDictation();
    await startTalking();
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
    if (!ble.isConnected && display.mode == DisplayMode.idle) return;
    // While the TouchBar is held the glasses show their own listening screen;
    // writing to the display then could interrupt the microphone.
    if (voice.source == 'glasses') return;
    final target = settings.target == Target.terminal ? 'Terminal' : 'Claude';
    display.showLive(draft.isEmpty ? 'Listening...\nSay your message, then "send".' : draft,
        title: 'To $target');
  }

  void _onVoiceChanged() {
    final changed = voice.handsFree != _lastHandsFree || voice.autoOffAt != _lastAutoOffAt;
    _lastHandsFree = voice.handsFree;
    _lastAutoOffAt = voice.autoOffAt;
    if (changed) _updateService();
    // Left dictation without sending: take the live text off the glasses.
    // Checked a moment later, because a send replaces it with "thinking".
    if (!voice.dictating && display.mode == DisplayMode.live) {
      _liveCleanup?.cancel();
      _liveCleanup = Timer(const Duration(milliseconds: 600), () {
        if (!voice.dictating && display.mode == DisplayMode.live) display.clear();
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

  /// Send a message to Claude, or type it into the terminal.
  Future<void> submit(String text) async {
    final t = text.trim();
    if (t.isEmpty) return;
    HapticFeedback.lightImpact();
    try {
      if (settings.target == Target.terminal) {
        await terminal.sendLine(t);
        if (settings.repliesOnGlasses && ble.isConnected) terminal.setMirror(true);
      } else {
        chat.system ??= glassesPrimer();
        if (ble.isConnected && settings.repliesOnGlasses && !mirror.running) {
          display.showText('You: $t\n\nClaude is thinking...', title: 'Claude', ai: _aiSession);
        }
        await chat.send(t);
      }
    } catch (e) {
      _toast('$e');
      if (display.mode != DisplayMode.idle && ble.isConnected) {
        display.showText('Could not send:\n$e', title: 'Error');
      }
    }
  }

  void _onReply(BridgeMessage m) {
    HapticFeedback.mediumImpact();
    // While the phone screen or flicker is on the glasses, replies stay on the phone.
    if (settings.repliesOnGlasses && ble.isConnected && !mirror.running && !flicker.running) {
      display.showText(m.isError ? 'Error: ${m.text}' : m.text, title: 'Claude', ai: _aiSession);
    }
    _aiSession = false;
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
        _showLastReply();
      case VoiceCommand.hide:
        display.clear();
        terminal.setMirror(false);
      case VoiceCommand.stop:
        if (chat.busy) {
          chat.stopReply();
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
