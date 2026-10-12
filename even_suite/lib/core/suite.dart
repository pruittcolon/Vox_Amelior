import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'ble/ble_manager.dart';
import 'ble/ble_receive.dart';
import 'ble/proto.dart';
import 'bridge/bridge_client.dart';
import 'bridge/chat_controller.dart';
import 'bridge/terminal_controller.dart';
import 'glasses/glasses_display.dart';
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
  }) {
    settings = Settings(system.setPref);
    ble = BleManager(transport, control: control);
    proto = Proto(ble);
    display = GlassesDisplay(proto);
    this.client = client ?? BridgeClient(settings.bridgeUrl);
    chat = ChatController(client);
    terminal = TerminalController(client, display);
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

  StreamSubscription<GlassesEvent>? _events;
  Timer? _heartbeat;
  Timer? _liveCleanup;
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
    display.autoScrollSeconds = settings.autoScrollSeconds;
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
    if (connected) {
      // Same 8 s heartbeat as the demo keeps the link from idling out.
      _heartbeat = Timer.periodic(const Duration(seconds: 8), (_) => proto.sendHeartBeat());
    }
    _updateService();
  }

  void _onGlassesEvent(GlassesEvent e) {
    switch (e.code) {
      case GlassesEvent.tap:
        if (display.mode == DisplayMode.idle) {
          _showLastReply();
        } else if (e.lr == 'L') {
          display.up();
        } else {
          display.down();
        }
      case GlassesEvent.exit:
        display.onGlassesExit();
        terminal.onGlassesExit();
      case GlassesEvent.evenAiStart:
        if (settings.holdToTalk) startTalking(hold: true);
      case GlassesEvent.evenAiEnd:
        if (voice.holding) voice.releaseHold();
    }
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
        if (ble.isConnected && settings.repliesOnGlasses) {
          display.showText('You: $t\n\nClaude is thinking...', title: 'Claude');
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
    if (settings.repliesOnGlasses && ble.isConnected) {
      display.showText(m.isError ? 'Error: ${m.text}' : m.text, title: 'Claude');
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
    settings.removeListener(applySettings);
    voice.dispose();
    chat.dispose();
    terminal.dispose();
    display.dispose();
    ble.dispose();
    client.close();
    toasts.close();
    super.dispose();
  }
}
