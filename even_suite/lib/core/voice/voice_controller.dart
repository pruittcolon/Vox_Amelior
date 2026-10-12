import 'dart:async';

import 'package:flutter/foundation.dart';

import 'speech.dart';
import 'voice_commands.dart';
import 'wake_word.dart';

enum VoiceMode {
  /// Not listening at all.
  off,

  /// Hands-free: listening only for "Claude".
  waiting,

  /// Taking down a message until "send" (or the Send button).
  dictating,
}

/// The voice state machine.
///
/// - Hands-free: the phone listens for "Claude" (or a sound-alike). What you
///   say next is the message; end it with "send". "Claude, scroll down" and
///   friends are commands. It switches itself off after [autoOff] without use.
/// - Talk button / glasses long press: start a message directly.
class VoiceController extends ChangeNotifier {
  VoiceController(
    this.engine, {
    required this.onSubmit,
    required this.onCommand,
    this.onDraft,
    this.onAutoOff,
  }) {
    _sub = engine.events.listen(_onEvent);
  }

  final SpeechEngine engine;

  /// Send a finished message (to Claude or the terminal).
  final Future<void> Function(String text) onSubmit;

  /// A spoken command other than send/cancel.
  final void Function(VoiceCommand command) onCommand;

  /// The message so far changed (shown live on the glasses).
  final void Function(String draft)? onDraft;

  /// Hands-free switched itself off.
  final void Function()? onAutoOff;

  StreamSubscription<SpeechEvent>? _sub;

  // Settings
  WakeWord wake = WakeWord();
  Duration autoOff = const Duration(minutes: 10);
  Duration dictationIdle = const Duration(seconds: 30);
  Duration releaseGrace = const Duration(milliseconds: 2500);
  bool quiet = true;
  String language = '';
  String engineName = 'auto';

  // State
  VoiceMode mode = VoiceMode.off;
  bool handsFree = false;
  bool holding = false;
  bool listening = false;
  String draft = '';
  String heard = '';
  String? error;
  String? hint;
  DateTime? autoOffAt;
  final level = ValueNotifier<double>(0);

  // Per recognition session
  String _session = ''; // latest text of the current utterance
  int _offset = 0; // where the message starts inside it
  bool _wakeSession = false; // dictation began with the wake word in this utterance
  String _draftBeforeWake = '';
  String _committed = ''; // message text from finished utterances
  bool _awaitingRelease = false;

  Timer? _autoOffTimer;
  Timer? _idleTimer;
  Timer? _releaseTimer;

  bool get active => mode != VoiceMode.off;
  bool get dictating => mode == VoiceMode.dictating;

  // ------------------------------------------------------------- controls

  Future<void> setHandsFree(bool on) async {
    if (on == handsFree) return;
    handsFree = on;
    error = null;
    if (on) {
      if (mode == VoiceMode.off) mode = VoiceMode.waiting;
      hint = null;
      _armAutoOff();
      await _ensureListening();
    } else {
      _autoOffTimer?.cancel();
      autoOffAt = null;
      if (mode == VoiceMode.waiting) {
        mode = VoiceMode.off;
        await _stopListening();
      }
    }
    notifyListeners();
  }

  /// Talk button: start a message, or stop taking it down (keeping the text).
  Future<void> toggleTalk() => dictating ? stopDictation() : startDictation();

  Future<void> startDictation({bool hold = false}) async {
    error = null;
    hint = null;
    holding = hold;
    _awaitingRelease = false;
    _releaseTimer?.cancel();
    _wakeSession = false;
    _committed = draft;
    _offset = _session.length; // words already heard belong to before
    mode = VoiceMode.dictating;
    _armIdle();
    if (handsFree) _armAutoOff();
    notifyListeners();
    onDraft?.call(draft);
    await _ensureListening();
  }

  /// Stop taking down the message but keep it (tap Send to send it).
  Future<void> stopDictation() async {
    if (!dictating) return;
    holding = false;
    _idleTimer?.cancel();
    mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
    if (!handsFree) await _stopListening();
    notifyListeners();
  }

  /// Glasses long press released: finish the utterance and send it.
  void releaseHold() {
    if (!dictating) return;
    holding = false;
    _awaitingRelease = true;
    engine.finish();
    _releaseTimer?.cancel();
    _releaseTimer = Timer(releaseGrace, () {
      if (_awaitingRelease) _submit(draft);
    });
    notifyListeners();
  }

  /// Send button.
  Future<void> send([String? text]) => _submit(text ?? draft);

  /// Throw the message away.
  Future<void> cancel() async {
    draft = '';
    _committed = '';
    _offset = _session.length;
    _awaitingRelease = false;
    holding = false;
    _idleTimer?.cancel();
    if (dictating) {
      mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
      if (!handsFree) await _stopListening();
    }
    notifyListeners();
  }

  /// The user typed in the message box.
  void setDraft(String text) {
    draft = text;
    _committed = text;
    _offset = _session.length;
    notifyListeners();
  }

  // --------------------------------------------------------------- events

  void _onEvent(SpeechEvent e) {
    switch (e.type) {
      case 'state':
        listening = e.listening;
        if (!e.listening) level.value = 0;
        notifyListeners();
      case 'ready':
        if (error != null && !handsFree && !dictating) break;
        error = null;
      case 'level':
        // Recognizer RMS is roughly -2..10 dB.
        level.value = ((e.level + 2) / 12).clamp(0.0, 1.0);
      case 'partial':
        _onPartial(e.text);
      case 'final':
        _onFinal(e.text);
      case 'end':
        _endSession();
        if (_awaitingRelease && e.finishing) _submit(draft);
      case 'error':
        error = e.message;
        if (e.fatal) {
          handsFree = false;
          mode = VoiceMode.off;
          listening = false;
          _autoOffTimer?.cancel();
          _idleTimer?.cancel();
          autoOffAt = null;
        }
        notifyListeners();
    }
  }

  void _onPartial(String text) {
    _session = text;
    if (mode == VoiceMode.waiting) {
      final m = wake.find(text);
      if (m == null) return;
      // Show "listening" right away; a command is sorted out on the final text.
      _draftBeforeWake = draft;
      _wakeSession = true;
      _committed = '';
      _offset = m.restStart;
      mode = VoiceMode.dictating;
      hint = null;
      _armAutoOff();
      _armIdle();
      _setDraft(m.rest);
      return;
    }
    if (dictating) {
      _armIdle();
      _setDraft(_join(_committed, _tail(text)));
    }
  }

  void _onFinal(String text) {
    _session = text;
    if (mode == VoiceMode.waiting) {
      final m = wake.find(text);
      if (m == null) {
        heard = text;
        notifyListeners();
      } else {
        _draftBeforeWake = draft;
        _afterWake(m.rest);
      }
      _endSession();
      return;
    }
    if (!dictating) {
      _endSession();
      return;
    }
    // The final text can differ from the partials, so find the wake word again.
    final m = _wakeSession ? wake.find(text) : null;
    final piece = m != null ? m.rest : _tail(text);
    if (_wakeSession) {
      // "Claude, scroll down" said in one go is a command, not a message.
      final cmd = VoiceCommands.parse(piece);
      if (cmd != null) {
        draft = _draftBeforeWake;
        _committed = draft;
        mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
        _idleTimer?.cancel();
        _endSession();
        _runCommand(cmd);
        notifyListeners();
        return;
      }
    }
    _committed = _join(_committed, piece);
    _endSession();
    if (_awaitingRelease) {
      _submit(_committed);
      return;
    }
    final ending = VoiceCommands.splitEnding(_committed);
    if (ending.command == VoiceCommand.send) {
      _submit(ending.body);
    } else if (ending.command == VoiceCommand.cancel) {
      cancel();
    } else {
      _setDraft(_committed);
    }
  }

  void _afterWake(String rest) {
    _armAutoOff();
    final cmd = VoiceCommands.parse(rest);
    if (cmd != null) {
      _runCommand(cmd);
      notifyListeners();
      return;
    }
    // A new message. It may already end with "send" (said in one breath).
    _committed = rest;
    mode = VoiceMode.dictating;
    _armIdle();
    final ending = VoiceCommands.splitEnding(rest);
    if (ending.command == VoiceCommand.send && ending.body.isNotEmpty) {
      _submit(ending.body);
    } else {
      _setDraft(rest);
    }
  }

  void _runCommand(VoiceCommand cmd) {
    switch (cmd) {
      case VoiceCommand.send:
        if (draft.trim().isNotEmpty) _submit(draft);
      case VoiceCommand.cancel:
        draft = '';
        _committed = '';
        notifyListeners();
      case VoiceCommand.stopListening:
        setHandsFree(false);
      default:
        onCommand(cmd);
    }
  }

  void _endSession() {
    _session = '';
    _offset = 0;
    _wakeSession = false;
  }

  String _tail(String text) => text
      .substring(_offset.clamp(0, text.length))
      .replaceFirst(RegExp(r'^[\s,.!?:;\-]+'), '')
      .trim();

  static String _join(String a, String b) {
    if (a.isEmpty) return b;
    if (b.isEmpty) return a;
    return '$a $b';
  }

  void _setDraft(String text) {
    draft = text;
    notifyListeners();
    onDraft?.call(text);
  }

  Future<void> _submit(String text) async {
    final t = text.trim();
    _awaitingRelease = false;
    _releaseTimer?.cancel();
    _idleTimer?.cancel();
    holding = false;
    draft = '';
    _committed = '';
    _offset = _session.length;
    mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
    hint = t.isEmpty ? 'Nothing to send' : null;
    notifyListeners();
    if (!handsFree) await _stopListening();
    if (t.isNotEmpty) await onSubmit(t);
  }

  // --------------------------------------------------------------- timers

  void _armAutoOff() {
    if (!handsFree) return;
    _autoOffTimer?.cancel();
    autoOffAt = DateTime.now().add(autoOff);
    _autoOffTimer = Timer(autoOff, () {
      setHandsFree(false);
      hint = 'Hands-free turned itself off';
      onAutoOff?.call();
      notifyListeners();
    });
  }

  void _armIdle() {
    _idleTimer?.cancel();
    _idleTimer = Timer(dictationIdle, () {
      if (!dictating || holding) return;
      mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
      hint = draft.isEmpty ? null : 'Paused. Tap Send, or say “Claude, send”.';
      if (!handsFree) _stopListening();
      notifyListeners();
    });
  }

  // --------------------------------------------------------------- engine

  Future<void> _ensureListening() async {
    if (listening) return;
    listening = true;
    try {
      await engine.start(
          continuous: true, language: language, quiet: quiet, engine: engineName);
    } catch (e) {
      listening = false;
      error = 'Could not start the microphone: $e';
      mode = VoiceMode.off;
      handsFree = false;
      notifyListeners();
    }
  }

  Future<void> _stopListening() async {
    if (!listening) return;
    listening = false;
    level.value = 0;
    try {
      await engine.stop();
    } catch (_) {}
  }

  @override
  void dispose() {
    _sub?.cancel();
    _autoOffTimer?.cancel();
    _idleTimer?.cancel();
    _releaseTimer?.cancel();
    level.dispose();
    super.dispose();
  }
}
