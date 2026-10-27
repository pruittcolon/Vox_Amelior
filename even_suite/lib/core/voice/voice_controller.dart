import 'dart:async';

import 'package:flutter/foundation.dart';

import '../log.dart';
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

  /// A follow-up window closed without a question (or with "thanks").
  void Function()? onFollowUpEnd;

  /// A message ended with nothing in it (e.g. the TouchBar was let go
  /// before anything was heard).
  void Function()? onNothing;

  StreamSubscription<SpeechEvent>? _sub;

  // Settings
  WakeWord wake = WakeWord();
  Duration autoOff = const Duration(minutes: 10);
  Duration dictationIdle = const Duration(seconds: 30);
  Duration releaseGrace = const Duration(milliseconds: 2500);
  bool quiet = true;
  String language = '';
  String engineName = 'auto';

  /// Send when you stop talking (mic button, TouchBar, follow-ups) instead
  /// of waiting for "send".
  bool autoSend = true;

  /// Same for hands-free messages that start with the wake word. Off by
  /// default: then a stray "Claude" never sends anything without "send".
  bool wakeAutoSend = false;

  /// How long a pause ends what you are saying.
  int silenceMs = 1500;

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

  /// Where the audio comes from: 'phone', or 'glasses' while the TouchBar is held.
  String source = 'phone';

  /// The recognizer confirmed it is hearing the glasses' microphone.
  bool glassesMic = false;

  /// Listening for a follow-up right after Claude answered (no wake word).
  bool followUp = false;

  /// The phone is talking: the microphone stays off so it does not hear itself.
  bool suspended = false;

  /// The last message sent was spoken (not typed in the box).
  bool lastSpoken = false;
  bool _draftTyped = false;
  final level = ValueNotifier<double>(0);

  // Per recognition session
  String _session = ''; // latest text of the current utterance
  int _offset = 0; // where the message starts inside it
  bool _wakeSession = false; // dictation began with the wake word in this utterance
  bool _wakeStarted = false; // the message being taken down began with the wake word
  String _draftBeforeWake = '';
  String _committed = ''; // message text from finished utterances
  bool _awaitingRelease = false;

  Timer? _autoOffTimer;
  Timer? _idleTimer;
  Timer? _releaseTimer;
  Timer? _followTimer;

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

  Future<void> startDictation({bool hold = false, String source = 'phone', bool followUp = false}) async {
    _followTimer?.cancel();
    this.followUp = followUp;
    suspended = false;
    _wakeStarted = false;
    _draftTyped = false;
    if (listening && this.source != source) await _stopListening();
    this.source = source;
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

  /// Right after an answer: listen for a follow-up question for [window],
  /// without the wake word. Saying nothing (or "thanks") ends it.
  Future<void> startFollowUp(Duration window) async {
    if (dictating) return;
    await startDictation(followUp: true);
    _followTimer?.cancel();
    _followTimer = Timer(window, () {
      if (followUp && draft.trim().isEmpty) endFollowUp();
    });
    notifyListeners();
  }

  /// Close the follow-up window (nothing to send).
  Future<void> endFollowUp() async {
    if (!followUp) return;
    followUp = false;
    _followTimer?.cancel();
    _idleTimer?.cancel();
    draft = '';
    _committed = '';
    _offset = _session.length;
    if (dictating) mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
    EventLog.add('voice', 'followUpEnd', {});
    await _afterMessage();
    notifyListeners();
    onFollowUpEnd?.call();
  }

  /// Turn the microphone off while the phone speaks (it would hear itself).
  Future<void> suspend() async {
    if (suspended) return;
    suspended = true;
    await _stopListening();
    notifyListeners();
  }

  /// The phone stopped speaking: listen again if anything needs it.
  Future<void> resume() async {
    if (!suspended) return;
    suspended = false;
    if (handsFree || dictating) await _ensureListening();
    notifyListeners();
  }

  /// Stop taking down the message but keep it (tap Send to send it).
  Future<void> stopDictation() async {
    _followTimer?.cancel();
    followUp = false;
    if (!dictating) return;
    holding = false;
    _idleTimer?.cancel();
    mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
    await _afterMessage();
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
    _wakeStarted = false;
    draft = '';
    _committed = '';
    _offset = _session.length;
    _awaitingRelease = false;
    holding = false;
    _idleTimer?.cancel();
    if (dictating) {
      mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
      await _afterMessage();
    }
    notifyListeners();
  }

  /// The user typed in the message box.
  void setDraft(String text) {
    _draftTyped = true;
    draft = text;
    _committed = text;
    _offset = _session.length;
    notifyListeners();
  }

  // --------------------------------------------------------------- events

  void _onEvent(SpeechEvent e) {
    if (e.type == 'final' || e.type == 'error' || e.type == 'state' || e.type == 'source' || e.type == 'engine') {
      EventLog.add('voice', e.type, {
        if (e.type == 'final') 'text': e.text,
        if (e.type == 'error') 'message': e.message,
        if (e.type == 'error') 'fatal': e.fatal,
        if (e.type == 'state') 'listening': e.listening,
        if (e.type == 'source') 'glassesMic': e.glassesMic,
        'mode': mode.name,
        'src': source,
      });
    }
    switch (e.type) {
      case 'state':
        listening = e.listening;
        if (!e.listening) level.value = 0;
        notifyListeners();
      case 'ready':
        if (error != null && !handsFree && !dictating) break;
        error = null;
      case 'source':
        glassesMic = e.glassesMic;
        notifyListeners();
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
    if (followUp) _followTimer?.cancel(); // they are answering
    if (mode == VoiceMode.waiting) {
      final m = wake.find(text);
      if (m == null) return;
      // Show "listening" right away; a command is sorted out on the final text.
      _draftBeforeWake = draft;
      _wakeSession = true;
      _wakeStarted = true;
      _draftTyped = false;
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
    if (followUp && _committed.isEmpty) {
      if (VoiceCommands.endsConversation(piece)) {
        _endSession();
        endFollowUp();
        return;
      }
      // "scroll down", "repeat", "stop" work without the wake word here.
      final cmd = VoiceCommands.parse(piece);
      if (cmd != null && cmd != VoiceCommand.send && cmd != VoiceCommand.cancel) {
        _endSession();
        _runCommand(cmd);
        _followTimer?.cancel();
        if (followUp) _followTimer = Timer(const Duration(seconds: 6), () => endFollowUp());
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
    final auto = (_wakeStarted ? wakeAutoSend : autoSend) && !holding;
    if (ending.command == VoiceCommand.send) {
      _submit(ending.body);
    } else if (ending.command == VoiceCommand.cancel) {
      cancel();
    } else if (auto && _committed.trim().isNotEmpty) {
      _submit(_committed);
    } else {
      _setDraft(_committed);
    }
  }

  void _afterWake(String rest) {
    EventLog.add('voice', 'wake', {'rest': rest});
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
    _wakeStarted = true;
    _draftTyped = false;
    _armIdle();
    final ending = VoiceCommands.splitEnding(rest);
    if (ending.command == VoiceCommand.send && ending.body.isNotEmpty) {
      _submit(ending.body);
    } else if (wakeAutoSend && rest.trim().isNotEmpty && ending.command == null) {
      _submit(rest);
    } else {
      _setDraft(rest);
    }
  }

  void _runCommand(VoiceCommand cmd) {
    EventLog.add('voice', 'command', {'cmd': cmd.name});
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
    _draftTyped = false;
    draft = text;
    notifyListeners();
    onDraft?.call(text);
  }

  Future<void> _submit(String text) async {
    final t = text.trim();
    lastSpoken = !_draftTyped;
    _draftTyped = false;
    EventLog.add('voice', 'submit',
        {'chars': t.length, 'src': source, 'hold': holding || _awaitingRelease, 'followUp': followUp});
    _awaitingRelease = false;
    _wakeStarted = false;
    followUp = false;
    _followTimer?.cancel();
    _releaseTimer?.cancel();
    _idleTimer?.cancel();
    holding = false;
    draft = '';
    _committed = '';
    _offset = _session.length;
    mode = handsFree ? VoiceMode.waiting : VoiceMode.off;
    hint = t.isEmpty ? 'Nothing to send' : null;
    notifyListeners();
    await _afterMessage();
    if (t.isNotEmpty) {
      await onSubmit(t);
    } else {
      onNothing?.call();
    }
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
      hint = draft.isEmpty ? null : 'Paused. Tap Send, or say “${wake.label}, send”.';
      _afterMessage();
      notifyListeners();
    });
  }

  // --------------------------------------------------------------- engine

  /// A message is done: the mic goes off, or back to the phone for hands-free.
  Future<void> _afterMessage() async {
    if (!handsFree) {
      await _stopListening();
      source = 'phone';
      return;
    }
    if (source != 'phone') {
      await _stopListening();
      source = 'phone';
      await _ensureListening();
    }
  }

  Future<void> _ensureListening() async {
    if (listening || suspended) return;
    listening = true;
    try {
      await engine.start(
          continuous: true,
          language: language,
          quiet: quiet && source == 'phone' && !followUp,
          engine: engineName,
          audioSource: source,
          silenceMs: silenceMs);
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
    glassesMic = false;
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
    _followTimer?.cancel();
    level.dispose();
    super.dispose();
  }
}
