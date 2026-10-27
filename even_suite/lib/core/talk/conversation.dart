import 'dart:async';

import 'package:flutter/foundation.dart';

import '../bridge/bridge_client.dart';
import '../glasses/glasses_display.dart';
import '../glasses/glasses_text.dart';

/// Where a back-and-forth with Claude is.
enum TalkStage {
  idle,

  /// Taking down what you say.
  listening,

  /// Sent; "Got it" plus your question on the glasses.
  sent,

  /// Claude is working (thinking, a tool step, or waiting its turn).
  working,

  /// The answer is being written (it streams onto the glasses).
  answering,

  /// The phone reads the answer aloud.
  speaking,

  /// Listening for a follow-up, no wake word needed.
  followUp,

  /// Claude did not answer.
  error,
}

/// One question and answer as you see it: the stage, the step Claude is on,
/// how long it has taken, and the matching screens on the glasses -
/// "Listening" with a voice meter and your words, "Got it" with your
/// question, "Claude · Searching the web 4s", then the answer streaming in
/// as a teleprompter.
class Conversation extends ChangeNotifier {
  Conversation(this.display);

  final GlassesDisplay display;

  /// May the glasses be drawn on now (connected, replies on, nothing else showing)?
  bool Function() canShow = () => true;

  TalkStage stage = TalkStage.idle;
  String question = '';
  String draft = '';
  double level = 0;

  /// What Claude is doing ("Thinking", "Reading server.py", ...).
  String step = '';
  List<String> steps = const [];
  String error = '';

  /// The question came from the TouchBar: use the Even AI screen states.
  bool ai = false;
  DateTime? sentAt;

  /// The answer on its way (or the last one).
  BridgeMessage? reply;

  int _frame = 0;
  Timer? _tick;
  DateTime? _gotItUntil;

  static const gotItTime = Duration(milliseconds: 1300);

  Duration get elapsed => sentAt == null ? Duration.zero : DateTime.now().difference(sentAt!);

  bool get busy => stage == TalkStage.sent || stage == TalkStage.working || stage == TalkStage.answering;

  // ------------------------------------------------------------ listening

  /// What you have said so far. In a follow-up the answer stays on the
  /// glasses until you actually start talking.
  void listening(String text, {double level = 0, bool followUp = false}) {
    draft = text;
    this.level = level;
    if (followUp && text.trim().isEmpty) {
      _set(TalkStage.followUp);
      return;
    }
    _set(TalkStage.listening);
    _startTick(const Duration(milliseconds: 450));
    _draw();
  }

  /// The voice meter moved.
  void setLevel(double v) {
    level = v;
    if (stage == TalkStage.listening) _draw();
  }

  /// Stopped listening without sending: take "Listening" off the glasses.
  void listeningStopped() {
    if (stage != TalkStage.listening) return;
    _stopTick();
    _set(TalkStage.idle);
    if (display.mode == DisplayMode.status) display.clear();
  }

  // --------------------------------------------------------------- asking

  void asked(String text, {bool ai = false}) {
    question = text.trim();
    this.ai = ai;
    sentAt = DateTime.now();
    _gotItUntil = sentAt!.add(gotItTime);
    step = 'Sending';
    steps = const [];
    error = '';
    reply = null;
    draft = '';
    _set(TalkStage.sent);
    _startTick(const Duration(seconds: 1));
    _draw();
  }

  /// Claude's progress on the answer.
  void progress(BridgeMessage m) {
    reply = m;
    steps = m.steps;
    step = switch (m.stage) {
      'queued' => 'Waiting for the last answer',
      'starting' => 'Starting',
      '' => 'Thinking',
      'thinking' => 'Thinking',
      'tool' => m.tool.isEmpty ? 'Working' : m.tool,
      'writing' => 'Writing',
      _ => step,
    };
    if (m.stage == 'writing' && m.live.trim().isNotEmpty) {
      _set(TalkStage.answering);
      if (_canShow) display.updateReply(m.live, title: 'Claude', ai: ai);
    } else {
      _set(TalkStage.working);
      _draw();
    }
  }

  /// The finished answer (or Claude's error).
  void answered(BridgeMessage m) {
    reply = m;
    steps = m.steps;
    _stopTick();
    if (m.isError) {
      failed(m.text);
      return;
    }
    step = 'Done';
    _set(TalkStage.idle);
    if (_canShow) display.updateReply(m.text, title: 'Claude', ai: ai, done: true);
  }

  void failed(String why) {
    _stopTick();
    error = why.trim().isEmpty ? 'No answer' : why.trim();
    _set(TalkStage.error);
    if (_canShow) {
      display.showStatus('Claude did not answer\n${_short(error, 90)}\nTap right to try again',
          title: 'Claude', ai: ai);
    }
  }

  void speaking() => _set(TalkStage.speaking);
  void followingUp() => _set(TalkStage.followUp);

  /// The back-and-forth is over (nothing more to say or hear).
  void ended() {
    _stopTick();
    if (stage != TalkStage.error) _set(TalkStage.idle);
  }

  // -------------------------------------------------------------- screens

  bool get _canShow => canShow();

  void _draw() {
    if (!_canShow) return;
    switch (stage) {
      case TalkStage.listening:
        display.showStatus(listeningScreen(draft, level, _frame), title: 'Listening', ai: false);
      case TalkStage.sent:
      case TalkStage.working:
        display.showStatus(workingScreen(), title: 'Claude', ai: ai);
      default:
        break;
    }
  }

  /// "Listening..." with a voice meter, then the last lines you said.
  @visibleForTesting
  static String listeningScreen(String draft, double level, int frame) {
    final dots = '.' * (1 + frame % 3);
    final bars = (2 + level * 22).round().clamp(1, 24);
    final said = GlassesText.wrap(GlassesText.clean(draft));
    final tail = said.length > 3 ? said.sublist(said.length - 3) : said;
    return ['Listening$dots', '|' * bars, if (tail.isEmpty) 'Say your question' else ...tail].join('\n');
  }

  /// "Got it" / "Claude · Thinking 4s ..", your question, the current step.
  @visibleForTesting
  String workingScreen({DateTime? now}) {
    final t = now ?? DateTime.now();
    final secs = sentAt == null ? 0 : t.difference(sentAt!).inSeconds;
    final dots = '.' * (1 + _frame % 3);
    final header = _gotItUntil != null && t.isBefore(_gotItUntil!)
        ? 'Got it'
        : _fit('Claude · ${step.isEmpty ? 'Thinking' : step} ${secs}s $dots');
    return [header, ...quoteLines(question)].join('\n');
  }

  /// Your question in quotes, at most two lines.
  @visibleForTesting
  static List<String> quoteLines(String q) {
    final lines = GlassesText.wrap('"${GlassesText.clean(q)}"');
    if (lines.length <= 2) return lines;
    var second = lines[1];
    while (second.isNotEmpty && GlassesText.wrap('$second..."').length > 1) {
      final cut = second.lastIndexOf(' ');
      second = cut <= 0 ? second.substring(0, second.length - 1) : second.substring(0, cut);
    }
    return [lines[0], '$second..."'];
  }

  static String _fit(String line) {
    var l = line;
    while (GlassesText.wrap(l).length > 1 && l.length > 10) {
      l = l.substring(0, l.length - 2);
    }
    return l;
  }

  static String _short(String s, int n) {
    final t = s.replaceAll(RegExp(r'\s+'), ' ').trim();
    return t.length <= n ? t : '${t.substring(0, n - 3)}...';
  }

  // ---------------------------------------------------------------- timer

  void _set(TalkStage s) {
    if (stage == s) return;
    stage = s;
    notifyListeners();
  }

  void _startTick(Duration every) {
    _tick?.cancel();
    _tick = Timer.periodic(every, (_) {
      _frame++;
      if (stage == TalkStage.listening || stage == TalkStage.sent || stage == TalkStage.working) {
        if (stage == TalkStage.sent && _gotItUntil != null && DateTime.now().isAfter(_gotItUntil!)) {
          stage = TalkStage.working;
        }
        _draw();
        notifyListeners(); // elapsed time on the phone
      }
    });
  }

  void _stopTick() {
    _tick?.cancel();
    _tick = null;
  }

  @override
  void dispose() {
    _stopTick();
    super.dispose();
  }
}
