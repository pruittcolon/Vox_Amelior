import 'dart:async';

import 'package:flutter/foundation.dart';

import '../bridge/bridge_client.dart';
import '../glasses/claude_panel.dart';
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
  Timer? _nothingTimer;
  DateTime? _gotItUntil;

  /// The listening screen belongs to a TouchBar hold: keep the Even AI
  /// screen states, so it replaces the glasses' own "Even AI listening".
  bool _listenAi = false;

  /// How the "Claude is working" screen is arranged (Glasses layout page).
  PanelLayout layout = PanelLayout.working;

  /// How long "Got it" stays at the top before Claude's steps take over.
  Duration gotItTime = const Duration(milliseconds: 1300);

  Duration get elapsed => sentAt == null ? Duration.zero : DateTime.now().difference(sentAt!);

  bool get busy => stage == TalkStage.sent || stage == TalkStage.working || stage == TalkStage.answering;

  // ------------------------------------------------------------ listening

  /// What you have said so far. In a follow-up the answer stays on the
  /// glasses until you actually start talking.
  void listening(String text, {double level = 0, bool followUp = false, bool ai = false}) {
    draft = text;
    this.level = level;
    if (followUp && text.trim().isEmpty) {
      _set(TalkStage.followUp);
      return;
    }
    _nothingTimer?.cancel();
    final first = stage != TalkStage.listening;
    _listenAi = ai;
    _set(TalkStage.listening);
    _startTick(const Duration(milliseconds: 600));
    if (first && _canShow) {
      // At once: this is what replaces the glasses' own listening screen.
      display.showStatus(listeningScreen(draft, level, _frame), title: 'Listening', ai: _listenAi, immediate: true);
    } else {
      _draw();
    }
  }

  /// Let go of the TouchBar but nothing was heard: say so, then give the
  /// glasses back (so they never sit on a listening screen).
  void nothingHeard({bool ai = false}) {
    _stopTick();
    _set(TalkStage.idle);
    if (!_canShow) return;
    display.showStatus("Didn't catch that.\nHold the TouchBar and try again.", title: 'Claude', ai: ai, immediate: true);
    _nothingTimer?.cancel();
    _nothingTimer = Timer(const Duration(milliseconds: 2500), () {
      if (stage == TalkStage.idle && display.mode == DisplayMode.status) display.clear();
    });
  }

  /// The voice meter moved (shown on the phone; the glasses keep a calm screen).
  void setLevel(double v) => level = v;

  /// Stopped listening without sending: take "Listening" off the glasses.
  void listeningStopped() {
    if (stage != TalkStage.listening) return;
    _stopTick();
    _set(TalkStage.idle);
    if (display.mode == DisplayMode.status) display.clear();
  }

  // --------------------------------------------------------------- asking

  void asked(String text, {bool ai = false}) {
    _nothingTimer?.cancel();
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
    // Straight away, not behind the live-text throttle: this is the "it heard you" moment.
    if (_canShow) display.showStatus(workingScreen(), title: 'Claude', ai: ai, immediate: true);
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
    notifyListeners(); // the answer so far (the look-up screen follows it)
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
        display.showStatus(listeningScreen(draft, level, _frame), title: 'Listening', ai: _listenAi);
      case TalkStage.sent:
      case TalkStage.working:
        display.showStatus(workingScreen(), title: 'Claude', ai: ai);
      default:
        break;
    }
  }

  static const _dots = ['...', '..', '.', '..'];

  /// "Termux-Claude listening..." (the dots go ... .. . .. ...), then the
  /// last lines you said. Plain text: it is redrawn a few times a second.
  @visibleForTesting
  static String listeningScreen(String draft, double level, int frame) {
    final said = GlassesText.wrap(GlassesText.clean(draft));
    final tail = said.length > 4 ? said.sublist(said.length - 4) : said;
    return ['Termux-Claude listening${_dots[frame % _dots.length]}', if (tail.isEmpty) 'Say your question' else ...tail]
        .join('\n');
  }

  /// "Got it" / "Claude · 4s ..", your question, then what Claude is doing
  /// on a line of its own ("Searching the web: boston").
  @visibleForTesting
  String workingScreen({DateTime? now}) {
    final t = now ?? DateTime.now();
    final secs = sentAt == null ? 0 : t.difference(sentAt!).inSeconds;
    if (_gotItUntil != null && t.isBefore(_gotItUntil!)) {
      return ['Got it', ...quoteLines(question)].join('\n');
    }
    return [
      for (final r in workingRows(
        layout: layout,
        status: 'Claude · ${secs}s ${_dots[_frame % _dots.length]}',
        question: question,
        step: step,
        steps: steps,
      ))
        r.text,
    ].join('\n');
  }

  /// The "Claude is working" screen in [layout] order: the status line,
  /// your question (dim on the phone's preview), then Claude - the step it
  /// is on and the ones before.
  static List<PanelLine> workingRows({
    required PanelLayout layout,
    required String status,
    required String question,
    required String step,
    List<String> steps = const [],
  }) {
    final asked = layout.askLines <= 0 || question.isEmpty ? const <String>[] : quoteLines(question, max: layout.askLines);
    final meta = layout.metaLines > 0 ? [_fit(status)] : const <String>[];
    final room = (layout.total - asked.length - meta.length).clamp(1, layout.total);
    final now = step.isEmpty ? 'Thinking' : step;
    final claude = [now, ...steps.reversed.where((s) => s.isNotEmpty && s != now)].take(room).map(_fit).toList();
    final blocks = {
      'asked': [for (final l in asked) PanelLine(l, dim: true)],
      'claude': [for (final l in claude) PanelLine(l)],
      'meta': [for (final l in meta) PanelLine(l)],
    };
    return [for (final b in layout.order) ...?blocks[b]];
  }

  /// Your question in quotes, at most [max] lines (the last one cut with ...").
  @visibleForTesting
  static List<String> quoteLines(String q, {int max = 2}) {
    final lines = GlassesText.wrap('"${GlassesText.clean(q)}"');
    if (lines.length <= max) return lines;
    var last = lines[max - 1];
    while (last.isNotEmpty && GlassesText.wrap('$last..."').length > 1) {
      final cut = last.lastIndexOf(' ');
      last = cut <= 0 ? last.substring(0, last.length - 1) : last.substring(0, cut);
    }
    return [...lines.take(max - 1), '$last..."'];
  }

  /// One glasses line at most ("..." when cut).
  static String _fit(String line) {
    if (GlassesText.wrap(line).length <= 1) return line;
    var l = line;
    while (l.length > 4 && GlassesText.wrap('$l...').length > 1) {
      l = l.substring(0, l.length - 1);
    }
    return '${l.trimRight()}...';
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
    _nothingTimer?.cancel();
    super.dispose();
  }
}
