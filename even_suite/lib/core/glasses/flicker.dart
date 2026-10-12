import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../ble/ble_manager.dart';
import '../ble/proto.dart';
import '../log.dart';

/// How one command went during a flicker step.
class FlickSend {
  final bool ok;
  final bool answered; // false = no reply in time (or not waited for)
  final int ms;
  final String reply;
  const FlickSend(this.ok, this.ms, {this.answered = true, this.reply = ''});
  static const skipped = FlickSend(true, 0);
}

/// Flicker: moves the glasses' picture between a high and a low spot in your
/// view on a timer, with different text at each spot, to try to cover more
/// of your view than one spot does.
///
/// How the G1 firmware moves the picture (decompiled firmware, 0x26 sub 2):
/// state 1 stores the new height and starts the glasses' height preview
/// (task 0xf, the screen the Even app shows while you drag its height
/// slider). Only that preview works out where things are drawn
/// (`cal_panel_canvas_coord`: y = (8 - height) * 35 on the 480-row panel);
/// text is drawn wherever the last preview left it. State 0 only closes the
/// preview.
///
/// The preview only starts from a clear screen: `update_persist_task_status`
/// refuses task 0xf while text (task 0x10) is up, yet the glasses still
/// answer "ok". So each move: clear (0x18), wait for the glasses to go idle,
/// move (state 1), wait for the redraw at the new height, then the text
/// (which may take over from the preview at any time).
///
/// Methods: `live` leaves the preview open between moves and closes it when
/// it stops; `apply` closes it after every move (the Even app's way; the
/// right arm pauses ~0.5 s after each close).
///
/// Settings are read at every step, so changes apply while it runs. Every
/// step is logged (`flicker.step`), plus a summary each second
/// (`flicker.stats`) and at the end (`flicker.stop`).
class Flicker extends ChangeNotifier {
  Flicker(this.proto);

  final Proto proto;
  BleManager get _ble => proto.ble;

  /// Clock (tests use the fake one).
  DateTime Function() now = DateTime.now;

  // ------------------------------------------------------------ settings
  String topText = 'TOP';
  String bottomText = 'BOTTOM';
  int topMs = 100, bottomMs = 100; // how long each spot is shown
  int topHeight = 8, bottomHeight = 0; // 0 (low) .. 8 (high)
  int depth = 3; // 1 .. 9
  int homeHeight = 7, homeDepth = 3; // the normal spot, restored at stop
  String method = 'live'; // live (state 1 per move) or apply (state 1, then 0)
  bool clearFirst = true; // clear the text before each move (moves are ignored while text is up)
  int clearMs = 80; // pause after clearing, so the glasses are idle
  int settleMs = 150; // pause after a move (the glasses redraw there), before the text
  int previewMs = 300; // apply: time between state 1 and state 0
  bool waitForReplies = true; // steadier; off = just hand it to Bluetooth
  String what = 'both'; // both, move (spot only), text (text only)
  String arms = 'both'; // moves to both arms (each arm works out its own height) or the right arm
  int autoStopSec = 60;
  int replyTimeoutMs = 250;

  // --------------------------------------------------------------- state
  bool running = false;
  bool busy = false; // a one-off step or the reset is being sent
  bool top = true; // the spot being shown (or shown next)
  int steps = 0;
  int slow = 0; // steps that took longer than their time
  int failures = 0; // commands without an answer, or rejected
  double rate = 0; // switches a second over the last second
  int lastClearMs = 0, lastMoveMs = 0, lastTextMs = 0, lastStepMs = 0;
  String moveReply = ''; // what the glasses answer to a move
  bool moveSilent = false; // the glasses never answered moves: stop waiting
  bool textSilent = false;
  String? error;
  DateTime? startedAt;
  DateTime? offAt;

  int _seq = 0;
  int _textSeq = 0;
  int _gen = 0;
  bool _previewOpen = false; // a state-1 move without its state 0 yet
  bool _textUp = true; // something may be on the glasses (then clear before a move)
  String? _lastText;
  String _loggedConfig = '';
  int _moveMiss = 0, _textMiss = 0;
  Timer? _statsTimer;
  Timer? _autoStop;
  Timer? _changeLog;
  Timer? _sleepTimer;
  Completer<void>? _sleeping;

  // Figures for the per-second summary.
  final _winMove = <int>[], _winText = <int>[], _winStep = <int>[];
  int _winSteps = 0, _winLate = 0, _winFail = 0;
  DateTime? _winStart;

  int _ms() => now().millisecondsSinceEpoch;

  /// Asked-for switches a second.
  double get askedRate => 2000 / (topMs + bottomMs).clamp(1, 1 << 30);

  Map<String, Object?> config() => {
        'topMs': topMs,
        'bottomMs': bottomMs,
        'topH': topHeight,
        'bottomH': bottomHeight,
        'depth': depth,
        'homeH': homeHeight,
        'homeD': homeDepth,
        'method': method,
        'clear': clearFirst,
        'clearMs': clearMs,
        'settleMs': settleMs,
        'previewMs': previewMs,
        'wait': waitForReplies,
        'what': what,
        'arms': arms,
        'autoStopSec': autoStopSec,
        'topLen': topText.length,
        'bottomLen': bottomText.length,
      };

  /// Called when settings change; logged (once things settle, and only if
  /// something flicker uses changed) while running.
  void changed() {
    if (!running) return;
    _changeLog?.cancel();
    _changeLog = Timer(const Duration(milliseconds: 600), () {
      final c = config();
      final key = '$c';
      if (key == _loggedConfig) return;
      _loggedConfig = key;
      EventLog.add('flicker', 'change', c);
    });
  }

  // ---------------------------------------------------------- start/stop

  Future<void> start() async {
    if (running || busy) return;
    if (!_ble.isConnected) {
      error = 'Connect your glasses first';
      notifyListeners();
      return;
    }
    running = true;
    error = null;
    steps = slow = failures = 0;
    rate = 0;
    top = true;
    moveSilent = textSilent = false;
    _moveMiss = _textMiss = 0;
    _lastText = null;
    _textUp = true;
    startedAt = now();
    offAt = autoStopSec > 0 ? startedAt!.add(Duration(seconds: autoStopSec)) : null;
    final gen = ++_gen;
    _autoStop?.cancel();
    if (autoStopSec > 0) _autoStop = Timer(Duration(seconds: autoStopSec), () => stop(reason: 'auto'));
    _resetWindow();
    _statsTimer?.cancel();
    _statsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _logStats());
    final c = config();
    _loggedConfig = '$c';
    EventLog.add('flicker', 'start', c);
    notifyListeners();
    unawaited(_loop(gen));
  }

  /// [reason]: user, auto (timer), link (glasses disconnected), other.
  Future<void> stop({String reason = 'user', bool restore = true}) async {
    if (!running) return;
    running = false;
    _gen++;
    _wake();
    _autoStop?.cancel();
    _statsTimer?.cancel();
    _changeLog?.cancel();
    _logStats();
    final secs = startedAt == null ? 0.0 : (_ms() - startedAt!.millisecondsSinceEpoch) / 1000;
    EventLog.add('flicker', 'stop', {
      'reason': reason,
      'steps': steps,
      'secs': double.parse(secs.toStringAsFixed(1)),
      'rate': secs > 0 ? double.parse((steps / secs).toStringAsFixed(2)) : 0,
      'asked': double.parse(askedRate.toStringAsFixed(2)),
      'slow': slow,
      'failures': failures,
      'moveSilent': moveSilent,
      'textSilent': textSilent,
    });
    offAt = null;
    notifyListeners();
    if (restore) await reset();
  }

  /// Back to the normal spot and the glasses' dashboard: clear the text,
  /// move there (state 1), close the height preview (state 0), clear.
  Future<void> reset() async {
    if (!_ble.isConnected || busy) return;
    busy = true;
    notifyListeners();
    final t0 = _ms();
    var moved = false;
    var ok = true;
    try {
      if (what != 'text' || _previewOpen) {
        final lrs = arms == 'both' ? const ['L', 'R'] : const ['R'];
        if (_textUp) {
          await _send([_exitPacket], const ['L', 'R'], wait: true);
          _textUp = false;
          await _pause(clearMs);
        }
        final open = await _send([Proto.positionPacket(_seq++, height: homeHeight, depth: homeDepth, preview: true)], lrs,
            wait: true);
        await _pause(previewMs);
        final close = await _send([Proto.positionPacket(_seq++, height: homeHeight, depth: homeDepth)], lrs, wait: true);
        ok = open.ok && close.ok;
        moved = true;
        _previewOpen = false;
      }
      ok = await proto.exit() && ok;
      _textUp = false;
    } finally {
      _lastText = null;
      busy = false;
      notifyListeners();
    }
    EventLog.add('flicker', 'reset', {'homeH': homeHeight, 'homeD': homeDepth, 'moved': moved, 'ok': ok, 'ms': _ms() - t0});
  }

  /// One step by hand: show the top (or bottom) text at its spot.
  Future<void> showOnce(bool atTop) async {
    if (running || busy) return;
    if (!_ble.isConnected) {
      error = 'Connect your glasses first';
      notifyListeners();
      return;
    }
    busy = true;
    error = null;
    top = atTop;
    notifyListeners();
    try {
      _lastText = null;
      _textUp = true; // other screens may have put something up meanwhile
      final t0 = _ms();
      final (c, m, t) = await _step(atTop, wait: true);
      lastClearMs = c.ms;
      lastMoveMs = m.ms;
      lastTextMs = t.ms;
      lastStepMs = _ms() - t0;
      if (m.reply.isNotEmpty) moveReply = m.reply;
      if (!c.ok || !m.ok || !t.ok) failures++;
      EventLog.add('flicker', 'test', {
        'side': atTop ? 'top' : 'bottom',
        'h': atTop ? topHeight : bottomHeight,
        'ms': lastStepMs,
        'clearMs': c.ms,
        'moveMs': m.ms,
        'textMs': t.ms,
        'clearOk': c.ok,
        'moveOk': m.ok,
        'textOk': t.ok,
        'moveReply': m.reply,
        'textReply': t.reply,
        ...config(),
      });
    } finally {
      busy = false;
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------- loop

  Future<void> _loop(int gen) async {
    while (running && gen == _gen) {
      final side = top;
      final target = side ? topMs : bottomMs;
      final t0 = _ms();
      final (c, m, t) = await _step(side, wait: waitForReplies);
      if (!running || gen != _gen) break;
      final took = _ms() - t0;
      final isLate = took > target;
      final ok = c.ok && m.ok && t.ok;
      steps++;
      lastClearMs = c.ms;
      lastMoveMs = m.ms;
      lastTextMs = t.ms;
      lastStepMs = took;
      if (isLate) slow++;
      if (!ok) failures++;
      _winSteps++;
      if (isLate) _winLate++;
      if (!ok) _winFail++;
      if (m.answered && what != 'text') _winMove.add(m.ms);
      if (t.answered && t.ms > 0) _winText.add(t.ms);
      _winStep.add(took);
      final newReply = m.reply.isNotEmpty && m.reply != moveReply;
      if (m.reply.isNotEmpty) moveReply = m.reply;
      EventLog.add('flicker', 'step', {
        'n': steps,
        'side': side ? 'top' : 'bottom',
        'h': side ? topHeight : bottomHeight,
        'target': target,
        'ms': took,
        if (c.ms > 0) 'clearMs': c.ms,
        'moveMs': m.ms,
        'textMs': t.ms,
        'ok': ok,
        if (isLate) 'late': true,
        if (!m.answered && what != 'text' && waitForReplies && !moveSilent) 'moveTimeout': true,
        if (!t.answered && t.ms > 0 && waitForReplies && !textSilent) 'textTimeout': true,
        if (newReply) 'moveReply': m.reply,
      });
      top = !side;
      notifyListeners();
      final wait = target - took;
      if (wait > 0) await _sleep(wait);
    }
  }

  /// A wait that [stop] can cut short.
  Future<void> _sleep(int ms) {
    final c = _sleeping = Completer<void>();
    _sleepTimer = Timer(Duration(milliseconds: ms), () {
      if (!c.isCompleted) c.complete();
    });
    return c.future;
  }

  void _wake() {
    _sleepTimer?.cancel();
    final c = _sleeping;
    if (c != null && !c.isCompleted) c.complete();
  }

  /// A short fixed pause inside a step (not cut short by [stop]).
  Future<void> _pause(int ms) => ms > 0 ? Future<void>.delayed(Duration(milliseconds: ms)) : Future.value();

  /// One spot: clear, move there, then (re)draw its text at the new height.
  /// Returns how the clear, the move and the text went.
  Future<(FlickSend, FlickSend, FlickSend)> _step(bool atTop, {required bool wait}) async {
    final height = atTop ? topHeight : bottomHeight;
    final text = atTop ? topText : bottomText;
    var c = FlickSend.skipped;
    var m = FlickSend.skipped;
    if (what != 'text') {
      c = await _clear(wait: wait);
      m = await _move(height, wait: wait);
      if (what == 'both') await _pause(settleMs);
    }
    // After a move the text is drawn again even if it is the same.
    final t = what == 'move' ? FlickSend.skipped : await _text(text, wait: wait, force: what == 'both');
    return (c, m, t);
  }

  static final _exitPacket = Uint8List.fromList([0x18]);

  /// Clears the text so the glasses go idle; they only move from idle.
  Future<FlickSend> _clear({required bool wait}) async {
    if (!clearFirst || !_textUp) return FlickSend.skipped;
    final r = await _send([_exitPacket], const ['L', 'R'], wait: wait && !textSilent);
    _textUp = false;
    _lastText = null;
    await _pause(clearMs);
    return r;
  }

  Future<FlickSend> _move(int height, {required bool wait}) async {
    final lrs = arms == 'both' ? const ['L', 'R'] : const ['R'];
    final w = wait && !moveSilent;
    final open = Proto.positionPacket(_seq++, height: height, depth: depth, preview: true);
    var r = await _send([open], lrs, wait: w);
    _previewOpen = true;
    if (method == 'apply') {
      await _pause(previewMs);
      final close = await _send([Proto.positionPacket(_seq++, height: height, depth: depth)], lrs, wait: w);
      _previewOpen = false;
      r = FlickSend(r.ok && close.ok, r.ms + previewMs + close.ms,
          answered: r.answered && close.answered, reply: r.reply.isNotEmpty ? r.reply : close.reply);
    }
    if (w) {
      _moveMiss = r.answered ? 0 : _moveMiss + 1;
      if (_moveMiss >= 3 && running) {
        moveSilent = true;
        EventLog.add('flicker', 'silent', {'what': 'move', 'note': 'no answer to 3 moves in a row; not waiting any more'});
      }
    }
    return r;
  }

  Future<FlickSend> _text(String text, {required bool wait, bool force = false}) async {
    // Text only: the same text is still on the glasses.
    if (!force && text == _lastText) return FlickSend.skipped;
    final packets = Proto.evenaiPackets(utf8.encode(text.isEmpty ? ' ' : text),
        syncSeq: _textSeq++ & 0xff, newScreen: Proto.newScreen(Proto.textShow), pos: 0, currentPage: 1, maxPage: 1);
    final w = wait && !textSilent;
    _textUp = true;
    final r = await _send(packets, const ['L', 'R'], wait: w);
    if (r.ok) _lastText = text;
    if (w) {
      _textMiss = r.answered ? 0 : _textMiss + 1;
      if (_textMiss >= 3 && running) {
        textSilent = true;
        EventLog.add('flicker', 'silent', {'what': 'text', 'note': 'no answer to 3 texts in a row; not waiting any more'});
      }
    }
    return r;
  }

  /// [packets] to each arm (arms in parallel). With [wait], every packet's
  /// answer is awaited; without, they are only handed to Bluetooth.
  Future<FlickSend> _send(List<Uint8List> packets, List<String> lrs, {required bool wait}) async {
    final t0 = _ms();
    if (!wait) {
      await Future.wait([
        for (final lr in lrs)
          for (final p in packets) _ble.sendData(p, lr: lr),
      ]);
      return FlickSend(true, _ms() - t0, answered: false);
    }
    var ok = true;
    var answered = true;
    var reply = '';
    await Future.wait([
      for (final lr in lrs)
        () async {
          for (final p in packets) {
            final r = await _ble.request(p, lr: lr, timeoutMs: replyTimeoutMs, log: false);
            if (r.isTimeout) {
              ok = answered = false;
              return;
            }
            if (reply.isEmpty || lr == 'R') {
              reply = '$lr:${r.data.take(6).map((b) => b.toRadixString(16).padLeft(2, '0')).join(' ')}';
            }
            if (r.data.length > 1 && r.data[1] == 0xCA) ok = false;
          }
        }(),
    ]);
    return FlickSend(ok, _ms() - t0, answered: answered, reply: reply);
  }

  // ------------------------------------------------------------- figures

  void _resetWindow() {
    _winMove.clear();
    _winText.clear();
    _winStep.clear();
    _winSteps = _winLate = _winFail = 0;
    _winStart = now();
  }

  static int _pct(List<int> v, double p) {
    if (v.isEmpty) return 0;
    final s = List.of(v)..sort();
    return s[((s.length - 1) * p).round()];
  }

  void _logStats() {
    final start = _winStart;
    if (start == null) return;
    final secs = (_ms() - start.millisecondsSinceEpoch) / 1000;
    if (secs <= 0) return;
    rate = _winSteps / secs;
    if (_winSteps > 0 || _winFail > 0) {
      EventLog.add('flicker', 'stats', {
        'steps': _winSteps,
        'rate': double.parse(rate.toStringAsFixed(2)),
        'asked': double.parse(askedRate.toStringAsFixed(2)),
        'late': _winLate,
        'fail': _winFail,
        'moveP50': _pct(_winMove, .5),
        'moveP90': _pct(_winMove, .9),
        'textP50': _pct(_winText, .5),
        'textP90': _pct(_winText, .9),
        'stepP50': _pct(_winStep, .5),
        'stepP90': _pct(_winStep, .9),
        'stepMax': _winStep.isEmpty ? 0 : _winStep.reduce((a, b) => a > b ? a : b),
      });
    }
    _resetWindow();
    notifyListeners();
  }

  @override
  void dispose() {
    _gen++;
    running = false;
    _wake();
    _statsTimer?.cancel();
    _autoStop?.cancel();
    _changeLog?.cancel();
    super.dispose();
  }
}
