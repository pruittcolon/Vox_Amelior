import 'dart:async';

import 'package:flutter/foundation.dart';

import '../ble/ble_manager.dart';
import '../ble/proto.dart';
import '../log.dart';
import 'glasses_text.dart';

enum DisplayMode { idle, reply, live, mirror, status }

/// Everything the app shows on the glasses goes through here.
///
/// - one transfer at a time; a newer screen requested mid-transfer replaces
///   any older one that has not been sent yet;
/// - an identical screen is never resent;
/// - live text (dictation, terminal) is throttled so BLE is not flooded;
/// - replies scroll like a teleprompter: each page stays up as long as it
///   takes to read at [wpm] words per minute. Right tap = next (and keep
///   going), left tap = back and pause;
/// - the phone shows the same screen as a preview, glasses connected or not.
class GlassesDisplay extends ChangeNotifier {
  GlassesDisplay(this.proto, {this.liveGap = const Duration(milliseconds: 400)}) {
    proto.ble.addListener(_onLink);
    _wasConnected = proto.ble.isConnected;
  }

  final Proto proto;
  final Duration liveGap;

  DisplayMode mode = DisplayMode.idle;
  GlassesPager? pager;
  String title = '';
  String? error;

  /// Reading speed for automatic page turns; 0 = only by hand.
  int wpm = 0;

  /// Teleprompter paused (by a left tap, "pause", or the phone).
  bool paused = false;

  /// Go back to the dashboard after this long without changes (after the
  /// last page, plus the time to read it); null = stay until closed by hand.
  Duration? closeAfter = const Duration(seconds: 15);

  /// Something still needs the screen (you or the phone are talking, Claude
  /// is working): it waits.
  bool Function() holdOpen = () => false;

  /// Use the Even AI screen states for TouchBar answers. Off by default: in
  /// those states the glasses fade every update in (8 steps) and add a scroll
  /// bar, which looks jumpy for screens that change often. Plain text (0x70)
  /// is drawn as it is.
  bool aiStates = false;

  /// The reply answers a TouchBar question: with [aiStates], use the Even AI
  /// screen states (0x30 scrolling, 0x40 last page, 0x50 manual) like the demo.
  bool ai = false;

  bool _manual = false;
  bool _streaming = false;
  bool _sending = false;
  bool _dirty = false;
  bool _exitPending = false; // clear() came while a screen was still going out
  String _shownKey = '';
  DateTime _lastSend = DateTime.fromMillisecondsSinceEpoch(0);
  Timer? _throttle;
  Timer? _auto;
  Timer? _close;
  Timer? _notice;
  bool _wasConnected = false;

  static const wpmChoices = [0, 140, 180, 230];

  BleManager get _ble => proto.ble;

  /// Lines of the screen currently on (or meant for) the glasses.
  List<String> get previewLines {
    final p = pager;
    if (p == null || mode == DisplayMode.idle) return const [];
    return p.pageLines;
  }

  String get pageLabel {
    final p = pager;
    if (p == null || mode == DisplayMode.idle) return '';
    return '${p.page + 1} / ${p.pageCount}';
  }

  bool get canUp => pager != null && mode != DisplayMode.idle && !pager!.atStart;
  bool get canDown => pager != null && mode != DisplayMode.idle && !pager!.atEnd;

  /// A reply is turning its own pages right now.
  bool get scrolling => _auto != null;

  /// Teleprompter is available for what is shown (a multi-page reply).
  bool get canScroll => mode == DisplayMode.reply && wpm > 0 && (pager?.pageCount ?? 0) > 1;

  // ------------------------------------------------------------------ show

  /// A finished text (a Claude reply, a note): starts on the first screen and
  /// scrolls at reading speed when [wpm] is set.
  void showText(String text, {String title = 'Text', bool ai = false}) {
    _stopAuto();
    _streaming = false;
    mode = DisplayMode.reply;
    this.title = title;
    this.ai = ai;
    _manual = false;
    paused = false;
    pager = GlassesPager.fromText(text);
    _schedule(immediate: true);
    _armAuto(first: true);
  }

  /// A reply that is still being written: the pages grow as text arrives,
  /// and the teleprompter carries on into new pages. The page you are on
  /// stays put. [done] marks the finished answer (the last page then uses
  /// the "complete" screen state).
  void updateReply(String text, {String title = 'Claude', bool ai = false, bool done = false}) {
    if (mode != DisplayMode.reply || !_streaming || pager == null) {
      showText(text, title: title, ai: ai);
      _streaming = !done;
      return;
    }
    final page = pager!.page;
    final wasAtEnd = pager!.atEnd;
    pager = GlassesPager.fromText(text)..jumpTo(page);
    this.title = title;
    this.ai = ai;
    _streaming = !done;
    // The teleprompter had run out of pages: keep going now that there are more.
    if (_auto == null && wasAtEnd && !pager!.atEnd && !paused) _armAuto();
    _schedule(immediate: done);
  }

  /// A short screen that keeps changing (listening, "Claude is thinking"):
  /// shown as given, at most one screen, updates throttled like live text.
  void showStatus(String text, {String title = 'Claude', bool ai = false, bool immediate = false}) {
    _stopAuto();
    _streaming = false;
    final lines = GlassesText.wrap(GlassesText.clean(text));
    mode = DisplayMode.status;
    pager = GlassesPager(lines.take(GlassesText.linesPerScreen).toList());
    this.title = title;
    this.ai = ai;
    _schedule(immediate: immediate);
  }

  /// A short notice that goes away by itself after [duration] (unless
  /// something else is shown meanwhile).
  void notice(String text, {Duration duration = const Duration(seconds: 4), String title = 'Notice'}) {
    showStatus(text, title: title, immediate: true);
    final shown = pager;
    _notice?.cancel();
    _notice = Timer(duration, () {
      _notice = null;
      if (identical(pager, shown) && mode == DisplayMode.status) clear();
    });
  }

  /// A tiny sign (e.g. "*" when Claude is done) for [duration], only if the
  /// glasses show nothing of ours right now.
  bool flash(String text, {Duration duration = const Duration(seconds: 3)}) {
    if (mode != DisplayMode.idle || !_ble.isConnected) return false;
    notice(text, duration: duration, title: 'Done');
    return true;
  }

  /// Text that keeps changing (what you are saying): newest lines shown.
  void showLive(String text, {String title = 'Listening'}) {
    _stopAuto();
    _streaming = false;
    final lines = GlassesText.wrap(GlassesText.clean(text));
    if (mode != DisplayMode.live || pager == null) {
      mode = DisplayMode.live;
      pager = GlassesPager(lines, followTail: true);
    } else {
      pager!.update(lines);
    }
    this.title = title;
    ai = false;
    _schedule();
  }

  /// A mirrored screen (Termux / Claude Code): follows the bottom unless the
  /// user scrolled back.
  void showMirror(List<String> lines, {String title = 'Terminal'}) {
    _stopAuto();
    _streaming = false;
    if (mode != DisplayMode.mirror || pager == null) {
      mode = DisplayMode.mirror;
      pager = GlassesPager(lines, followTail: true);
    } else {
      pager!.update(lines);
    }
    this.title = title;
    ai = false;
    _schedule();
  }

  // ---------------------------------------------------------------- paging

  bool up() => _page(() => pager?.up() ?? false);
  bool down() => _page(() => pager?.down() ?? false);
  bool top() => _page(() => pager?.top() ?? false);
  bool bottom() => _page(() => pager?.bottom() ?? false);

  /// Right TouchBar: next page, and keep scrolling from there.
  void tapRight() {
    if (mode == DisplayMode.idle) return;
    final moved = down();
    if (canScroll && (paused || moved)) {
      paused = false;
      _armAuto();
      notifyListeners();
    }
  }

  /// Left TouchBar: back a page and hold it there.
  void tapLeft() {
    if (mode == DisplayMode.idle) return;
    up();
    if (canScroll) pause();
  }

  void pause() {
    if (paused && _auto == null) return;
    paused = true;
    _stopAuto();
    notifyListeners();
  }

  void resume() {
    if (!canScroll) return;
    paused = false;
    _armAuto();
    notifyListeners();
  }

  void togglePause() {
    if (paused || !scrolling) {
      resume();
    } else {
      pause();
    }
  }

  /// Change reading speed (from the phone or "faster" / "slower").
  void setWpm(int value) {
    wpm = value;
    if (value == 0) {
      _stopAuto();
    } else if (canScroll && !paused) {
      _armAuto();
    }
    notifyListeners();
  }

  void faster() => setWpm(_nextWpm(1));
  void slower() => setWpm(_nextWpm(-1));

  int _nextWpm(int dir) {
    final speeds = wpmChoices.where((w) => w > 0).toList();
    if (wpm == 0) return dir > 0 ? speeds.first : 0;
    var i = speeds.indexWhere((w) => w >= wpm);
    if (i < 0) i = speeds.length - 1;
    final next = i + dir;
    if (next < 0) return 0;
    return speeds[next.clamp(0, speeds.length - 1)];
  }

  bool _page(bool Function() move) {
    if (mode == DisplayMode.idle) return false;
    _manual = true;
    final moved = move();
    if (moved) {
      if (_auto != null) _armAuto(); // keep scrolling, from the new page
      _schedule(immediate: true);
    }
    return moved;
  }

  /// How long the current page stays up when scrolling.
  @visibleForTesting
  Duration pageTime({bool first = false}) {
    final words = pager?.pageWords ?? 0;
    final ms = (words * 60000 / (wpm == 0 ? 180 : wpm)).round() + (first ? 1200 : 600);
    return Duration(milliseconds: ms.clamp(2500, 15000));
  }

  void _armAuto({bool first = false}) {
    _stopAuto();
    if (!canScroll || paused) return;
    if (pager!.atEnd) return;
    _auto = Timer(pageTime(first: first), () {
      _auto = null;
      // The pager may have been rebuilt meanwhile (a reply still streaming in).
      final p = pager;
      if (p == null || mode != DisplayMode.reply || paused) return;
      if (p.down()) {
        _schedule(immediate: true);
        _armAuto();
      }
      notifyListeners();
    });
  }

  /// Clear the glasses (back to their dashboard). Goes out after a screen
  /// still being sent, never before it (or that screen would stay up).
  Future<void> clear() async {
    _reset();
    notifyListeners();
    if (!_ble.isConnected) return;
    _exitPending = true;
    await _render();
  }

  /// The user double-tapped on the glasses: they already left our screen.
  void onGlassesExit() {
    _reset();
    notifyListeners();
  }

  void _reset() {
    _stopAuto();
    _close?.cancel();
    _close = null;
    _throttle?.cancel();
    _throttle = null;
    mode = DisplayMode.idle;
    pager = null;
    title = '';
    ai = false;
    paused = false;
    _streaming = false;
    _shownKey = '';
  }

  void _stopAuto() {
    _auto?.cancel();
    _auto = null;
  }

  // -------------------------------------------------------------- sending

  void _onLink() {
    final now = _ble.isConnected;
    if (now && !_wasConnected && mode != DisplayMode.idle) {
      _shownKey = ''; // the glasses lost our screen while away
      _schedule(immediate: true);
    }
    _wasConnected = now;
  }

  /// (Re)start the countdown that gives the glasses back.
  void _armClose() {
    _close?.cancel();
    _close = null;
    final after = closeAfter;
    if (after == null || mode == DisplayMode.idle || mode == DisplayMode.mirror || mode == DisplayMode.live) return;
    var wait = after;
    if (mode == DisplayMode.reply) wait += pageTime();
    if (paused) wait *= 4;
    _close = Timer(wait, () {
      _close = null;
      if (_streaming || _auto != null || holdOpen()) {
        _armClose();
        return;
      }
      EventLog.add('display', 'autoClose', {'mode': mode.name});
      clear();
    });
  }

  void _schedule({bool immediate = false}) {
    _exitPending = false; // a new screen replaces the dashboard anyway
    _armClose();
    notifyListeners();
    if (!_ble.isConnected) return;
    final wait = liveGap - DateTime.now().difference(_lastSend);
    if (immediate || wait <= Duration.zero) {
      _throttle?.cancel();
      _throttle = null;
      _render();
    } else {
      _throttle ??= Timer(wait, () {
        _throttle = null;
        _render();
      });
    }
  }

  Future<void> _render() async {
    if (_sending) {
      _dirty = true;
      return;
    }
    _sending = true;
    try {
      do {
        _dirty = false;
        if (mode == DisplayMode.idle) {
          if (_exitPending && _ble.isConnected) {
            _exitPending = false;
            await proto.exit();
          }
        } else {
          await _sendCurrent();
        }
      } while (_dirty);
    } finally {
      _sending = false;
    }
  }

  /// 0x4E screen status for what is shown.
  int get _status {
    if (!ai || !aiStates) return Proto.textShow;
    if (mode == DisplayMode.status) return Proto.aiShowing;
    if (mode != DisplayMode.reply) return Proto.textShow;
    final p = pager!;
    if (_manual || paused) return Proto.aiManual;
    if (p.atEnd && p.pageCount > 1 && !_streaming) return Proto.aiComplete;
    return Proto.aiShowing;
  }

  Future<void> _sendCurrent() async {
    final p = pager;
    if (p == null || mode == DisplayMode.idle || !_ble.isConnected) return;
    final screen = p.screen();
    final status = _status;
    final key = '${p.page}/${p.pageCount}|$status|$screen';
    if (key == _shownKey) return;
    _lastSend = DateTime.now();
    // Live screens are replaced soon anyway; a finished reply gets retries.
    final tries = mode == DisplayMode.reply && !_streaming ? 3 : 1;
    var ok = false;
    var tried = 0;
    final sw = Stopwatch()..start();
    for (var i = 0; i < tries && !ok; i++) {
      if (_dirty || !identical(pager, p)) return; // something newer is waiting (or it was closed)
      tried++;
      ok = await proto.sendEvenAIData(screen,
          newScreen: Proto.newScreen(status),
          currentPage: p.page + 1,
          maxPage: p.pageCount);
    }
    EventLog.add('display', 'screen', {
      'mode': mode.name,
      'page': p.page + 1,
      'pages': p.pageCount,
      'status': status.toRadixString(16),
      'chars': screen.length,
      'ms': sw.elapsedMilliseconds,
      'tries': tried,
      'ok': ok,
      'auto': _auto != null,
    });
    if (ok) {
      _shownKey = key;
      error = null;
    } else {
      error = 'The glasses did not confirm the last screen';
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _ble.removeListener(_onLink);
    _throttle?.cancel();
    _auto?.cancel();
    _close?.cancel();
    _notice?.cancel();
    super.dispose();
  }
}
