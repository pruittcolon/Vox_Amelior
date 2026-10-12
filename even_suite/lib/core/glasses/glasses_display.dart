import 'dart:async';

import 'package:flutter/foundation.dart';

import '../ble/ble_manager.dart';
import '../ble/proto.dart';
import 'glasses_text.dart';

enum DisplayMode { idle, reply, live, mirror }

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

  /// The reply answers a TouchBar question: use the Even AI screen states
  /// (0x30 scrolling, 0x40 last page, 0x50 manual) like the demo.
  bool ai = false;

  bool _manual = false;
  bool _sending = false;
  bool _dirty = false;
  String _shownKey = '';
  DateTime _lastSend = DateTime.fromMillisecondsSinceEpoch(0);
  Timer? _throttle;
  Timer? _auto;
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
    mode = DisplayMode.reply;
    this.title = title;
    this.ai = ai;
    _manual = false;
    paused = false;
    pager = GlassesPager.fromText(text);
    _schedule(immediate: true);
    _armAuto(first: true);
  }

  /// Text that keeps changing (what you are saying): newest lines shown.
  void showLive(String text, {String title = 'Listening'}) {
    _stopAuto();
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
    final p = pager!;
    if (p.atEnd) return;
    _auto = Timer(pageTime(first: first), () {
      _auto = null;
      if (mode != DisplayMode.reply || paused) return;
      if (p.down()) {
        _schedule(immediate: true);
        _armAuto();
      }
      notifyListeners();
    });
  }

  /// Clear the glasses (back to their dashboard).
  Future<void> clear() async {
    _reset();
    notifyListeners();
    if (_ble.isConnected) await proto.exit();
  }

  /// The user double-tapped on the glasses: they already left our screen.
  void onGlassesExit() {
    _reset();
    notifyListeners();
  }

  void _reset() {
    _stopAuto();
    _throttle?.cancel();
    _throttle = null;
    mode = DisplayMode.idle;
    pager = null;
    title = '';
    ai = false;
    paused = false;
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

  void _schedule({bool immediate = false}) {
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
        await _sendCurrent();
      } while (_dirty && mode != DisplayMode.idle);
    } finally {
      _sending = false;
    }
  }

  /// 0x4E screen status for what is shown.
  int get _status {
    if (!ai || mode != DisplayMode.reply) return Proto.textShow;
    final p = pager!;
    if (_manual || paused) return Proto.aiManual;
    if (p.atEnd && p.pageCount > 1) return Proto.aiComplete;
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
    final tries = mode == DisplayMode.reply ? 3 : 1;
    var ok = false;
    for (var i = 0; i < tries && !ok; i++) {
      if (_dirty) return; // something newer is waiting
      ok = await proto.sendEvenAIData(screen,
          newScreen: Proto.newScreen(status),
          currentPage: p.page + 1,
          maxPage: p.pageCount);
    }
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
    super.dispose();
  }
}
