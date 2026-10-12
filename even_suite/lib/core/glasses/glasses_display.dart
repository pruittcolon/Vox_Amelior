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

  /// Seconds between automatic page turns for replies (0 = only by hand).
  int autoScrollSeconds = 0;

  bool _sending = false;
  bool _dirty = false;
  String _shownKey = '';
  DateTime _lastSend = DateTime.fromMillisecondsSinceEpoch(0);
  Timer? _throttle;
  Timer? _auto;
  bool _wasConnected = false;

  BleManager get _ble => proto.ble;

  /// Lines of the screen currently on (or meant for) the glasses.
  List<String> get previewLines {
    final p = pager;
    if (p == null || mode == DisplayMode.idle) return const [];
    return p.screen().split('\n').where((l) => l.isNotEmpty).toList();
  }

  String get pageLabel {
    final p = pager;
    if (p == null || mode == DisplayMode.idle) return '';
    return '${p.page + 1} / ${p.pageCount}';
  }

  bool get canUp => pager != null && mode != DisplayMode.idle && !pager!.atStart;
  bool get canDown => pager != null && mode != DisplayMode.idle && !pager!.atEnd;

  // ------------------------------------------------------------------ show

  /// A finished text (a Claude reply, a note): starts on the first screen.
  void showText(String text, {String title = 'Text'}) {
    _stopAuto();
    mode = DisplayMode.reply;
    this.title = title;
    pager = GlassesPager.fromText(text);
    _schedule(immediate: true);
    if (autoScrollSeconds > 0 && pager!.pageCount > 1) {
      _auto = Timer.periodic(Duration(seconds: autoScrollSeconds), (t) {
        if (!(pager?.down() ?? false)) {
          t.cancel();
        } else {
          _schedule(immediate: true);
        }
      });
    }
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
    _schedule();
  }

  // ---------------------------------------------------------------- paging

  bool up() => _page(() => pager?.up() ?? false);
  bool down() => _page(() => pager?.down() ?? false);
  bool top() => _page(() => pager?.top() ?? false);
  bool bottom() => _page(() => pager?.bottom() ?? false);

  bool _page(bool Function() move) {
    if (mode == DisplayMode.idle) return false;
    _stopAuto();
    final moved = move();
    if (moved) _schedule(immediate: true);
    return moved;
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

  Future<void> _sendCurrent() async {
    final p = pager;
    if (p == null || mode == DisplayMode.idle || !_ble.isConnected) return;
    final screen = p.screen();
    final key = '${p.page}/${p.pageCount}|$screen';
    if (key == _shownKey) return;
    _lastSend = DateTime.now();
    // Live screens are replaced soon anyway; a finished reply gets retries.
    final tries = mode == DisplayMode.reply ? 3 : 1;
    var ok = false;
    for (var i = 0; i < tries && !ok; i++) {
      if (_dirty) return; // something newer is waiting
      ok = await proto.sendEvenAIData(screen,
          newScreen: Proto.newScreen(Proto.textShow),
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
