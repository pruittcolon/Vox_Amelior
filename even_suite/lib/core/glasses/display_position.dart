import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../ble/proto.dart';
import '../log.dart';

/// Where text sits in your view - the Even app's "display height" and
/// "distance" sliders, done the same way (its own code names this
/// startOrEndSettingDisplayHeight / updateDisplayHeight):
///
/// 1. clear the screen (the glasses only open their height preview from
///    their idle screen),
/// 2. `26 08 00 seq 02 01 <height> <depth>` - stores the setting and shows
///    the preview (their dashboard at the new spot), sent again while you
///    drag,
/// 3. `26 08 00 seq 02 00 ...` when you are done - closes the preview.
///
/// The glasses take 0.5-1.5 s to answer a move (they redraw first), so only
/// one move is ever on its way; newer slider values wait and only the
/// latest is sent. (Flicker failed because it asked for a move every 0.1 s.)
class DisplayPosition extends ChangeNotifier {
  DisplayPosition(this.proto);

  final Proto proto;

  /// 0 (low) .. 8 (high), and 1 (near) .. 9 (far).
  int height = 4;
  int depth = 5;

  /// The preview is open on the glasses.
  bool open = false;
  bool _busy = false;
  bool _pending = false;
  int _seq = 0;
  String? error;

  bool get _connected => proto.ble.isConnected;

  /// Open the preview at [h] / [d] (the page opened).
  Future<void> begin(int h, int d) async {
    height = h.clamp(0, 8);
    depth = d.clamp(1, 9);
    if (!_connected) return;
    open = true;
    notifyListeners();
    await proto.exit();
    await Future<void>.delayed(const Duration(milliseconds: 120));
    await _flush();
  }

  /// A slider moved: sent now, or right after the move on its way.
  void set({int? h, int? d}) {
    height = (h ?? height).clamp(0, 8);
    depth = (d ?? depth).clamp(1, 9);
    notifyListeners();
    if (!open || !_connected) return;
    _pending = true;
    if (!_busy) unawaited(_flush());
  }

  /// Done: close the preview; the glasses keep the new spot.
  Future<void> end() async {
    if (!open) return;
    while (_busy) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
    }
    open = false;
    notifyListeners();
    if (_connected) await _send(preview: false);
    EventLog.add('position', 'set', {'height': height, 'depth': depth, 'error': error});
  }

  Future<void> _flush() async {
    _busy = true;
    try {
      do {
        _pending = false;
        await _send(preview: true);
      } while (_pending && open);
    } finally {
      _busy = false;
    }
  }

  /// To each arm in turn, waiting for its answer (as the Even app does).
  Future<bool> _send({required bool preview}) async {
    final p = Proto.positionPacket(_seq++, height: height, depth: depth, preview: preview);
    var ok = true;
    for (final lr in const ['L', 'R']) {
      final r = await proto.ble.request(p, lr: lr, timeoutMs: 2000);
      ok = ok && !r.isTimeout && _ok(r.data);
    }
    error = ok ? null : 'The glasses did not confirm the move';
    notifyListeners();
    return ok;
  }

  /// The glasses answer `26 06 00 <seq> 02 c9`.
  static bool _ok(Uint8List d) => d.length > 1 && d[0] == 0x26 && (d[1] == 0xc9 || (d.length > 5 && d[5] == 0xc9));
}
