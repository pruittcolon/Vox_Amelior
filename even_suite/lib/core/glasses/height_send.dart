import 'dart:async';

import 'package:flutter/foundation.dart';

import '../ble/proto.dart';
import '../log.dart';
import 'display_position.dart';

/// One spot in your view: a height (0 low .. 8 high), its text, and how long
/// it stays before the next one.
class HeightSpot {
  HeightSpot(this.height, this.text, this.holdMs);
  int height;
  String text;
  int holdMs;
}

/// Text at two heights, one after the other: move the text to the spot (the
/// same move as the Text height sliders, which the glasses accept), send its
/// text, wait, then do the same at the other spot. Settings are read at every
/// step, so changing them while it runs takes effect on the next one.
///
/// A move takes the glasses about 0.5-1.5 s, so the real speed is the hold
/// time plus that; [lastMoveMs] / [lastTextMs] / [cycleMs] show what it was.
class HeightSend extends ChangeNotifier {
  HeightSend(this.proto, this.position);

  final Proto proto;
  final DisplayPosition position;

  final top = HeightSpot(8, 'TOP', 1000);
  final bottom = HeightSpot(0, 'BOTTOM', 1000);

  bool running = false;
  bool busy = false;
  int steps = 0;
  int lastMoveMs = 0, lastTextMs = 0, cycleMs = 0;
  String? error;

  int _gen = 0;

  bool get connected => proto.ble.isConnected;

  /// Move to [spot] and show its text once.
  Future<void> sendOnce(HeightSpot spot) async {
    if (busy || running) return;
    await _step(spot);
  }

  Future<void> start() async {
    if (running || busy) return;
    if (!connected) {
      error = 'Connect your glasses first';
      notifyListeners();
      return;
    }
    running = true;
    steps = 0;
    error = null;
    final gen = ++_gen;
    notifyListeners();
    EventLog.add('heightsend', 'start', _config());
    unawaited(_loop(gen));
  }

  Future<void> stop() async {
    if (!running) return;
    running = false;
    _gen++;
    EventLog.add('heightsend', 'stop', {'steps': steps, 'cycleMs': cycleMs});
    notifyListeners();
  }

  Map<String, Object?> _config() => {
        'topH': top.height,
        'bottomH': bottom.height,
        'topMs': top.holdMs,
        'bottomMs': bottom.holdMs,
        'depth': position.depth,
      };

  Future<void> _loop(int gen) async {
    var atTop = true;
    while (running && gen == _gen && connected) {
      final spot = atTop ? top : bottom;
      final t0 = DateTime.now();
      await _step(spot);
      if (!running || gen != _gen) return;
      final left = spot.holdMs - DateTime.now().difference(t0).inMilliseconds;
      if (left > 0) await Future<void>.delayed(Duration(milliseconds: left));
      cycleMs = DateTime.now().difference(t0).inMilliseconds;
      atTop = !atTop;
    }
    if (gen == _gen && running) {
      running = false; // the link dropped
      notifyListeners();
    }
  }

  Future<void> _step(HeightSpot spot) async {
    if (!connected) return;
    busy = true;
    notifyListeners();
    try {
      final t0 = DateTime.now();
      await position.begin(spot.height, position.depth); // clear, then move
      await position.end(); // keep the spot
      final t1 = DateTime.now();
      final ok = await proto.sendEvenAIData(spot.text.isEmpty ? ' ' : spot.text,
          newScreen: Proto.newScreen(Proto.textShow), currentPage: 1, maxPage: 1);
      final t2 = DateTime.now();
      lastMoveMs = t1.difference(t0).inMilliseconds;
      lastTextMs = t2.difference(t1).inMilliseconds;
      steps++;
      error = ok && position.error == null ? null : 'The glasses did not confirm a step';
      EventLog.add('heightsend', 'step', {
        'h': spot.height,
        'moveMs': lastMoveMs,
        'textMs': lastTextMs,
        'ok': ok,
        'n': steps,
      });
    } finally {
      busy = false;
      notifyListeners();
    }
  }
}
