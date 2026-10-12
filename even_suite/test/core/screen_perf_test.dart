// Screen sharing speed against simulated glasses calibrated from real logs
// of the user's G1 (2026-10-12, 2M PHY, 9856-byte pictures):
//   - packets: 51 per arm at (gap + ~1.2) ms each
//   - each arm rejects (CRC 0xCA) the FIRST send of a new picture and accepts
//     the same picture sent again - at any packet speed (seen 33 times in a
//     row on both arms); modelled pessimistically: a CRC re-check without
//     resending does not help (the app tries it, the logs will tell)
//   - end reply after the last packet: ~415 ms (accepted) / ~565 ms (first
//     send) at 5 ms spacing, ~60-120 ms at 15+ ms spacing; scales with size
//   - the glasses DISPLAY a picture only when both arms accepted it
// Compared: the previous release in the state the logs caught it in (both
// arms slowed to 20 ms, one after the other: ~0.21 fps on the real glasses)
// and the current sender, at full and half picture height.
import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

/// Measured end-reply delays (ms) at 5 ms spacing, and at 15+ ms.
const acceptedEndMs = [380, 393, 405, 413, 418, 420, 426, 432, 441, 464];
const rejectedEndMs = [423, 448, 505, 527, 564, 570, 576, 617, 673, 689];
const slowPaceEndMs = [56, 61, 64, 68, 72, 80, 95, 120];

class CalibratedGlasses implements BleTransport {
  final _rx = StreamController<BleReceive>.broadcast();
  final _batchDone = <String, DateTime>{};
  final _gap = <String, int>{};
  final _packets = <String, int>{};
  final _accepted = <String, Set<int>>{'L': {}, 'R': {}};
  // One-behind: an arm accepts a picture only if its previous transfer
  // carried the same picture.
  final _prev = <String, int?>{};
  final _cur = <String, int?>{};
  final _fresh = <String, bool>{};
  final _lastSent = <String, int>{};
  final _lastBatch = <String, int>{};
  final _resend = <String, bool>{};
  final DateTime Function() now;
  int _sample = 0;
  int frames = 0;
  final shownAt = <int>[];
  final shownTimes = <int>[];
  final int Function() elapsed;

  bool _ok(String arm) => _cur[arm] != null && _cur[arm] == _prev[arm];

  CalibratedGlasses(this.now, this.elapsed);

  @override
  Stream<BleReceive> get receive => _rx.stream;

  void _reply(String lr, List<int> data, int afterMs) {
    Timer(Duration(milliseconds: afterMs), () {
      _rx.add(BleReceive()
        ..lr = lr
        ..data = Uint8List.fromList(data)
        ..type = 'Receive');
    });
  }

  @override
  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0}) async {
    final ms = (packets.length * (gapMs + 1.22)).round();
    await Future<void>.delayed(Duration(milliseconds: ms));
    _batchDone[lr] = now();
    _gap[lr] = gapMs;
    _packets[lr] = packets.length;
    _fresh[lr] = true;
    final h = Object.hashAll(packets.map(Object.hashAll));
    _resend[lr] = _lastBatch[lr] == h;
    _lastBatch[lr] = h;
    return packets.length;
  }

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      switch (data[0]) {
        case 0x20:
          // The pace, and whether this arm just had the same picture (a
          // resend), set the delay.
          final slow = (_gap[arm] ?? 5) >= 15;
          final resend = _resend[arm] ?? false;
          final list = slow ? slowPaceEndMs : (resend ? acceptedEndMs : rejectedEndMs);
          final scale = (_packets[arm] ?? 51) / 51;
          final delay = (list[_sample++ % list.length] * scale).round();
          final since = now().difference(_batchDone[arm] ?? now()).inMilliseconds;
          _reply(arm, [0x20, 0xC9], (delay - since).clamp(5, 5000));
        case 0x16:
          frames++;
          final id = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
          if (_fresh[arm] == true) {
            _prev[arm] = _cur[arm];
            _cur[arm] = id;
            _fresh[arm] = false;
          }
          _lastSent[arm] = id;
          final ok = _ok(arm);
          if (ok) {
            _accepted[arm]!.add(id);
            final other = arm == 'L' ? 'R' : 'L';
            if (_accepted[other]!.contains(id) && !shownAt.contains(id)) {
              shownAt.add(id);
              shownTimes.add(elapsed());
            }
          }
          _reply(arm, [0x16, data[1], data[2], data[3], data[4], ok ? 0xC9 : 0xCA], 41);
        case 0x25:
          _reply(arm, [0x25, 6, 0, data[3], 0x04, data[5]], 40);
        default:
          _reply(arm, [data[0], 0xC9], 40);
      }
    }
  }
}

class Result {
  final int shown;
  final double fps;
  final int p90GapMs;
  final int transfers;
  final String pace;
  Result(this.shown, this.fps, this.p90GapMs, this.transfers, this.pace);
  @override
  String toString() => '$shown frames displayed in 60 s = ${fps.toStringAsFixed(2)} fps, '
      'p90 gap $p90GapMs ms, $transfers arm transfers, final pace $pace';
}

Future<Result> simulate(WidgetTester tester, {required bool oldBehaviour, int rows = 136}) async {
  final clock = tester.binding.clock;
  var elapsed = 0;
  final glasses = CalibratedGlasses(() => clock.now(), () => elapsed);
  final control = FakeControl();
  final ble = BleManager(glasses, control: control);
  final screen = FakeScreen();
  final proto = Proto(ble);
  final mirror = ScreenMirror(screen, proto)
    ..now = (() => clock.now())
    ..autoOff = const Duration(minutes: 10);
  mirror.rows = rows;
  if (oldBehaviour) {
    // The previous release as the logs caught it: both arms slowed to
    // 20 ms, one after the other, rejections answered by a full resend.
    proto
      ..bmpGapMs = 20
      ..adaptivePacing = false
      ..recheckFirst = false
      ..sequentialArms = true;
  }
  await control.connectNow();
  await mirror.start();

  // The phone screen changes constantly; capture follows the interval the
  // mirror asks for (it adapts to the send time).
  var tick = 0;
  while (elapsed < 60000) {
    final interval = (screen.configs.last['intervalMs'] as int?) ?? 1000;
    tick++;
    screen.frame((x, y) => (x + tick * 7) % 97 < 30, rows: rows);
    for (var t = 0; t < interval; t += 50) {
      await tester.pump(const Duration(milliseconds: 50));
      elapsed += 50;
    }
  }
  final shown = glasses.shownTimes;
  final gaps = [for (var i = 1; i < shown.length; i++) shown[i] - shown[i - 1]]..sort();
  final r = Result(
    shown.length,
    shown.length / 60,
    gaps.isEmpty ? 60000 : gaps[(gaps.length * 0.9).floor().clamp(0, gaps.length - 1)],
    glasses.frames,
    'L ${proto.armGap['L']} ms / R ${proto.armGap['R']} ms${proto.sequentialArms ? ' (sequential)' : ''}',
  );
  await mirror.stop(clearGlasses: false);
  await tester.pump(const Duration(seconds: 2));
  mirror.dispose();
  ble.dispose();
  return r;
}

void main() {
  testWidgets('screen sharing: faster than the previous release, faster still at half height', (tester) async {
    final before = await simulate(tester, oldBehaviour: true);
    final after = await simulate(tester, oldBehaviour: false);
    final half = await simulate(tester, oldBehaviour: false, rows: 68);
    // ignore: avoid_print
    print('Calibrated G1 simulation\n  previous: $before\n  now:      $after\n  half:     $half');
    expect(before.fps, lessThan(0.3), reason: 'matches the ~0.21 fps seen on the real glasses');
    expect(after.fps, greaterThanOrEqualTo(before.fps * 2), reason: 'at least twice as fast');
    expect(after.p90GapMs, lessThanOrEqualTo(2500), reason: 'no long freezes');
    expect(half.fps, greaterThan(after.fps * 1.5), reason: 'half the data, much faster');
  }, timeout: const Timeout(Duration(minutes: 4)));
}
