// Screen sharing speed against simulated glasses calibrated from real logs
// (2026-10-12, 123 frames on the user's G1, 2M PHY):
//   packets: 51 per arm at ~6.2 ms each with a 5 ms gap  (~317 ms)
//   end reply after the last packet: 381..1119 ms, median ~560 ms
//   CRC reply: ~41 ms, status 0xCA (mismatch) - the worst case
// The same frames are run through the old behaviour (CRC mismatch = miss,
// 1 s pause, resend) and the current one, and frame rates are compared.
import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

/// Measured end-reply delays (ms), every 6th of the sorted log sample.
const measuredEndMs = [
  381, 415, 423, 431, 434, 441, 447, 451, 463, 536, 562, //
  569, 574, 584, 617, 666, 675, 687, 715, 889, 1119,
];

class CalibratedGlasses implements BleTransport {
  final _rx = StreamController<BleReceive>.broadcast();
  final _batchDone = <String, DateTime>{};
  final DateTime Function() now;
  int _sample = 0;
  int frames = 0;

  CalibratedGlasses(this.now);

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
    return packets.length;
  }

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      switch (data[0]) {
        case 0x20:
          final delay = measuredEndMs[(_sample++ ~/ 2) % measuredEndMs.length];
          final since = now().difference(_batchDone[arm] ?? now()).inMilliseconds;
          _reply(arm, [0x20, 0xC9], (delay - since).clamp(5, 5000));
        case 0x16:
          if (arm == 'R') frames++;
          _reply(arm, [0x16, data[1], data[2], data[3], data[4], 0xCA], 41);
        case 0x25:
          _reply(arm, [0x25, 6, 0, data[3], 0x04, data[5]], 40);
        default:
          _reply(arm, [data[0], 0xC9], 40);
      }
    }
  }
}

class Result {
  final int delivered;
  final double fps;
  final int p90GapMs;
  final int attempts;
  Result(this.delivered, this.fps, this.p90GapMs, this.attempts);
  @override
  String toString() =>
      '$delivered new frames in 60 s = ${fps.toStringAsFixed(2)} fps, p90 gap $p90GapMs ms, $attempts transfers';
}

Future<Result> simulate(WidgetTester tester, {required bool oldBehaviour}) async {
  final clock = tester.binding.clock;
  final glasses = CalibratedGlasses(() => clock.now());
  final control = FakeControl();
  final ble = BleManager(glasses, control: control);
  final screen = FakeScreen();
  final mirror = ScreenMirror(screen, Proto(ble))
    ..now = (() => clock.now())
    ..autoOff = const Duration(minutes: 10);
  if (oldBehaviour) {
    mirror
      ..requireCrc = true
      ..coolDown = const Duration(seconds: 1);
  }
  await control.connectNow();
  await mirror.start();

  // The phone screen changes constantly; capture follows the interval the
  // mirror asks for (it adapts to the send time).
  var elapsed = 0;
  var tick = 0;
  final deliveredAt = <int>[];
  var lastSent = 0;
  while (elapsed < 60000) {
    final interval = (screen.configs.last['intervalMs'] as int?) ?? 1000;
    tick++;
    screen.frame((x, y) => (x + tick * 7) % 97 < 30);
    for (var t = 0; t < interval; t += 50) {
      await tester.pump(const Duration(milliseconds: 50));
      elapsed += 50;
      if (mirror.sent != lastSent) {
        lastSent = mirror.sent;
        deliveredAt.add(elapsed);
      }
    }
  }
  final gaps = [for (var i = 1; i < deliveredAt.length; i++) deliveredAt[i] - deliveredAt[i - 1]]..sort();
  final r = Result(
    mirror.sent,
    mirror.sent / 60,
    gaps.isEmpty ? 60000 : gaps[(gaps.length * 0.9).floor().clamp(0, gaps.length - 1)],
    glasses.frames,
  );
  await mirror.stop(clearGlasses: false);
  await tester.pump(const Duration(seconds: 2));
  mirror.dispose();
  ble.dispose();
  return r;
}

void main() {
  testWidgets('screen sharing reaches ~1 fps on glasses that answer like the real ones', (tester) async {
    final before = await simulate(tester, oldBehaviour: true);
    final after = await simulate(tester, oldBehaviour: false);
    // ignore: avoid_print
    print('Calibrated G1 simulation\n  before: $before\n  now:    $after');
    expect(after.fps, greaterThanOrEqualTo(0.9), reason: 'about one new frame a second');
    expect(after.p90GapMs, lessThanOrEqualTo(1300), reason: 'steady: 9 in 10 frames within 1.3 s');
    expect(after.fps, greaterThanOrEqualTo(before.fps * 2), reason: 'at least twice the old rate');
  }, timeout: const Timeout(Duration(minutes: 3)));
}
