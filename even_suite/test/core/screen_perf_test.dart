// Screen sharing speed against simulated glasses calibrated from real logs
// (2026-10-12, 177 frames on the user's G1, 2M PHY, 9856-byte pictures):
//   packets: 51 per arm at (gap + ~1.2) ms each
//   an arm ACCEPTS a picture (CRC 0xC9) only when it was not sent faster
//   than it can take: left needs ~4 ms between packets, the slower right
//   arm more (at 5 ms it rejected 53 of 54); accepted pictures are
//   confirmed fast (end reply p50 ~415 ms), rejected ones slowly (~565 ms)
//   CRC reply: ~41 ms
//   the glasses DISPLAY a picture only when both arms accepted it
// Frames displayed per second are compared for the previous release and
// the current sender (per-arm pacing that adapts, retry of a rejecting arm).
import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

/// Measured end-reply delays (ms) for accepted and rejected pictures.
const acceptedEndMs = [380, 393, 405, 413, 418, 420, 426, 432, 441, 464];
const rejectedEndMs = [423, 448, 505, 527, 564, 570, 576, 617, 673, 689];

/// Slowest packet spacing (ms) each arm takes without corrupting a picture.
const needGap = {'L': 4, 'R': 10};

class CalibratedGlasses implements BleTransport {
  final _rx = StreamController<BleReceive>.broadcast();
  final _batchDone = <String, DateTime>{};
  final _gap = <String, int>{};
  final _accepted = <String, Set<int>>{'L': {}, 'R': {}};
  final DateTime Function() now;
  int _sample = 0;
  int frames = 0;
  final shownAt = <int>[];
  final shownTimes = <int>[];
  final int Function() elapsed;

  bool _ok(String arm) => (_gap[arm] ?? 0) >= needGap[arm]!;

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
    return packets.length;
  }

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      switch (data[0]) {
        case 0x20:
          final list = _ok(arm) ? acceptedEndMs : rejectedEndMs;
          final delay = list[_sample++ % list.length];
          final since = now().difference(_batchDone[arm] ?? now()).inMilliseconds;
          _reply(arm, [0x20, 0xC9], (delay - since).clamp(5, 5000));
        case 0x16:
          frames++;
          final ok = _ok(arm);
          final id = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
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

Future<Result> simulate(WidgetTester tester, {required bool oldBehaviour}) async {
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
  if (oldBehaviour) {
    // The previous release: both arms at a fixed 5 ms, no retry, a frame
    // counted once the CRC was answered.
    proto
      ..adaptivePacing = false
      ..retryRejectedArm = false;
    mirror.requireCrc = false;
  }
  await control.connectNow();
  await mirror.start();

  // The phone screen changes constantly; capture follows the interval the
  // mirror asks for (it adapts to the send time).
  var tick = 0;
  while (elapsed < 60000) {
    final interval = (screen.configs.last['intervalMs'] as int?) ?? 1000;
    tick++;
    screen.frame((x, y) => (x + tick * 7) % 97 < 30);
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
  testWidgets('screen sharing keeps the glasses display updating', (tester) async {
    final before = await simulate(tester, oldBehaviour: true);
    final after = await simulate(tester, oldBehaviour: false);
    // ignore: avoid_print
    print('Calibrated G1 simulation\n  before: $before\n  now:    $after');
    expect(before.shown, lessThanOrEqualTo(2), reason: 'the stall seen on the real glasses');
    expect(after.fps, greaterThanOrEqualTo(0.6), reason: 'the display keeps updating');
    expect(after.p90GapMs, lessThanOrEqualTo(2500), reason: 'no long freezes');
  }, timeout: const Timeout(Duration(minutes: 3)));
}
