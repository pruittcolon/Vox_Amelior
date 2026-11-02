// Screen sharing speed against simulated glasses calibrated from the user's
// G1 logs (2026-10-11/12, 2M PHY):
//   - packets: (gap + ~1.2) ms each, 51 per arm for a full picture
//   - each arm checks a picture's CRC against the picture of its PREVIOUS
//     transfer: the first send of a new picture is rejected (0xCA), the same
//     picture sent again is accepted (seen on 70 frames in a row), and a
//     picture equal to the last one is accepted at once
//   - end reply after the last packet: ~565 ms for a first send, ~415 ms
//     for a repeat (5 ms spacing), proportional to the picture size
//   - the glasses display a picture only when both arms accepted it
// Compared: send-twice (what ran on the glasses: ~0.58 fps at full height)
// against fast mode (transfer the new picture, verify the previous one).
import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

const firstSendEndMs = [423, 448, 505, 527, 564, 570, 576, 617, 673, 689];
const repeatEndMs = [380, 393, 405, 413, 418, 420, 426, 432, 441, 464];

int _crcId(Uint8List bmp) {
  final c = Proto.bmpCrcCommand(bmp);
  return (c[1] << 24) | (c[2] << 16) | (c[3] << 8) | c[4];
}

class OneBehindGlasses implements BleTransport {
  OneBehindGlasses(this.elapsed);

  final int Function() elapsed;
  final _rx = StreamController<BleReceive>.broadcast();
  final _transfers = <String, List<int>>{'L': [], 'R': []};
  final _repeat = <String, bool>{};
  final _size = <String, int>{};
  final _accepted = <String, Set<int>>{'L': {}, 'R': {}};
  final _shownIds = <int>{};
  final shownTimes = <int>[];
  int transfers = 0;
  int _sample = 0;

  @override
  Stream<BleReceive> get receive => _rx.stream;

  void _reply(String lr, List<int> data, int afterMs) => Timer(Duration(milliseconds: afterMs), () {
        _rx.add(BleReceive()
          ..lr = lr
          ..data = Uint8List.fromList(data)
          ..type = 'Receive');
      });

  @override
  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0}) async {
    await Future<void>.delayed(Duration(milliseconds: (packets.length * (gapMs + 1.22)).round()));
    final bmp = Uint8List.fromList([
      for (var i = 0; i < packets.length; i++) ...packets[i].sublist(i == 0 ? 6 : 2),
    ]);
    final id = _crcId(bmp);
    final list = _transfers[lr]!;
    _repeat[lr] = list.isNotEmpty && list.last == id;
    list.add(id);
    _size[lr] = packets.length;
    transfers++;
    return packets.length;
  }

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      switch (data[0]) {
        case 0x20:
          final list = _repeat[arm] == true ? repeatEndMs : firstSendEndMs;
          final ms = (list[_sample++ % list.length] * (_size[arm] ?? 51) / 51).round();
          _reply(arm, [0x20, 0xC9], ms);
        case 0x16:
          final id = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
          final t = _transfers[arm]!;
          final ok = t.length >= 2 && t[t.length - 2] == id;
          if (ok) {
            _accepted[arm]!.add(id);
            final other = arm == 'L' ? 'R' : 'L';
            if (_accepted[other]!.contains(id) && _shownIds.add(id)) shownTimes.add(elapsed());
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

class Run {
  final String name;
  final int shown;
  final int p90GapMs;
  final int transfers;
  Run(this.name, this.shown, this.p90GapMs, this.transfers);
  double get fps => shown / 60;
  @override
  String toString() =>
      '$name: $shown frames displayed in 60 s = ${fps.toStringAsFixed(2)} fps, p90 gap $p90GapMs ms, $transfers transfers';
}

Future<Run> simulate(WidgetTester tester, String name, {required bool fast, required int rows}) async {
  final clock = tester.binding.clock;
  var elapsed = 0;
  final glasses = OneBehindGlasses(() => elapsed);
  final control = FakeControl();
  final ble = BleManager(glasses, control: control);
  final screen = FakeScreen();
  final mirror = ScreenMirror(screen, Proto(ble))
    ..now = (() => clock.now())
    ..autoOff = const Duration(minutes: 10)
    ..fpsTenths = 20 // capture up to 2 fps so the glasses are the limit
    ..rows = rows
    ..fastMode = fast;
  await control.connectNow();
  await mirror.start();
  var tick = 0;
  while (elapsed < 60000) {
    final interval = (screen.configs.last['intervalMs'] as int?) ?? 500;
    tick++;
    screen.frame((x, y) => (x + tick * 7) % 97 < 30, rows: rows);
    for (var t = 0; t < interval; t += 50) {
      await tester.pump(const Duration(milliseconds: 50));
      elapsed += 50;
    }
  }
  final s = glasses.shownTimes;
  final gaps = [for (var i = 1; i < s.length; i++) s[i] - s[i - 1]]..sort();
  final run = Run(name, s.length, gaps.isEmpty ? 60000 : gaps[(gaps.length * 0.9).floor()], glasses.transfers);
  await mirror.stop(clearGlasses: false);
  await tester.pump(const Duration(seconds: 3));
  mirror.dispose();
  ble.dispose();
  return run;
}

void main() {
  testWidgets('fast mode and half height speed up screen sharing', (tester) async {
    final twice = await simulate(tester, 'send twice, full', fast: false, rows: 136);
    final fast = await simulate(tester, 'fast mode, full ', fast: true, rows: 136);
    final half = await simulate(tester, 'fast mode, half ', fast: true, rows: 68);
    // ignore: avoid_print
    print('Calibrated G1 simulation\n  $twice\n  $fast\n  $half');
    expect(twice.fps, inInclusiveRange(0.4, 0.75), reason: 'matches ~0.58 fps measured on the glasses');
    expect(fast.fps, greaterThanOrEqualTo(twice.fps * 1.6), reason: 'one transfer per frame instead of two');
    expect(half.fps, greaterThan(fast.fps * 1.3), reason: 'half the data per frame');
    expect(fast.p90GapMs, lessThanOrEqualTo(1500), reason: 'steady');
  }, timeout: const Timeout(Duration(minutes: 4)));
}
