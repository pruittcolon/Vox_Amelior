import 'dart:io';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/screen/mono_encoder.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

Uint8List grayOf(int Function(int x, int y) f) {
  final g = Uint8List(576 * 136);
  for (var y = 0; y < 136; y++) {
    for (var x = 0; x < 576; x++) {
      g[y * 576 + x] = f(x, y);
    }
  }
  return g;
}

/// Lit flag of pixel (x, y) read back from the BMP bytes.
bool litAt(Uint8List bmp, int x, int y) {
  final row = 135 - y; // bottom-up
  final byte = bmp[62 + row * 72 + x ~/ 8];
  return (byte >> (7 - x % 8)) & 1 == 0; // 0 = white = lit
}

void main() {
  group('1-bit encoder', () {
    test('BMP header matches the demo images', () {
      final demo = File('assets/images/image_2.bmp').readAsBytesSync();
      final mine = MonoEncoder.encode(grayOf((x, y) => 0)).bmp;
      expect(mine.length, demo.length, reason: '9856 bytes like the demo (2 padding bytes)');
      expect(mine.sublist(0, 30), demo.sublist(0, 30)); // BM, file size, offset, header, w, h, planes, bpp
      expect(mine.sublist(34, 54), demo.sublist(34, 54)); // image size, 72 dpi, colours
      expect(mine.sublist(54, 62), demo.sublist(54, 62)); // palette: white, black
      expect(mine.sublist(mine.length - 2), [0, 0]);
    });

    test('bright pixels are lit, rows are stored bottom-up', () {
      final g = grayOf((x, y) => (x < 10 && y < 5) ? 250 : 5);
      final f = MonoEncoder.encode(g, invert: 'off');
      expect(litAt(f.bmp, 0, 0), isTrue);
      expect(litAt(f.bmp, 9, 4), isTrue);
      expect(litAt(f.bmp, 10, 0), isFalse);
      expect(litAt(f.bmp, 0, 5), isFalse);
      expect(litAt(f.bmp, 575, 135), isFalse);
    });

    test('Otsu finds the gap between two gray levels', () {
      final hist = List<int>.filled(256, 0)
        ..[40] = 500
        ..[200] = 300;
      final t = MonoEncoder.otsu(hist, 800);
      expect(t, inInclusiveRange(40, 199));
    });

    test('auto invert: a light app (dark text on white) becomes lit text on black', () {
      // White page with a dark "text" block.
      final g = grayOf((x, y) => (x > 100 && x < 200 && y > 50 && y < 60) ? 20 : 240);
      final f = MonoEncoder.encode(g);
      expect(f.flipped, isTrue);
      expect(litAt(f.bmp, 150, 55), isTrue, reason: 'the text is lit');
      expect(litAt(f.bmp, 10, 10), isFalse, reason: 'the white page is dark');
      expect(f.litShare, lessThan(0.05));
    });

    test('photo style dithers mid-gray to about half the pixels', () {
      final f = MonoEncoder.encode(grayOf((x, y) => 128), style: 'photo', invert: 'off');
      expect(f.litShare, closeTo(0.5, 0.05));
    });

    test('letterbox stays dark', () {
      final f = MonoEncoder.encode(grayOf((x, y) => 255), x0: 200, x1: 376, invert: 'off', style: 'photo');
      expect(litAt(f.bmp, 10, 60), isFalse);
      expect(litAt(f.bmp, 288, 60), isTrue);
    });

    test('encoding a frame is fast enough for several frames a second', () {
      final g = grayOf((x, y) => (x * 7 + y * 13) & 0xff);
      final sw = Stopwatch()..start();
      for (var i = 0; i < 10; i++) {
        MonoEncoder.encode(g, style: i.isEven ? 'text' : 'photo');
      }
      // Debug-mode test VM; on the phone (release build) this is ~10x faster.
      expect(sw.elapsedMilliseconds / 10, lessThan(150));
    });
  });

  group('screen mirror', () {
    late FakeGlasses glasses;
    late FakeControl control;
    late BleManager ble;
    late FakeScreen screen;
    late ScreenMirror mirror;

    setUp(() async {
      glasses = FakeGlasses();
      control = FakeControl();
      ble = BleManager(glasses, control: control);
      screen = FakeScreen();
      mirror = ScreenMirror(screen, Proto(ble))
        ..endTimeoutMs = 200
        ..crcTimeoutMs = 300
        ..coolDown = Duration.zero;
      await control.connectNow();
    });

    tearDown(() {
      mirror.dispose();
      ble.dispose();
    });

    Future<void> settle() async {
      for (var i = 0; i < 30; i++) {
        await Future<void>.delayed(Duration.zero);
      }
    }

    int framesOnGlasses() => glasses.sent['L']!.where((p) => p[0] == 0x16).length;

    test('start asks Android with the chosen region and speed', () async {
      mirror
        ..region = 'top'
        ..fpsTenths = 20;
      await mirror.start();
      expect(mirror.running, isTrue);
      expect(screen.configs.single,
          {'region': 'top', 'strip': 0, 'intervalMs': 500, 'cx': 0, 'cy': 0, 'cw': 100, 'ch': 30, 'fit': 'fit', 'rows': 136});
    });

    test('a frame reaches both lenses; an unchanged frame is skipped', () async {
      await mirror.start();
      screen.frame((x, y) => x < 288);
      await settle();
      expect(framesOnGlasses(), 1);
      expect(glasses.sent['R']!.where((p) => p[0] == 0x16).length, 1);
      expect(mirror.preview, isNotNull);
      screen.frame((x, y) => x < 288); // same picture again
      await settle();
      expect(framesOnGlasses(), 1);
      expect(mirror.skipped, 1);
      screen.frame((x, y) => x >= 288);
      await settle();
      expect(framesOnGlasses(), 2);
      expect(mirror.sent, 2);
      await mirror.stop();
    });

    test('while a frame is sending, only the newest waiting frame goes next', () async {
      await mirror.start();
      screen.frame((x, y) => x < 100);
      screen.frame((x, y) => x < 200);
      screen.frame((x, y) => x < 300);
      await settle();
      // The first went out; of the two that queued behind it, only the last.
      expect(mirror.sent, 2);
      await mirror.stop();
    });

    test('a frame the glasses missed is sent again on the next tick', () async {
      await mirror.start();
      glasses.ignoreCmds.add(0x16); // CRC never confirmed
      screen.frame((x, y) => y < 68);
      await Future<void>.delayed(const Duration(milliseconds: 1000));
      await settle();
      expect(mirror.failed, 1);
      glasses.ignoreCmds.clear();
      screen.same();
      await settle();
      expect(mirror.sent, 1);
      await mirror.stop();
    });

    test('stopping clears the glasses; Android ending sharing is noticed', () async {
      await mirror.start();
      await mirror.stop();
      expect(screen.stops, 1);
      expect(glasses.sent['L']!.last, [0x18]);

      await mirror.start();
      screen.stopped('User stopped sharing');
      await settle();
      expect(mirror.running, isFalse);
      expect(mirror.error, 'User stopped sharing');
    });

    test('denied permission is reported, nothing runs', () async {
      screen.answer = 'denied';
      await mirror.start();
      expect(mirror.running, isFalse);
      expect(mirror.error, contains('not allowed'));
    });

    testWidgets('one frame a second keeps up, and sharing turns itself off', (tester) async {
      final g = FakeGlasses();
      final c = FakeControl();
      final b = BleManager(g, control: c);
      final s = FakeScreen();
      final clock = tester.binding.clock;
      final m = ScreenMirror(s, Proto(b))
        ..autoOff = const Duration(seconds: 20)
        ..now = (() => clock.now());
      await c.connectNow();
      await m.start();
      for (var i = 0; i < 10; i++) {
        s.frame((x, y) => x < 50 * (i + 1));
        await tester.pump(const Duration(seconds: 1));
      }
      expect(m.sent, 10, reason: 'every 1 s frame got through');
      expect(m.fps, closeTo(1.0, 0.15));
      await tester.pump(const Duration(seconds: 15));
      expect(m.running, isFalse);
      expect(m.error, contains('turned itself off'));
      m.dispose();
      b.dispose();
    });
  });
}
