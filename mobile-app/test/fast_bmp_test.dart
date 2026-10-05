import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:crclib/catalog.dart';
import 'package:demo_ai_even/ble_manager.dart';
import 'package:demo_ai_even/controllers/bmp_update_manager.dart';
import 'package:demo_ai_even/services/fast_bmp.dart';
import 'package:demo_ai_even/views/features/speed_test_page.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

/// Simulated G1: two arms behind the `method.bluetooth` channel, replying on
/// `eventBleReceive` the way the demo app expects (0x20 → [0x20, 0xC9],
/// 0x16 → [0x16, crc×4, 0xC9/0xCA]). Each arm rebuilds the BMP from the 0x15
/// packets and only "shows" it when the CRC matches.
class FakeGlasses {
  MockStreamHandlerEventSink? sink;
  final log = <String, List<Uint8List>>{'L': [], 'R': []};
  final shown = <String, List<Uint8List>>{'L': [], 'R': []};
  final _buf = <String, List<int>>{'L': [], 'R': []};
  final _ok = <String, bool>{'L': true, 'R': true};
  final _expect = <String, int>{'L': 0, 'R': 0};
  int platformCalls = 0;
  int dropNextLeftPacket = 0;

  void reset() {
    for (final lr in ['L', 'R']) {
      log[lr]!.clear();
      shown[lr]!.clear();
      _buf[lr]!.clear();
      _ok[lr] = true;
      _expect[lr] = 0;
    }
    platformCalls = 0;
    dropNextLeftPacket = 0;
  }

  Future<Object?> handle(MethodCall call) async {
    platformCalls++;
    final args = call.arguments as Map;
    final lr = args['lr'] as String;
    switch (call.method) {
      case 'send':
        _packet(lr, args['data'] as Uint8List);
        return null;
      case 'sendBatch':
        final packets = (args['packets'] as List).cast<Uint8List>();
        packets.forEach((p) => _packet(lr, p));
        return packets.length;
    }
    return null;
  }

  void _packet(String lr, Uint8List p) {
    log[lr]!.add(p);
    switch (p[0]) {
      case 0x15:
        if (lr == 'L' && dropNextLeftPacket > 0) {
          dropNextLeftPacket--;
          return; // lost on the air
        }
        final seq = p[1];
        if (seq == 0) {
          _buf[lr]!.clear();
          _ok[lr] = listEquals(p.sublist(2, 6), FastBmp.address);
          _buf[lr]!.addAll(p.sublist(6));
        } else {
          if (seq != _expect[lr]) _ok[lr] = false;
          _buf[lr]!.addAll(p.sublist(2));
        }
        _expect[lr] = (seq + 1) & 0xff;
      case 0x20:
        _reply(lr, [0x20, 0xc9]);
      case 0x16:
        final val = Crc32Xz()
            .convert([...FastBmp.address, ..._buf[lr]!])
            .toBigInt()
            .toInt();
        final good = _ok[lr]! &&
            p[1] == (val >> 24 & 0xff) &&
            p[2] == (val >> 16 & 0xff) &&
            p[3] == (val >> 8 & 0xff) &&
            p[4] == (val & 0xff);
        if (good) shown[lr]!.add(Uint8List.fromList(_buf[lr]!));
        _reply(lr, [0x16, p[1], p[2], p[3], p[4], good ? 0xc9 : 0xca]);
      case 0x18:
        _reply(lr, [0x18, 0xc9]);
    }
  }

  void _reply(String lr, List<int> data) {
    scheduleMicrotask(() => sink!.success(
        {'lr': lr, 'data': Uint8List.fromList(data), 'type': 'Receive'}));
  }
}

Uint8List testFrame(int n) {
  final c = G1Canvas();
  c.text(4, 4, 'FRAME $n', scale: 3);
  c.rect(4 + n * 10, 100, 40 + n * 10, 130);
  return c.toBmp();
}

Future<void> waitIdle(FrameStreamer s) async {
  while (s.isSending) {
    await Future.delayed(const Duration(milliseconds: 5));
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final glasses = FakeGlasses();

  setUpAll(() {
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(
        const MethodChannel('method.bluetooth'), glasses.handle);
    messenger.setMockStreamHandler(
        const EventChannel('eventBleReceive'),
        MockStreamHandler.inline(
            onListen: (args, sink) => glasses.sink = sink));
    BleManager.get().startListening();
  });

  setUp(glasses.reset);

  group('G1Canvas BMP', () {
    test('has the exact size and header of the demo images', () {
      final bmp = G1Canvas().toBmp();
      final bd = ByteData.view(bmp.buffer);
      expect(bmp.length, 9854);
      expect(String.fromCharCodes(bmp.sublist(0, 2)), 'BM');
      expect(bd.getUint32(10, Endian.little), 62);
      expect(bd.getInt32(18, Endian.little), 576);
      expect(bd.getInt32(22, Endian.little), 136);
      expect(bd.getUint16(28, Endian.little), 1);
      final sample = File('assets/images/image_2.bmp').readAsBytesSync();
      expect(bmp.sublist(54, 62), sample.sublist(54, 62),
          reason: 'same palette as the demo images');
    });

    test('blank is all background (0xFF), lit top-left pixel clears one bit', () {
      final c = G1Canvas();
      expect(c.toBmp().sublist(62).every((b) => b == 0xff), isTrue);
      c.pixel(0, 0);
      final bmp = c.toBmp();
      // Rows are stored bottom-up, so the top row is the last 72 bytes.
      expect(bmp[62 + 135 * 72], 0x7f);
      expect(bmp.sublist(62).where((b) => b != 0xff).length, 1);
    });
  });

  group('FastBmp protocol', () {
    test('packets: 51 of them, address only in the first, payload intact', () {
      final bmp = testFrame(1);
      final packs = FastBmp.packets(bmp);
      expect(packs.length, 51);
      expect(packs[0].sublist(0, 6), [0x15, 0x00, 0x00, 0x1c, 0x00, 0x00]);
      expect(packs[1].sublist(0, 2), [0x15, 0x01]);
      expect(packs[50][1], 50);
      final payload = [
        ...packs[0].sublist(6),
        for (final p in packs.skip(1)) ...p.sublist(2)
      ];
      expect(payload, bmp);
    });

    test('sends byte-for-byte what the original demo sends', () async {
      final bmp = testFrame(2);
      final demo = BmpUpdateManager();
      final demoOk = await Future.wait(
          [demo.updateBmp('L', bmp, seq: 0), demo.updateBmp('R', bmp, seq: 0)]);
      expect(demoOk, [true, true]);
      final demoLog = {
        for (final lr in ['L', 'R']) lr: List.of(glasses.log[lr]!)
      };
      final demoCalls = glasses.platformCalls;

      glasses.reset();
      expect(await FastBmp.sendFrame(bmp), isTrue);
      for (final lr in ['L', 'R']) {
        expect(glasses.log[lr], demoLog[lr], reason: 'arm $lr');
        expect(glasses.shown[lr]!.single, bmp);
      }
      // ignore: avoid_print
      print('platform-channel calls per frame: demo $demoCalls, '
          'fast ${glasses.platformCalls}');
      expect(glasses.platformCalls, lessThan(demoCalls ~/ 10));
    });

    test('frame shows on both arms without waiting for the end reply', () async {
      final bmp = testFrame(3);
      expect(await FastBmp.sendFrame(bmp, waitForEnd: false), isTrue);
      expect(glasses.shown['L']!.single, bmp);
      expect(glasses.shown['R']!.single, bmp);
    });

    test('a lost packet fails the CRC, and the next frame recovers', () async {
      glasses.dropNextLeftPacket = 1;
      expect(await FastBmp.sendFrame(testFrame(4)), isFalse);
      expect(glasses.shown['L'], isEmpty);
      final next = testFrame(5);
      expect(await FastBmp.sendFrame(next), isTrue);
      expect(glasses.shown['L']!.single, next);
    });
  });

  group('FrameStreamer', () {
    test('sends the newest frame and skips frames already shown', () async {
      final s = FrameStreamer();
      final f1 = testFrame(1), f2 = testFrame(2), f3 = testFrame(3);
      s.push(f1);
      s.push(f2); // replaced by f3 before f1 finishes
      s.push(f3);
      await waitIdle(s);
      expect(glasses.shown['L'], [f1, f3]);
      expect(glasses.shown['R'], [f1, f3]);

      s.push(testFrame(3));
      await waitIdle(s);
      expect(s.framesSkipped, 1);
      expect(s.framesOk, 2);
      expect(glasses.shown['L']!.length, 2);
    });

    test('counts a failed frame and keeps going', () async {
      final s = FrameStreamer();
      glasses.dropNextLeftPacket = 1;
      s.push(testFrame(6));
      await waitIdle(s);
      s.push(testFrame(7));
      await waitIdle(s);
      expect(s.framesFailed, 1);
      expect(s.framesOk, 1);
    });
  });

  testWidgets('speed test page builds', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SpeedTestPage()));
    expect(find.text('Start streaming'), findsOneWidget);
    expect(find.text('Frames per second'), findsOneWidget);
  });
}
