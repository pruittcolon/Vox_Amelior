import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crclib/catalog.dart';
import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

// ---- Reference copies of the Even Realities demo's packet code ----------

List<Uint8List> demoEvenaiMultiPackListV2(int cmd,
    {int len = 191,
    required Uint8List data,
    required int syncSeq,
    required int newScreen,
    required int pos,
    required int currentPageNum,
    required int maxPageNum}) {
  final send = <Uint8List>[];
  var maxSeq = data.length ~/ len;
  if (data.length % len > 0) maxSeq++;
  for (var seq = 0; seq < maxSeq; seq++) {
    final start = seq * len;
    var end = start + len;
    if (end > data.length) end = data.length;
    final itemData = data.sublist(start, end);
    final byteData = ByteData(2)..setInt16(0, pos, Endian.big);
    send.add(Uint8List.fromList([
      cmd, syncSeq, maxSeq, seq, newScreen, ...byteData.buffer.asUint8List(),
      currentPageNum, maxPageNum, ...itemData,
    ]));
  }
  return send;
}

List<Uint8List> demoBmpPacks(Uint8List image) {
  const packLen = 194;
  final out = <Uint8List>[];
  final multiPacks = <Uint8List>[];
  for (var i = 0; i < image.length; i += packLen) {
    final end = (i + packLen < image.length) ? i + packLen : image.length;
    multiPacks.add(image.sublist(i, end));
  }
  for (var index = 0; index < multiPacks.length; index++) {
    final pack = multiPacks[index];
    out.add(index == 0
        ? Uint8List.fromList([0x15, index & 0xff, 0x00, 0x1c, 0x00, 0x00, ...pack])
        : Uint8List.fromList([0x15, index & 0xff, ...pack]));
  }
  return out;
}

Uint8List demoCrc(Uint8List image) {
  final result = Uint8List.fromList([0x00, 0x1c, 0x00, 0x00, ...image]);
  final val = Crc32Xz().convert(result).toBigInt().toInt();
  return Uint8List.fromList([0x16, val >> 8 * 3 & 0xff, val >> 8 * 2 & 0xff, val >> 8 & 0xff, val & 0xff]);
}

void main() {
  group('packets are byte-for-byte the demo\'s', () {
    test('text (0x4E), one and several packets', () {
      for (final text in ['Hello G1', 'x' * 500, 'ünïcödé ' * 40]) {
        final data = Uint8List.fromList(utf8.encode(text));
        final mine = Proto.evenaiPackets(data,
            syncSeq: 7, newScreen: 0x71, pos: 300, currentPage: 2, maxPage: 5);
        final demo = demoEvenaiMultiPackListV2(0x4E,
            data: data, syncSeq: 7, newScreen: 0x71, pos: 300, currentPageNum: 2, maxPageNum: 5);
        expect(mine, demo);
      }
    });

    test('heartbeat', () {
      expect(Proto.heartbeatPacket(3), [0x25, 6, 0, 3, 0x04, 3]);
      expect(Proto.heartbeatPacket(300), [0x25, 6, 0, 300 % 0xff, 0x04, 300 % 0xff]);
    });

    test('notification packets split at 176 bytes', () {
      final data = List<int>.generate(400, (i) => i & 0xff);
      final packs = Proto.notifyPackets(9, data);
      expect(packs.length, 3);
      expect(packs[0].sublist(0, 4), [0x4B, 9, 3, 0]);
      expect(packs.expand((p) => p.sublist(4)).toList(), data);
    });

    test('whitelist packets (count 180)', () {
      final data = List<int>.generate(350, (i) => i & 0xff);
      final packs = Proto.packList(0x04, data, count: 180);
      expect(packs.length, 2);
      expect(packs[1].sublist(0, 3), [0x04, 2, 1]);
      expect(packs.expand((p) => p.sublist(3)).toList(), data);
    });

    test('images: sample BMPs give the demo\'s packets and CRC', () {
      for (final name in ['image_1.bmp', 'image_2.bmp']) {
        final bmp = File('assets/images/$name').readAsBytesSync();
        expect(Proto.bmpPackets(bmp), demoBmpPacks(bmp));
        expect(Proto.bmpCrcCommand(bmp), demoCrc(bmp));
      }
    });
  });

  group('with simulated glasses', () {
    late FakeGlasses glasses;
    late BleManager ble;
    late Proto proto;

    setUp(() {
      glasses = FakeGlasses();
      ble = BleManager(glasses);
      proto = Proto(ble);
    });

    tearDown(() => ble.dispose());

    test('text goes to the left arm first, then the right', () async {
      final ok = await proto.sendEvenAIData('hi', newScreen: 0x71, currentPage: 1, maxPage: 1);
      expect(ok, isTrue);
      expect(glasses.screens('L'), ['hi']);
      expect(glasses.screens('R'), ['hi']);
    });

    test('a silent right arm makes the send fail (and times out quickly)', () async {
      glasses.deaf.add('R');
      final ok = await proto.sendEvenAIData('hi', newScreen: 0x71, currentPage: 1, maxPage: 1, timeoutMs: 100);
      expect(ok, isFalse);
    });

    test('heartbeat, exit, notify, whitelist', () async {
      expect(await proto.sendHeartBeat(), isTrue);
      expect(await proto.exit(), isTrue);
      expect(await proto.sendNotify({'title': 't'}, 1), isTrue);
      expect(await proto.sendAppWhitelist('{"app":{}}'), isTrue);
    });

    test('image: batch to both arms, then end and CRC', () async {
      final bmp = File('assets/images/image_1.bmp').readAsBytesSync();
      expect(await proto.sendBmp(bmp), isTrue);
      final packs = Proto.bmpPackets(bmp);
      for (final arm in ['L', 'R']) {
        final got = glasses.sent[arm]!;
        expect(got.sublist(0, packs.length), packs);
        expect(got[packs.length], [0x20, 0x0d, 0x0e]);
        expect(got[packs.length + 1], Proto.bmpCrcCommand(bmp));
      }
    });

    test('image: a late end reply is asked for again, not abandoned', () async {
      final bmp = File('assets/images/image_2.bmp').readAsBytesSync();
      glasses.ignoreFirst[0x20] = 2; // first end command to each arm gets no answer
      proto.bmpGapMs = 5;
      expect(await proto.sendBmp(bmp, endTimeoutMs: 100), isTrue);
      final ends = glasses.sent['L']!.where((p) => p[0] == 0x20).length +
          glasses.sent['R']!.where((p) => p[0] == 0x20).length;
      expect(ends, 4, reason: 'one retry per arm');
      expect(glasses.batchGaps, [5, 5], reason: 'packets paced like the demo');
    });

    test('image: a CRC mismatch (0xCA) still counts as shown when the CRC is not required', () async {
      final bmp = File('assets/images/image_2.bmp').readAsBytesSync();
      final mismatch = _MismatchGlasses();
      final b = BleManager(mismatch);
      final p = Proto(b);
      expect(await p.sendBmp(bmp), isFalse, reason: 'strict: mismatch is a failure');
      expect(await p.sendBmp(bmp, requireCrc: false), isTrue, reason: 'the glasses show it anyway');
      b.dispose();
    });

    test('TouchBar packets become events, not replies', () async {
      final events = <GlassesEvent>[];
      final sub = ble.events.listen(events.add);
      glasses.touch(1, 'L');
      glasses.touch(1, 'R');
      glasses.touch(23, 'L');
      await Future<void>.delayed(Duration.zero);
      expect(events.map((e) => '${e.code}${e.lr}'), ['1L', '1R', '23L']);
      expect(events.first.isTapLeft, isTrue);
      await sub.cancel();
    });

    test('a reply only completes the request for the same arm and command', () async {
      glasses.ignoreCmds.add(0x18);
      final pending = ble.request(Uint8List.fromList([0x18]), lr: 'L', timeoutMs: 200);
      // A reply for another command must not complete it.
      await ble.request(Uint8List.fromList([0x25, 6, 0, 1, 4, 1]), lr: 'L');
      final r = await pending;
      expect(r.isTimeout, isTrue);
    });
  });
}

/// Glasses that answer the CRC with 0xCA (mismatch), as seen in the logs.
class _MismatchGlasses extends FakeGlasses {
  @override
  Future<void> send(Uint8List data, String? lr) async {
    if (data[0] != 0x16) return super.send(data, lr);
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      sent[arm]!.add(data);
      Future.microtask(() => injectReply(arm, Uint8List.fromList([0x16, data[1], data[2], data[3], data[4], 0xCA])));
    }
  }
}
