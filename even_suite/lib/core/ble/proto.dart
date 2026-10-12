import 'dart:convert';
import 'dart:typed_data';

import 'package:crclib/catalog.dart';

import '../log.dart';
import 'ble_manager.dart';

/// G1 commands from the Even Realities demo. Packet builders are static and
/// byte-for-byte the demo's, so they are unit tested on their own.
class Proto {
  Proto(this.ble);

  final BleManager ble;

  // ------------------------------------------------------------ text (0x4E)

  /// Screen status for 0x4E, upper nibble. Lower nibble 0x01 = new content.
  static const int aiShowing = 0x30; // Even AI, automatic paging
  static const int aiComplete = 0x40; // Even AI, last page
  static const int aiManual = 0x50; // Even AI, manual paging
  static const int aiNetworkError = 0x60;
  static const int textShow = 0x70; // plain text page

  static int newScreen(int status, [int type = 0x01]) => status | type;

  int _evenaiSeq = 0;

  /// Send one screen of text to both arms (left first, then right).
  Future<bool> sendEvenAIData(String text,
      {required int newScreen,
      int pos = 0,
      required int currentPage,
      required int maxPage,
      int? timeoutMs}) async {
    final packets = evenaiPackets(utf8.encode(text),
        syncSeq: _evenaiSeq & 0xff,
        newScreen: newScreen,
        pos: pos,
        currentPage: currentPage,
        maxPage: maxPage);
    _evenaiSeq++;
    if (!await ble.requestList(packets, lr: 'L', timeoutMs: timeoutMs ?? 2000)) {
      return false;
    }
    return ble.requestList(packets, lr: 'R', timeoutMs: timeoutMs ?? 2000);
  }

  static List<Uint8List> evenaiPackets(List<int> data,
      {int cmd = 0x4E,
      int len = 191,
      required int syncSeq,
      required int newScreen,
      required int pos,
      required int currentPage,
      required int maxPage}) {
    final send = <Uint8List>[];
    var maxSeq = data.length ~/ len;
    if (data.length % len > 0) maxSeq++;
    for (var seq = 0; seq < maxSeq; seq++) {
      final start = seq * len;
      final end = start + len > data.length ? data.length : start + len;
      send.add(Uint8List.fromList([
        cmd,
        syncSeq,
        maxSeq,
        seq,
        newScreen,
        (pos >> 8) & 0xff,
        pos & 0xff,
        currentPage,
        maxPage,
        ...data.sublist(start, end),
      ]));
    }
    return send;
  }

  // ------------------------------------------------------- microphone (0x0E)

  /// Glasses microphone on/off (the demo turns it on at the right arm when the
  /// left TouchBar is held). Audio then streams as 0xF1 packets.
  Future<bool> micOn(bool on) async {
    final r = await ble.request(Uint8List.fromList([0x0E, on ? 0x01 : 0x00]), lr: 'R', timeoutMs: 1000);
    final ok = !r.isTimeout && r.data.length > 1 && r.data[1] == 0xc9;
    EventLog.add('glasses', 'mic', {'on': on, 'ok': ok});
    return ok;
  }

  // -------------------------------------------------------- heartbeat (0x25)

  int _beatSeq = 0;

  static Uint8List heartbeatPacket(int seq) =>
      Uint8List.fromList([0x25, 6, 0, seq % 0xff, 0x04, seq % 0xff]);

  Future<bool> sendHeartBeat() async {
    final data = heartbeatPacket(_beatSeq++);
    bool ok(Uint8List d) => d.length > 5 && d[0] == 0x25 && d[4] == 0x04;
    final retL = await ble.request(data, lr: 'L', timeoutMs: 1500);
    if (retL.isTimeout || !ok(retL.data)) return false;
    final retR = await ble.request(data, lr: 'R', timeoutMs: 1500);
    return !retR.isTimeout && ok(retR.data);
  }

  // -------------------------------------------------------------- exit (0x18)

  /// Back to the dashboard (clears whatever the app was showing).
  Future<bool> exit() async {
    final data = Uint8List.fromList([0x18]);
    bool ok(Uint8List d) => d.length > 1 && d[1] == 0xc9;
    final retL = await ble.request(data, lr: 'L', timeoutMs: 1500);
    if (retL.isTimeout || !ok(retL.data)) return false;
    final retR = await ble.request(data, lr: 'R', timeoutMs: 1500);
    return !retR.isTimeout && ok(retR.data);
  }

  // ----------------------------------------------------- notifications (0x04, 0x4B)

  static List<Uint8List> packList(int cmd, List<int> data, {int count = 20}) {
    final realCount = count - 3;
    final send = <Uint8List>[];
    var maxSeq = data.length ~/ realCount;
    if (data.length % realCount > 0) maxSeq++;
    for (var seq = 0; seq < maxSeq; seq++) {
      final start = seq * realCount;
      final end = start + realCount > data.length ? data.length : start + realCount;
      send.add(Uint8List.fromList([cmd, maxSeq, seq, ...data.sublist(start, end)]));
    }
    return send;
  }

  static List<Uint8List> notifyPackets(int msgId, List<int> data, {int cmd = 0x4B}) {
    final send = <Uint8List>[];
    var maxSeq = data.length ~/ 176;
    if (data.length % 176 > 0) maxSeq++;
    for (var seq = 0; seq < maxSeq; seq++) {
      final start = seq * 176;
      final end = start + 176 > data.length ? data.length : start + 176;
      send.add(Uint8List.fromList([cmd, msgId, maxSeq, seq, ...data.sublist(start, end)]));
    }
    return send;
  }

  /// Which phone apps may show notifications on the glasses.
  Future<bool> sendAppWhitelist(String whitelistJson) async {
    final packets = packList(0x04, utf8.encode(whitelistJson), count: 180);
    for (var i = 0; i < 3; i++) {
      if (await ble.requestList(packets, timeoutMs: 300, lr: 'L')) return true;
    }
    return false;
  }

  Future<bool> sendNotify(Map appData, int notifyId, {int retry = 6}) async {
    final json = jsonEncode({'ncs_notification': appData});
    final packets = notifyPackets(notifyId, utf8.encode(json));
    for (var i = 0; i < retry; i++) {
      if (await ble.requestList(packets, timeoutMs: 1000, lr: 'L')) return true;
    }
    return false;
  }

  // ------------------------------------------------------- images (0x15, 0x20, 0x16)

  static const int bmpPackLen = 194;
  static const List<int> bmpAddress = [0x00, 0x1c, 0x00, 0x00];

  /// Same packets as the demo's BmpUpdateManager.
  static List<Uint8List> bmpPackets(Uint8List bmp) {
    final packs = <Uint8List>[];
    for (int i = 0, index = 0; i < bmp.length; i += bmpPackLen, index++) {
      final end = i + bmpPackLen < bmp.length ? i + bmpPackLen : bmp.length;
      final head = index == 0 ? [0x15, index & 0xff, ...bmpAddress] : [0x15, index & 0xff];
      packs.add(Uint8List.fromList([...head, ...bmp.sublist(i, end)]));
    }
    return packs;
  }

  static Uint8List bmpCrcCommand(Uint8List bmp) {
    final val = Crc32Xz().convert([...bmpAddress, ...bmp]).toBigInt().toInt();
    return Uint8List.fromList(
        [0x16, val >> 24 & 0xff, val >> 16 & 0xff, val >> 8 & 0xff, val & 0xff]);
  }

  /// Show a 576x136 1-bit BMP on both arms. All packets for an arm go down
  /// in one flow-controlled batch, both arms in parallel, then end + CRC.
  Future<bool> sendBmp(Uint8List bmp, {int timeoutMs = 3000}) async {
    final sw = Stopwatch()..start();
    var stage = 'packets';
    var ok = false;
    int? packetsMs, endMs;
    try {
      final packs = bmpPackets(bmp);
      final sent = await Future.wait([ble.sendBatch(packs, 'L'), ble.sendBatch(packs, 'R')]);
      packetsMs = sw.elapsedMilliseconds;
      if (sent[0] != packs.length || sent[1] != packs.length) return false;

      stage = 'end';
      final end = Uint8List.fromList([0x20, 0x0d, 0x0e]);
      final ends = await Future.wait([
        ble.request(end, lr: 'L', timeoutMs: timeoutMs),
        ble.request(end, lr: 'R', timeoutMs: timeoutMs),
      ]);
      endMs = sw.elapsedMilliseconds - packetsMs;
      if (ends.any((r) => r.isTimeout || r.data.length < 2 || r.data[1] != 0xc9)) {
        return false;
      }
      stage = 'crc';
      final crc = bmpCrcCommand(bmp);
      final crcs = await Future.wait([
        ble.request(crc, lr: 'L', timeoutMs: timeoutMs),
        ble.request(crc, lr: 'R', timeoutMs: timeoutMs),
      ]);
      // Same check as the demo: byte 5 of the reply is the status.
      ok = crcs.every((r) => !r.isTimeout && r.data.length > 5 && r.data[5] == 0xc9);
      if (ok) stage = 'done';
      return ok;
    } finally {
      EventLog.add('bmp', 'frame', {
        'ok': ok,
        'stage': stage,
        'ms': sw.elapsedMilliseconds,
        'packetsMs': packetsMs,
        'endMs': endMs,
        'bytes': bmp.length,
      });
    }
  }
}
