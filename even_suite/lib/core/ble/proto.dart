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

  /// Picture bytes per packet: the demo's 194, or up to 236 (fits the 247
  /// byte MTU with the header) for ~20% fewer packets if the glasses accept it.
  int bmpChunk = bmpPackLen;

  /// Same packets as the demo's BmpUpdateManager (with [chunk] = 194).
  static List<Uint8List> bmpPackets(Uint8List bmp, {int chunk = bmpPackLen}) {
    final packs = <Uint8List>[];
    for (int i = 0, index = 0; i < bmp.length; i += chunk, index++) {
      final end = i + chunk < bmp.length ? i + chunk : bmp.length;
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

  // ------------------------------------------------- picture pacing (tuned)

  /// Gap between picture packets per arm (ms). The logs show the glasses
  /// reject (CRC 0xCA) the first send of a NEW picture and accept the same
  /// picture right after, at any packet speed - so a rejection is answered
  /// by re-checking, then resending, not by slowing down. The pace only
  /// rises when even a resend is rejected, and comes back quickly.
  int get bmpGapMs => _baseGap;
  set bmpGapMs(int v) {
    _baseGap = v;
    armGap['L'] = v;
    armGap['R'] = v;
    _streak.updateAll((_, __) => 0);
    _strikes.updateAll((_, __) => 0);
    sequentialArms = false;
  }

  int _baseGap = 5;
  final Map<String, int> armGap = {'L': 5, 'R': 5};
  final Map<String, int> _streak = {'L': 0, 'R': 0};
  final Map<String, int> _strikes = {'L': 0, 'R': 0};
  static const int maxGap = 20;

  /// Tune the pace per arm (off = fixed [bmpGapMs]).
  bool adaptivePacing = true;

  /// After a rejection: ask for the CRC again after [crcRecheckMs] (cheap),
  /// then resend the picture to that arm once.
  bool retryRejectedArm = true;
  int crcRecheckMs = 120;

  /// Try the cheap CRC re-check before resending (off = resend straight away).
  bool recheckFirst = true;

  /// Left first, then right (only if an arm keeps rejecting resends).
  bool sequentialArms = false;

  static bool _accepted(String? crc) => crc == 'c9' || crc == 'short';

  Future<String> _crc(String lr, Uint8List crcCmd, int timeoutMs) async {
    final c = await ble.request(crcCmd, lr: lr, timeoutMs: timeoutMs);
    // Reply: 16 <crc echoed> <status>: 0xC9 = match, 0xCA = mismatch.
    return c.isTimeout ? 'timeout' : (c.data.length > 5 ? c.data[5].toRadixString(16) : 'short');
  }

  /// Packets, end, CRC to one arm (the demo's per-arm sequence).
  Future<ArmResult> _transfer(String lr, List<Uint8List> packs, Uint8List crcCmd,
      {required int endTimeoutMs, required int endRetries, required int crcTimeoutMs}) async {
    final sw = Stopwatch()..start();
    final gap = adaptivePacing ? armGap[lr]! : _baseGap;
    final n = await ble.sendBatch(packs, lr, gapMs: gap);
    final packetsMs = sw.elapsedMilliseconds;
    if (n != packs.length) return ArmResult(lr, gap, packetsMs: packetsMs, stage: 'packets');
    final end = Uint8List.fromList([0x20, 0x0d, 0x0e]);
    var tries = 0;
    var endOk = false;
    for (var i = 0; i <= endRetries; i++) {
      tries++;
      final r = await ble.request(end, lr: lr, timeoutMs: endTimeoutMs);
      if (!r.isTimeout) {
        endOk = r.data.length > 1 && r.data[1] == 0xc9;
        break;
      }
    }
    final endMs = sw.elapsedMilliseconds - packetsMs;
    if (!endOk) return ArmResult(lr, gap, packetsMs: packetsMs, endMs: endMs, endTries: tries, stage: 'end');
    final crc = await _crc(lr, crcCmd, crcTimeoutMs);
    return ArmResult(lr, gap,
        packetsMs: packetsMs, endMs: endMs, endTries: tries, crc: crc, ms: sw.elapsedMilliseconds, stage: 'crc');
  }

  /// One arm, with the recovery steps; returns the final result and what it took.
  Future<ArmResult> _sendArm(String lr, List<Uint8List> packs, Uint8List crcCmd,
      {required int endTimeoutMs, required int endRetries, required int crcTimeoutMs}) async {
    final sw = Stopwatch()..start();
    var a = await _transfer(lr, packs, crcCmd,
        endTimeoutMs: endTimeoutMs, endRetries: endRetries, crcTimeoutMs: crcTimeoutMs);
    final steps = <String>[a.crc ?? a.stage];
    if (retryRejectedArm && a.crc != null && a.crc != 'timeout' && !_accepted(a.crc)) {
      // 1. Same data, just ask again a moment later.
      var again = 'skipped';
      if (recheckFirst) {
        await Future<void>.delayed(Duration(milliseconds: crcRecheckMs));
        again = await _crc(lr, crcCmd, crcTimeoutMs);
        steps.add('recheck:$again');
      }
      if (_accepted(again)) {
        a = a.copyWith(crc: again);
      } else {
        // 2. Resend the picture to this arm.
        final redo = await _transfer(lr, packs, crcCmd,
            endTimeoutMs: endTimeoutMs, endRetries: endRetries, crcTimeoutMs: crcTimeoutMs);
        steps.add('resend:${redo.crc ?? redo.stage}');
        a = redo;
      }
    }
    _tune(lr, a.crc, resent: steps.any((x) => x.startsWith('resend')));
    return a.copyWith(ms: sw.elapsedMilliseconds, steps: steps.join(' '));
  }

  void _tune(String lr, String? crc, {required bool resent}) {
    if (!adaptivePacing || crc == null || crc == 'timeout') return;
    final before = armGap[lr]!;
    if (_accepted(crc)) {
      _strikes[lr] = 0;
      _streak[lr] = _streak[lr]! + 1;
      if (_streak[lr]! >= 4 && armGap[lr]! > _baseGap) {
        armGap[lr] = (armGap[lr]! - 2).clamp(_baseGap, maxGap);
        _streak[lr] = 0;
      }
    } else if (resent) {
      // Even the resend was rejected: now slowing down may help.
      _streak[lr] = 0;
      armGap[lr] = (armGap[lr]! + 3).clamp(_baseGap, maxGap);
      if (armGap[lr]! >= 14) _strikes[lr] = _strikes[lr]! + 1;
      if (_strikes[lr]! >= 3 && !sequentialArms) {
        sequentialArms = true;
        EventLog.add('bmp', 'mode', {'sequential': true, 'arm': lr, 'gap': armGap[lr]});
      }
    }
    if (armGap[lr] != before) {
      EventLog.add('bmp', 'pace', {'arm': lr, 'from': before, 'to': armGap[lr], 'crc': crc});
    }
    if (sequentialArms && armGap['L']! <= _baseGap + 3 && armGap['R']! <= _baseGap + 3) {
      sequentialArms = false;
      EventLog.add('bmp', 'mode', {'sequential': false});
    }
  }

  /// Show a 1-bit BMP (576 wide) on both arms - per arm: packets (paced),
  /// the end command (waited for patiently), the CRC; an arm that rejects
  /// the picture is asked to re-check, then gets the picture once more.
  ///
  /// The glasses only display a picture both arms accepted. With
  /// [requireCrc] false, it counts as done once both arms confirmed the end
  /// and answered the CRC, whatever the CRC said (used for single pictures).
  Future<bool> sendBmp(Uint8List bmp,
      {int endTimeoutMs = 3000, int endRetries = 2, int crcTimeoutMs = 2000, bool requireCrc = true}) async {
    final sw = Stopwatch()..start();
    var arms = <ArmResult>[];
    var ok = false;
    var shown = false;
    try {
      final packs = bmpPackets(bmp, chunk: bmpChunk);
      final crcCmd = bmpCrcCommand(bmp);
      Future<ArmResult> go(String lr) => _sendArm(lr, packs, crcCmd,
          endTimeoutMs: endTimeoutMs, endRetries: endRetries, crcTimeoutMs: crcTimeoutMs);
      if (sequentialArms) {
        arms = [await go('L'), await go('R')];
      } else {
        arms = await Future.wait([go('L'), go('R')]);
      }
      shown = arms.every((a) => _accepted(a.crc));
      final delivered = arms.every((a) => a.crc != null && a.crc != 'timeout');
      ok = requireCrc ? shown : delivered;
      return ok;
    } finally {
      ArmResult? arm(String lr) => arms.where((a) => a.lr == lr).firstOrNull;
      final l = arm('L'), r = arm('R');
      EventLog.add('bmp', 'frame', {
        'ok': ok,
        'shown': shown,
        'stage': shown ? 'done' : [l?.stage, r?.stage].join('/'),
        'ms': sw.elapsedMilliseconds,
        'crc': '${l?.crc}/${r?.crc}',
        'stepsL': l?.steps,
        'stepsR': r?.steps,
        'gapL': l?.gap,
        'gapR': r?.gap,
        'msL': l?.ms,
        'msR': r?.ms,
        'packetsMs': [l?.packetsMs ?? 0, r?.packetsMs ?? 0].reduce((a, b) => a > b ? a : b),
        'endMs': [l?.endMs ?? 0, r?.endMs ?? 0].reduce((a, b) => a > b ? a : b),
        'endL': l?.endMs,
        'endR': r?.endMs,
        'mode': sequentialArms ? 'seq' : 'par',
        'gapMs': _baseGap,
        'chunk': bmpChunk,
        'bytes': bmp.length,
      });
    }
  }
}

/// One arm's part of a picture transfer.
class ArmResult {
  final String lr;
  final int gap;
  final int? packetsMs;
  final int? endMs;
  final int endTries;
  final String? crc; // c9, ca, short, timeout, or null when it stopped earlier
  final int? ms;
  final String stage;
  final String steps; // e.g. "ca recheck:c9" or "ca recheck:ca resend:c9"
  const ArmResult(this.lr, this.gap,
      {this.packetsMs, this.endMs, this.endTries = 0, this.crc, this.ms, this.stage = 'packets', this.steps = ''});

  ArmResult copyWith({String? crc, int? ms, String? steps}) => ArmResult(lr, gap,
      packetsMs: packetsMs,
      endMs: endMs,
      endTries: endTries,
      crc: crc ?? this.crc,
      ms: ms ?? this.ms,
      stage: stage,
      steps: steps ?? this.steps);
}

