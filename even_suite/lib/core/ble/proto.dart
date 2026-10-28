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

  /// Send text screens to both arms at once instead of left, then right
  /// (the right arm takes ~200 ms to confirm a screen; the left ~50 ms).
  bool parallelText = false;

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
    if (parallelText) {
      final both = await Future.wait([
        ble.requestList(packets, lr: 'L', timeoutMs: timeoutMs ?? 2000),
        ble.requestList(packets, lr: 'R', timeoutMs: timeoutMs ?? 2000),
      ]);
      return both[0] && both[1];
    }
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

  // ------------------------------------------------ display position (0x26)

  /// Where the glasses' picture sits in your view, the "display height and
  /// depth" setting of the Even app: [height] 0 (low) .. 8 (high), [depth]
  /// 1 (near) .. 9 (far). `26 08 00 <seq> 02 <state> <height> <depth>`
  /// (0x26 = hardware settings, sub-command 0x02 = display); the glasses
  /// answer `26 06 00 <seq> 02 c9`.
  ///
  /// From the decompiled firmware: [preview] true (state 1) stores the height
  /// at once and opens the glasses' height preview - the picture moves, and
  /// text drawn afterwards uses the new height (y = (8 - height) x 35 px on
  /// the panel). State 0 only closes the preview; it moves nothing. The Even
  /// app sends 1 while you drag its slider and 0 when you are done.
  static Uint8List positionPacket(int seq, {required int height, required int depth, bool preview = false}) =>
      Uint8List.fromList([0x26, 0x08, 0x00, seq & 0xff, 0x02, preview ? 1 : 0, height.clamp(0, 8), depth.clamp(0, 9)]);

  // --------------------------------------------- dashboard (0x06, 0x1E)
  //
  // What you see when you look up. Formats from the decompiled firmware
  // (ble_process_put_req.c, cases 0x06 and 0x1E) and the community G1 apps.

  int _dashSeq = 0;

  /// Clock and weather: `06 15 00 <seq> 01 <seconds u32> <ms u64> <icon>
  /// <temp °C> <°F?> <24h?>`. The glasses have no time zone, so the time is
  /// local time written as if it were UTC.
  static Uint8List timeWeatherPacket(int seq, DateTime now,
      {int icon = 0, int tempC = 0, bool fahrenheit = false, bool h24 = false}) {
    final localMs = now.millisecondsSinceEpoch + now.timeZoneOffset.inMilliseconds;
    final b = ByteData(21)
      ..setUint8(0, 0x06)
      ..setUint8(1, 21)
      ..setUint8(2, 0)
      ..setUint8(3, seq & 0xff)
      ..setUint8(4, 0x01)
      ..setUint32(5, localMs ~/ 1000, Endian.little)
      ..setUint32(9, localMs & 0xffffffff, Endian.little)
      ..setUint32(13, (localMs ~/ 0x100000000) & 0xffffffff, Endian.little)
      ..setUint8(17, icon & 0xff)
      ..setInt8(18, tempC.clamp(-128, 127))
      ..setUint8(19, fahrenheit ? 1 : 0)
      ..setUint8(20, h24 ? 1 : 0);
    return b.buffer.asUint8List();
  }

  /// Dashboard layout: [mode] 0 full, 1 dual, 2 minimal; [pane] (the side
  /// panel) 0 notes, 1 stocks, 2 news, 3 calendar, 4 map.
  static Uint8List dashboardModePacket(int seq, {required int mode, required int pane}) =>
      Uint8List.fromList([0x06, 0x07, 0x00, seq & 0xff, 0x06, mode & 0xff, pane & 0xff]);

  /// Quick note [slot] 1..4 on the dashboard's notes panel; an empty [title]
  /// and [text] deletes it. `1E <len u16> <seq> 03 01 00 01 00 <slot> <1=set>
  /// <title len> <title> <text len u16> <text>` (title up to 40 bytes, text 280).
  static Uint8List notePacket(int seq, int slot, String title, String text) {
    List<int> cut(String s, int max) {
      var bytes = utf8.encode(s);
      while (bytes.length > max) {
        s = s.substring(0, s.length - 1);
        bytes = utf8.encode(s);
      }
      return bytes;
    }

    final t = cut(title, 40);
    final x = cut(text, 280);
    final delete = t.isEmpty && x.isEmpty;
    final body = <int>[
      0x03, 0x01, 0x00, 0x01, 0x00, slot.clamp(1, 4), delete ? 0x00 : 0x01,
      if (!delete) ...[t.length, ...t, x.length & 0xff, (x.length >> 8) & 0xff, ...x],
    ];
    final len = 4 + body.length;
    return Uint8List.fromList([0x1E, len & 0xff, (len >> 8) & 0xff, seq & 0xff, ...body]);
  }

  /// Sequence numbers of our last dashboard packets: the glasses echo them
  /// in their answers, so an answer with another number was for someone
  /// else (another app on the phone writing the dashboard).
  final _ourDash = <int>[];

  bool isOurDashSeq(int seq) => _ourDash.contains(seq & 0xff);

  Future<bool> _toBoth(Uint8List packet, String what) async {
    _ourDash.add(packet[3]);
    if (_ourDash.length > 8) _ourDash.removeAt(0);
    var ok = true;
    for (final lr in const ['L', 'R']) {
      final r = await ble.request(packet, lr: lr, timeoutMs: 1500);
      if (r.isTimeout) ok = false;
    }
    EventLog.add('glasses', what, {'ok': ok, 'bytes': packet.length});
    return ok;
  }

  Future<bool> sendTimeWeather(DateTime now, {int icon = 0, int tempC = 0, bool fahrenheit = false, bool h24 = false}) =>
      _toBoth(timeWeatherPacket(_dashSeq++, now, icon: icon, tempC: tempC, fahrenheit: fahrenheit, h24: h24),
          'dashTime');

  Future<bool> sendDashboardMode(int mode, int pane) =>
      _toBoth(dashboardModePacket(_dashSeq++, mode: mode, pane: pane), 'dashMode');

  Future<bool> sendNote(int slot, String title, String text) =>
      _toBoth(notePacket(_dashSeq++, slot, title, text), 'dashNote');

  // ------------------------------------------------------- microphone (0x0E)

  /// Ask each arm for its battery. The answer, `2C 66 <percent> ...`, is read
  /// by [BleManager] (into batteryL / batteryR) like any other.
  Future<void> queryBattery() async {
    for (final lr in const ['L', 'R']) {
      await ble.request(Uint8List.fromList([0x2C, 0x01]), lr: lr, timeoutMs: 1500, log: false);
    }
  }

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
  /// Called whenever the app closes the glasses' screen (they answer that
  /// with the same F5 00 a double tap sends).
  void Function()? onExit;

  Future<bool> exit() async {
    onExit?.call();
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

  // ------------------------------------------------------- picture sending

  /// Gap between picture packets per arm (ms). Fixed: the logs showed the
  /// packet speed makes no difference to whether the glasses accept a
  /// picture, and slowing down only made everything slower.
  int bmpGapMs = 5;

  /// Recovery when an arm rejects a picture (CRC 0xCA): resend it; if that
  /// is rejected too, wait [lateCheckMs] and ask again; then a last resend.
  bool retryRejectedArm = true;
  int lateCheckMs = 300;

  /// Pause + extra CRC query before a resend: the sequence that got 95% of
  /// resends accepted in the user's 21:05 session (an immediate resend did
  /// much worse the next day).
  int beforeResendMs = 150;

  static bool _accepted(String? crc) => crc == 'c9' || crc == 'short';

  Future<String> _crc(String lr, Uint8List crcCmd, int timeoutMs) async {
    final c = await ble.request(crcCmd, lr: lr, timeoutMs: timeoutMs);
    // Reply: 16 <crc echoed> <status>: 0xC9 = match, 0xCA = mismatch.
    return c.isTimeout ? 'timeout' : (c.data.length > 5 ? c.data[5].toRadixString(16) : 'short');
  }

  /// Packets and the end command to one arm (the demo's sequence, minus CRC).
  Future<ArmResult> _transfer(String lr, List<Uint8List> packs,
      {required int endTimeoutMs, required int endRetries}) async {
    final sw = Stopwatch()..start();
    final n = await ble.sendBatch(packs, lr, gapMs: bmpGapMs);
    final packetsMs = sw.elapsedMilliseconds;
    if (n != packs.length) return ArmResult(lr, bmpGapMs, packetsMs: packetsMs, stage: 'packets');
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
    return ArmResult(lr, bmpGapMs,
        packetsMs: packetsMs,
        endMs: sw.elapsedMilliseconds - packetsMs,
        endTries: tries,
        stage: endOk ? 'sent' : 'end');
  }

  /// One arm: transfer [packs], then ask the glasses to verify [verifyCmd]
  /// (normally this picture's CRC; in fast mode the PREVIOUS picture's),
  /// with the recovery steps if they say no.
  Future<ArmResult> _sendArm(String lr, List<Uint8List> packs, Uint8List crcCmd, Uint8List? verifyCmd,
      {required int endTimeoutMs, required int endRetries, required int crcTimeoutMs}) async {
    final sw = Stopwatch()..start();
    Future<ArmResult> transfer() => _transfer(lr, packs, endTimeoutMs: endTimeoutMs, endRetries: endRetries);
    final steps = <String>[];
    var a = await transfer();
    if (a.stage != 'sent') {
      return a.copyWith(ms: sw.elapsedMilliseconds, steps: 'failed:${a.stage}');
    }
    var crc = 'none';
    var viaPrevious = false;
    if (verifyCmd != null) {
      crc = await _crc(lr, verifyCmd, crcTimeoutMs);
      steps.add('prev:$crc');
      viaPrevious = _accepted(crc);
    }
    if (!viaPrevious) {
      crc = await _crc(lr, crcCmd, crcTimeoutMs);
      steps.add(crc);
      if (retryRejectedArm && crc != 'timeout' && !_accepted(crc)) {
        // 1. Short pause and ask again, then resend (what worked best so far).
        await Future<void>.delayed(Duration(milliseconds: beforeResendMs));
        final again = await _crc(lr, crcCmd, crcTimeoutMs);
        steps.add('pause:$again');
        if (_accepted(again)) {
          return a.copyWith(crc: again, ms: sw.elapsedMilliseconds, steps: steps.join(' '));
        }
        final redo = await transfer();
        crc = redo.stage == 'sent' ? await _crc(lr, crcCmd, crcTimeoutMs) : 'failed:${redo.stage}';
        steps.add('resend:$crc');
        if (!_accepted(crc) && crc != 'timeout') {
          // 2. Maybe it is still storing: give it a moment and ask again.
          await Future<void>.delayed(Duration(milliseconds: lateCheckMs));
          crc = await _crc(lr, crcCmd, crcTimeoutMs);
          steps.add('late:$crc');
        }
        if (!_accepted(crc) && crc != 'timeout') {
          // 3. Last resend.
          final third = await transfer();
          crc = third.stage == 'sent' ? await _crc(lr, crcCmd, crcTimeoutMs) : 'failed:${third.stage}';
          steps.add('resend2:$crc');
        }
      }
    }
    return a.copyWith(crc: crc, ms: sw.elapsedMilliseconds, steps: steps.join(' '), viaPrevious: viaPrevious);
  }

  /// Show a 1-bit BMP (576 wide) on both arms (in parallel).
  ///
  /// With [previous] (fast mode), the glasses are asked to verify the
  /// previous picture after this one arrives - they seem to check a picture
  /// against the one sent before it, so this needs one transfer per frame
  /// instead of two; the display then runs one frame behind. If they do not
  /// accept that, the normal check and recovery steps follow.
  Future<BmpOutcome> sendFrame(Uint8List bmp,
      {Uint8List? previous, int endTimeoutMs = 3000, int endRetries = 2, int crcTimeoutMs = 2000}) async {
    final sw = Stopwatch()..start();
    var arms = <ArmResult>[];
    var outcome = const BmpOutcome(shown: false, delivered: false, viaPrevious: false);
    try {
      final packs = bmpPackets(bmp, chunk: bmpChunk);
      final crcCmd = bmpCrcCommand(bmp);
      final verifyCmd = previous == null ? null : bmpCrcCommand(previous);
      arms = await Future.wait([
        for (final lr in const ['L', 'R'])
          _sendArm(lr, packs, crcCmd, verifyCmd,
              endTimeoutMs: endTimeoutMs, endRetries: endRetries, crcTimeoutMs: crcTimeoutMs),
      ]);
      outcome = BmpOutcome(
        shown: arms.every((a) => _accepted(a.crc)),
        delivered: arms.every((a) => a.crc != null && a.crc != 'timeout' && !a.crc!.startsWith('failed')),
        viaPrevious: arms.every((a) => a.viaPrevious),
      );
      return outcome;
    } finally {
      ArmResult? arm(String lr) => arms.where((a) => a.lr == lr).firstOrNull;
      final l = arm('L'), r = arm('R');
      EventLog.add('bmp', 'frame', {
        'ok': outcome.shown,
        'shown': outcome.shown,
        'fast': previous != null,
        'viaPrevious': outcome.viaPrevious,
        'stage': outcome.shown ? 'done' : [l?.stage, r?.stage].join('/'),
        'ms': sw.elapsedMilliseconds,
        'crc': '${l?.crc}/${r?.crc}',
        'stepsL': l?.steps,
        'stepsR': r?.steps,
        'msL': l?.ms,
        'msR': r?.ms,
        'packetsMs': [l?.packetsMs ?? 0, r?.packetsMs ?? 0].reduce((a, b) => a > b ? a : b),
        'endMs': [l?.endMs ?? 0, r?.endMs ?? 0].reduce((a, b) => a > b ? a : b),
        'gapMs': bmpGapMs,
        'chunk': bmpChunk,
        'bytes': bmp.length,
        'crcHex': bmpCrcCommand(bmp).sublist(1).map((b) => b.toRadixString(16).padLeft(2, '0')).join(),
        if (previous != null)
          'prevHex': bmpCrcCommand(previous).sublist(1).map((b) => b.toRadixString(16).padLeft(2, '0')).join(),
      });
    }
  }

  /// One picture (images page etc.): shown when both arms accepted it; with
  /// [requireCrc] false, done once both arms answered.
  Future<bool> sendBmp(Uint8List bmp,
      {int endTimeoutMs = 3000, int endRetries = 2, int crcTimeoutMs = 2000, bool requireCrc = true}) async {
    final o = await sendFrame(bmp, endTimeoutMs: endTimeoutMs, endRetries: endRetries, crcTimeoutMs: crcTimeoutMs);
    return requireCrc ? o.shown : o.delivered;
  }
}

/// What happened to a picture.
class BmpOutcome {
  /// Both arms accepted (this picture, or in fast mode the previous one).
  final bool shown;

  /// Both arms answered the CRC (whatever it said).
  final bool delivered;

  /// Fast mode worked: the previous picture was verified with one transfer.
  final bool viaPrevious;

  const BmpOutcome({required this.shown, required this.delivered, required this.viaPrevious});
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
  final String steps; // e.g. "ca resend:c9" or "prev:c9"
  final bool viaPrevious;
  const ArmResult(this.lr, this.gap,
      {this.packetsMs,
      this.endMs,
      this.endTries = 0,
      this.crc,
      this.ms,
      this.stage = 'packets',
      this.steps = '',
      this.viaPrevious = false});

  ArmResult copyWith({String? crc, int? ms, String? steps, bool? viaPrevious}) => ArmResult(lr, gap,
      packetsMs: packetsMs,
      endMs: endMs,
      endTries: endTries,
      crc: crc ?? this.crc,
      ms: ms ?? this.ms,
      stage: stage,
      steps: steps ?? this.steps,
      viaPrevious: viaPrevious ?? this.viaPrevious);
}

