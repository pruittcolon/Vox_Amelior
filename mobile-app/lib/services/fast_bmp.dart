import 'dart:async';
import 'dart:typed_data';

import 'package:crclib/catalog.dart';
import 'package:demo_ai_even/ble_manager.dart';

/// Streaming version of [BmpUpdateManager]: same G1 protocol (0x15 packets,
/// 0x20 end, 0x16 CRC) but built to send frame after frame.
///
/// Differences from the demo:
/// - all 51 packets per arm go down in one `sendBatch` call, queued natively
///   with flow control instead of a 5～8ms sleep per packet;
/// - both arms are sent in parallel, then end + CRC to both in parallel;
/// - short timeouts and no retries: a failed frame is simply replaced by the
///   next one.
class FastBmp {
  static const int packLen = 194;
  static const List<int> address = [0x00, 0x1c, 0x00, 0x00];

  static List<Uint8List> packets(Uint8List bmp) {
    final packs = <Uint8List>[];
    for (int i = 0, index = 0; i < bmp.length; i += packLen, index++) {
      final end = i + packLen < bmp.length ? i + packLen : bmp.length;
      final head = index == 0
          ? [0x15, index & 0xff, ...address]
          : [0x15, index & 0xff];
      packs.add(Uint8List.fromList([...head, ...bmp.sublist(i, end)]));
    }
    return packs;
  }

  static Uint8List crcCommand(Uint8List bmp) {
    final val = Crc32Xz().convert([...address, ...bmp]).toBigInt().toInt();
    return Uint8List.fromList([
      0x16,
      val >> 24 & 0xff,
      val >> 16 & 0xff,
      val >> 8 & 0xff,
      val & 0xff,
    ]);
  }

  /// Send one full frame to both arms. When [waitForEnd] is false the CRC is
  /// sent right after the end command without waiting for its reply (saves a
  /// round trip; whether the firmware tolerates it needs testing).
  static Future<bool> sendFrame(Uint8List bmp,
      {bool waitForEnd = true, int timeoutMs = 400}) async {
    final packs = packets(bmp);
    final sent = await Future.wait(
        [BleManager.sendBatch(packs, "L"), BleManager.sendBatch(packs, "R")]);
    if (sent[0] != packs.length || sent[1] != packs.length) return false;

    final end = Uint8List.fromList([0x20, 0x0d, 0x0e]);
    if (waitForEnd) {
      final ends = await Future.wait([
        BleManager.request(end, lr: "L", timeoutMs: timeoutMs),
        BleManager.request(end, lr: "R", timeoutMs: timeoutMs),
      ]);
      if (ends.any((r) => r.isTimeout || r.data.length < 2 || r.data[1] != 0xc9)) {
        return false;
      }
    } else {
      await Future.wait([
        BleManager.sendData(end, lr: "L"),
        BleManager.sendData(end, lr: "R"),
      ]);
    }

    final crc = crcCommand(bmp);
    final crcs = await Future.wait([
      BleManager.request(crc, lr: "L", timeoutMs: timeoutMs),
      BleManager.request(crc, lr: "R", timeoutMs: timeoutMs),
    ]);
    // Same check as the demo: byte 5 of the reply is the status.
    return crcs.every((r) => !r.isTimeout && r.data.length > 5 && r.data[5] == 0xc9);
  }
}

/// Sends frames as fast as the link allows, always the newest one.
/// Frames pushed while one is in flight replace each other, and a frame equal
/// to the last one shown is skipped.
class FrameStreamer {
  bool waitForEnd = true;
  int framesOk = 0;
  int framesFailed = 0;
  int framesSkipped = 0;
  int lastFrameMs = 0;
  final _window = <int>[]; // completion times of recent good frames
  void Function()? onStats;

  Uint8List? _pending;
  Uint8List? _shown;
  bool _sending = false;

  double get fps {
    if (_window.length < 2) return 0;
    return (_window.length - 1) * 1000 / (_window.last - _window.first);
  }

  bool get isSending => _sending;

  void push(Uint8List bmp) {
    _pending = bmp;
    if (!_sending) _loop();
  }

  /// Forget a frame waiting to be sent (e.g. when stopping).
  void dropPending() => _pending = null;

  void reset() {
    framesOk = framesFailed = framesSkipped = lastFrameMs = 0;
    _window.clear();
    _shown = null;
  }

  Future<void> _loop() async {
    _sending = true;
    while (_pending != null) {
      final frame = _pending!;
      _pending = null;
      if (_shown != null && _equal(frame, _shown!)) {
        framesSkipped++;
        continue;
      }
      final t0 = DateTime.now().millisecondsSinceEpoch;
      final ok = await FastBmp.sendFrame(frame, waitForEnd: waitForEnd);
      final t1 = DateTime.now().millisecondsSinceEpoch;
      lastFrameMs = t1 - t0;
      if (ok) {
        framesOk++;
        _shown = frame;
        _window.add(t1);
        if (_window.length > 10) _window.removeAt(0);
      } else {
        framesFailed++;
        _shown = null;
      }
      onStats?.call();
    }
    _sending = false;
  }

  static bool _equal(Uint8List a, Uint8List b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// 576x136 1-bit canvas that encodes to the BMP layout the G1 expects
/// (62-byte header, bottom-up rows, palette 0 = white/lit, 1 = black/off).
class G1Canvas {
  static const int width = 576, height = 136, stride = width ~/ 8;
  final lit = Uint8List(width * height);

  void clear() => lit.fillRange(0, lit.length, 0);

  void pixel(int x, int y) {
    if (x >= 0 && x < width && y >= 0 && y < height) lit[y * width + x] = 1;
  }

  void rect(int x0, int y0, int x1, int y1, {bool fill = true}) {
    for (var y = y0; y < y1; y++) {
      for (var x = x0; x < x1; x++) {
        if (fill || y == y0 || y == y1 - 1 || x == x0 || x == x1 - 1) {
          pixel(x, y);
        }
      }
    }
  }

  /// 5x7 font, columns as bytes with bit 0 at the top.
  static const Map<String, List<int>> _font = {
    'A': [0x7C, 0x12, 0x11, 0x12, 0x7C], 'B': [0x7F, 0x49, 0x49, 0x49, 0x36],
    'C': [0x3E, 0x41, 0x41, 0x41, 0x22], 'D': [0x7F, 0x41, 0x41, 0x22, 0x1C],
    'E': [0x7F, 0x49, 0x49, 0x49, 0x41], 'F': [0x7F, 0x09, 0x09, 0x09, 0x01],
    'G': [0x3E, 0x41, 0x49, 0x49, 0x7A], 'H': [0x7F, 0x08, 0x08, 0x08, 0x7F],
    'I': [0x00, 0x41, 0x7F, 0x41, 0x00], 'J': [0x20, 0x40, 0x41, 0x3F, 0x01],
    'K': [0x7F, 0x08, 0x14, 0x22, 0x41], 'L': [0x7F, 0x40, 0x40, 0x40, 0x40],
    'M': [0x7F, 0x02, 0x0C, 0x02, 0x7F], 'N': [0x7F, 0x04, 0x08, 0x10, 0x7F],
    'O': [0x3E, 0x41, 0x41, 0x41, 0x3E], 'P': [0x7F, 0x09, 0x09, 0x09, 0x06],
    'Q': [0x3E, 0x41, 0x51, 0x21, 0x5E], 'R': [0x7F, 0x09, 0x19, 0x29, 0x46],
    'S': [0x46, 0x49, 0x49, 0x49, 0x31], 'T': [0x01, 0x01, 0x7F, 0x01, 0x01],
    'U': [0x3F, 0x40, 0x40, 0x40, 0x3F], 'V': [0x1F, 0x20, 0x40, 0x20, 0x1F],
    'W': [0x3F, 0x40, 0x38, 0x40, 0x3F], 'X': [0x63, 0x14, 0x08, 0x14, 0x63],
    'Y': [0x07, 0x08, 0x70, 0x08, 0x07], 'Z': [0x61, 0x51, 0x49, 0x45, 0x43],
    '0': [0x3E, 0x51, 0x49, 0x45, 0x3E], '1': [0x00, 0x42, 0x7F, 0x40, 0x00],
    '2': [0x42, 0x61, 0x51, 0x49, 0x46], '3': [0x21, 0x41, 0x45, 0x4B, 0x31],
    '4': [0x18, 0x14, 0x12, 0x7F, 0x10], '5': [0x27, 0x45, 0x45, 0x45, 0x39],
    '6': [0x3C, 0x4A, 0x49, 0x49, 0x30], '7': [0x01, 0x71, 0x09, 0x05, 0x03],
    '8': [0x36, 0x49, 0x49, 0x49, 0x36], '9': [0x06, 0x49, 0x49, 0x29, 0x1E],
    ':': [0x00, 0x36, 0x36, 0x00, 0x00], '%': [0x23, 0x13, 0x08, 0x64, 0x62],
    '>': [0x00, 0x41, 0x22, 0x14, 0x08], '<': [0x08, 0x14, 0x22, 0x41, 0x00],
    '-': [0x08, 0x08, 0x08, 0x08, 0x08], '.': [0x00, 0x60, 0x60, 0x00, 0x00],
    ',': [0x00, 0x50, 0x30, 0x00, 0x00], '!': [0x00, 0x00, 0x5F, 0x00, 0x00],
    '/': [0x20, 0x10, 0x08, 0x04, 0x02], '?': [0x02, 0x01, 0x51, 0x09, 0x06],
    ' ': [0x00, 0x00, 0x00, 0x00, 0x00], '*': [0x14, 0x08, 0x3E, 0x08, 0x14],
    '_': [0x40, 0x40, 0x40, 0x40, 0x40], '#': [0x14, 0x7F, 0x14, 0x7F, 0x14],
    '[': [0x00, 0x7F, 0x41, 0x41, 0x00], ']': [0x00, 0x41, 0x41, 0x7F, 0x00],
    '(': [0x00, 0x1C, 0x22, 0x41, 0x00], ')': [0x00, 0x41, 0x22, 0x1C, 0x00],
    '+': [0x08, 0x08, 0x3E, 0x08, 0x08], '=': [0x14, 0x14, 0x14, 0x14, 0x14],
    "'": [0x00, 0x05, 0x03, 0x00, 0x00], '"': [0x00, 0x07, 0x00, 0x07, 0x00],
  };

  void text(int x, int y, String s, {int scale = 2}) {
    for (final ch in s.toUpperCase().split('')) {
      final cols = _font[ch] ?? _font[' ']!;
      for (var cx = 0; cx < 5; cx++) {
        for (var cy = 0; cy < 7; cy++) {
          if (cols[cx] >> cy & 1 == 1) {
            rect(x + cx * scale, y + cy * scale, x + (cx + 1) * scale,
                y + (cy + 1) * scale);
          }
        }
      }
      x += 6 * scale;
    }
  }

  Uint8List toBmp() {
    const dataLen = stride * height;
    final out = Uint8List(62 + dataLen);
    final bd = ByteData.view(out.buffer);
    out[0] = 0x42; // 'B'
    out[1] = 0x4D; // 'M'
    bd.setUint32(2, out.length, Endian.little);
    bd.setUint32(10, 62, Endian.little);
    bd.setUint32(14, 40, Endian.little);
    bd.setInt32(18, width, Endian.little);
    bd.setInt32(22, height, Endian.little);
    bd.setUint16(26, 1, Endian.little);
    bd.setUint16(28, 1, Endian.little);
    bd.setUint32(34, dataLen, Endian.little);
    bd.setUint32(46, 2, Endian.little);
    out.setRange(54, 62, [0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00]);
    var o = 62;
    for (var y = height - 1; y >= 0; y--) {
      for (var xb = 0; xb < stride; xb++) {
        var b = 0;
        for (var bit = 0; bit < 8; bit++) {
          b = b << 1 | (lit[y * width + xb * 8 + bit] == 1 ? 0 : 1);
        }
        out[o++] = b;
      }
    }
    return out;
  }
}
