import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../ble/proto.dart';
import '../log.dart';
import 'mono_encoder.dart';

/// One answer from one arm during the picture test.
class LabResult {
  final String trial;
  final String step;
  final String arm;
  final String crc;
  const LabResult(this.trial, this.step, this.arm, this.crc);
}

/// Picture test: a scripted set of picture transfers that shows how these
/// particular glasses decide to accept (CRC 0xC9) or reject (0xCA) a
/// picture - new vs repeated pictures, waiting, clearing first, one arm at
/// a time, slow packets, dark vs bright, short pictures. Every answer goes
/// to the log (`lab.*`) so the sending logic can be designed from facts.
class PictureLab extends ChangeNotifier {
  PictureLab(this.proto);

  final Proto proto;
  bool running = false;
  String status = '';
  double progress = 0;
  final results = <LabResult>[];

  static Uint8List _pic(String name, {int rows = MonoEncoder.height}) {
    final g = Uint8List(MonoEncoder.width * rows);
    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < MonoEncoder.width; x++) {
        final lit = switch (name) {
          'A' => (x ~/ 12).isEven && (y ~/ 9) % 3 != 0, // vertical bars
          'B' => (y ~/ 6).isEven && (x ~/ 30) % 4 != 1, // horizontal bars
          'K' => x > 280 && x < 296 && y > 60 && y < 76, // almost all dark
          'W' => !(x > 280 && x < 296 && y > 60 && y < 76), // almost all lit
          _ => false,
        };
        g[y * MonoEncoder.width + x] = lit ? 255 : 0;
      }
    }
    return MonoEncoder.encode(g, rows: rows, invert: 'off', style: 'photo').bmp;
  }

  String _trial = '';
  int _done = 0;
  static const int _total = 30;

  void _log(String step, String arm, String crc, [Map<String, Object?> extra = const {}]) {
    results.add(LabResult(_trial, step, arm, crc));
    EventLog.add('lab', 'step', {'trial': _trial, 'step': step, 'arm': arm, 'crc': crc, ...extra});
  }

  Future<void> _send(String name, Uint8List bmp, {List<String> arms = const ['L', 'R'], int? gapMs}) async {
    final rs = await Future.wait([for (final lr in arms) proto.labTransfer(lr, bmp, gapMs: gapMs)]);
    for (final r in rs) {
      _log('send $name', r.lr, r.stage, {'packetsMs': r.packetsMs, 'endMs': r.endMs, 'bytes': bmp.length});
    }
    _done++;
    progress = _done / _total;
    notifyListeners();
  }

  Future<void> _crc(String name, Uint8List bmp, {List<String> arms = const ['L', 'R']}) async {
    final rs = await Future.wait([for (final lr in arms) proto.labCrc(lr, bmp)]);
    for (var i = 0; i < arms.length; i++) {
      _log('crc $name', arms[i], rs[i]);
    }
  }

  Future<void> _wait(int ms) => Future<void>.delayed(Duration(milliseconds: ms));

  void _start(String trial) {
    _trial = trial;
    status = trial;
    notifyListeners();
  }

  Future<void> run() async {
    if (running) return;
    running = true;
    results.clear();
    _done = 0;
    progress = 0;
    notifyListeners();
    EventLog.add('lab', 'start', {'gapMs': proto.bmpGapMs, 'chunk': proto.bmpChunk});
    final a = _pic('A'), b = _pic('B'), k = _pic('K'), w = _pic('W');
    final a68 = _pic('A', rows: 68), b68 = _pic('B', rows: 68);
    try {
      _start('1 new pictures (demo way)');
      for (final (n, p) in [('A', a), ('B', b), ('A', a), ('B', b)]) {
        await _send(n, p);
        await _crc(n, p);
      }
      _start('2 same picture again');
      await _send('B', b);
      await _crc('B', b);

      _start('3 new, then same at once');
      await _send('A', a);
      await _crc('A', a);
      await _send('A', a);
      await _crc('A', a);

      _start('4 new, pause 600 ms, same');
      await _send('B', b);
      await _crc('B', b);
      await _wait(600);
      await _send('B', b);
      await _crc('B', b);

      _start('5 wait 600 ms before the check');
      await _send('A', a);
      await _wait(600);
      await _crc('A', a);

      _start('6 check twice, no resend');
      await _send('B', b);
      await _crc('B', b);
      await _wait(300);
      await _crc('B', b);

      _start('7 clear first');
      await proto.exit();
      await _wait(600);
      await _send('A', a);
      await _crc('A', a);

      _start('8 one arm at a time');
      await _send('B', b, arms: ['L']);
      await _crc('B', b, arms: ['L']);
      await _send('B', b, arms: ['R']);
      await _crc('B', b, arms: ['R']);

      _start('9 slow packets (15 ms)');
      await _send('A', a, gapMs: 15);
      await _crc('A', a);
      await _send('B', b, gapMs: 15);
      await _crc('B', b);

      _start('10 dark and bright');
      for (final (n, p) in [('K', k), ('W', w), ('K', k), ('W', w)]) {
        await _send(n, p);
        await _crc(n, p);
      }

      _start('11 check the previous picture');
      await _send('A', a);
      await _crc('W (previous)', w);
      await _crc('A', a);

      _start('12 half height');
      await _send('A68', a68);
      await _crc('A68', a68);
      await _send('B68', b68);
      await _crc('B68', b68);
      await _send('B68', b68);
      await _crc('B68', b68);

      status = 'Done - results are in the log';
    } catch (e) {
      status = 'Stopped: $e';
      EventLog.add('lab', 'error', {'error': '$e'});
    } finally {
      EventLog.add('lab', 'done', {'answers': results.length});
      running = false;
      progress = 1;
      notifyListeners();
      unawaited(EventLog.instance.flush());
    }
  }
}
