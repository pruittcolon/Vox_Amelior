import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';

import '../glasses/g1_font.dart';
import '../glasses/glasses_text.dart';
import '../log.dart';
import '../screen/mono_encoder.dart';
import '../suite.dart';

/// Claude lab: Claude, in Termux, does what the app can do with the glasses
/// (`glasses-lab` there, through Agent Hub) to try ideas out directly - text
/// the app's way, exact text screens, pictures, text drawn as a picture at any
/// size, raw Bluetooth packets, the app's own log. Off unless you turn it on
/// (Settings > Claude lab); it turns itself off after [idleOff] without a
/// command.
class GlassesLab extends ChangeNotifier {
  GlassesLab(this.suite);

  final Suite suite;

  bool get on => _on;
  bool _on = false;

  /// The last command and when, and how many so far.
  String lastCommand = '';
  DateTime? lastAt;
  int count = 0;

  /// Agent Hub could not be reached (null when it can).
  String? error;

  Duration idleOff = const Duration(minutes: 30);
  Timer? _idle;
  bool _looping = false;

  /// What the app answers to: name -> what it takes.
  static const commands = {
    'help': 'this list',
    'status': 'glasses link, battery, display, text layout',
    'text': "text, title? - shown the app's normal way (wrapped, paged)",
    'screen': 'lines[] or text, status? (hex, 71), page? (1), pages? (1), pos? (0) - one raw text screen (0x4E), '
        'no wrapping',
    'textbmp': 'text, size? (14), font? (sans|mono), bold?, lineHeight? (1.15), x? (8), y? (4), width? (560) - '
        'text drawn as a picture',
    'bmp': 'base64 (a 576x136 1-bit BMP) - shown as a picture',
    'raw': 'hex, side? (L|R|both), reply? (true), timeoutMs? (1500) - send bytes, return the replies',
    'clear': 'clear the glasses',
    'notice': 'text, seconds? (4) - a short notice',
    'events': "n? (30), cat? - the app's latest log events",
  };

  void setOn(bool value) {
    if (value == _on) return;
    _on = value;
    error = null;
    if (suite.settings.labOn != value) suite.settings.set('labOn', value);
    EventLog.add('lab', value ? 'on' : 'off');
    if (value) {
      _armIdle();
      unawaited(_loop());
    } else {
      _idle?.cancel();
      suite.client.abort('lab');
    }
    notifyListeners();
  }

  void _armIdle() {
    _idle?.cancel();
    _idle = Timer(idleOff, () {
      EventLog.add('lab', 'idleOff', {'minutes': idleOff.inMinutes});
      setOn(false);
    });
  }

  Future<void> _loop() async {
    if (_looping) return;
    _looping = true;
    var backoff = 1;
    try {
      while (_on) {
        try {
          final job = await suite.client.labNext(waitSec: 25);
          backoff = 1;
          if (error != null) {
            error = null;
            notifyListeners();
          }
          if (!_on || job == null) continue;
          final cmd = '${job['cmd'] ?? ''}';
          final args = <String, Object?>{...?(job['args'] as Map?)?.cast<String, Object?>()};
          final sw = Stopwatch()..start();
          Map<String, Object?> out;
          try {
            out = {'ok': true, 'result': await run(cmd, args)};
          } catch (e) {
            out = {'ok': false, 'error': '$e'};
          }
          EventLog.add('lab', 'run', {'cmd': cmd, 'ok': out['ok'], 'ms': sw.elapsedMilliseconds});
          lastCommand = cmd;
          lastAt = DateTime.now();
          count++;
          _armIdle();
          notifyListeners();
          // Whatever came back, as plain JSON.
          final safe = jsonDecode(jsonEncode(out, toEncodable: (o) => '$o')) as Map<String, dynamic>;
          await suite.client.labResult('${job['id'] ?? ''}',
              ok: safe['ok'] == true, result: safe['result'], error: safe['error'] as String?, ms: sw.elapsedMilliseconds);
        } catch (e) {
          if (!_on) break;
          if ('$e' != 'cancelled') {
            error = '$e';
            notifyListeners();
          }
          await Future.delayed(Duration(seconds: backoff));
          backoff = min(backoff * 2, 15);
        }
      }
    } finally {
      _looping = false;
    }
  }

  /// Carry out one command (public for tests).
  Future<Object?> run(String cmd, Map<String, Object?> args) async {
    final ble = suite.ble;
    String s(String k, [String d = '']) => args[k] == null ? d : '${args[k]}';
    num n(String k, num d) => args[k] is num ? args[k] as num : (num.tryParse(s(k)) ?? d);
    bool b(String k, bool d) => args[k] is bool ? args[k] as bool : (args[k] == null ? d : s(k) == 'true');
    void needGlasses() {
      if (!ble.isConnected) throw StateError('The glasses are not connected');
    }

    switch (cmd) {
      case 'help':
        return commands;
      case 'status':
        return {
          'connected': ble.isConnected,
          'link': ble.state.name,
          'batteryL': ble.batteryL,
          'batteryR': ble.batteryR,
          'display': suite.display.mode.name,
          'textLayout': {
            'width': GlassesText.maxWidth,
            'fontSize': GlassesText.fontSize,
            'lines': GlassesText.linesPerScreen,
            'charsPerLine': GlassesText.screenSize().charsPerLine,
          },
          'bitmap': {'width': MonoEncoder.width, 'height': MonoEncoder.height},
          'labCommands': count,
        };
      case 'text':
        needGlasses();
        final text = s('text');
        if (text.trim().isEmpty) throw ArgumentError('text is empty');
        suite.display.showText(text, title: s('title', 'Lab'));
        return {'pages': suite.display.pager?.pageCount};
      case 'screen':
        needGlasses();
        final lines = args['lines'] is List ? [for (final l in args['lines'] as List) '$l'] : s('text').split('\n');
        final text = lines.join('\n');
        final sw = Stopwatch()..start();
        final ok = await suite.proto.sendEvenAIData(text,
            newScreen: int.tryParse(s('status', '71'), radix: 16) ?? 0x71,
            pos: n('pos', 0).toInt(),
            currentPage: n('page', 1).toInt(),
            maxPage: n('pages', 1).toInt());
        return {
          'ok': ok,
          'ms': sw.elapsedMilliseconds,
          'lines': lines.length,
          'bytes': utf8.encode(text).length,
          'widestPx': lines.fold<int>(0, (w, l) => max(w, G1Font.width(l))),
        };
      case 'textbmp':
        needGlasses();
        final text = s('text');
        if (text.trim().isEmpty) throw ArgumentError('text is empty');
        final drawn = await drawText(text,
            size: n('size', 14).toDouble(),
            mono: s('font', 'sans') == 'mono',
            bold: b('bold', false),
            lineHeight: n('lineHeight', 1.15).toDouble(),
            x: n('x', 8).toDouble(),
            y: n('y', 4).toDouble(),
            width: n('width', 560).toDouble());
        final sw = Stopwatch()..start();
        final shown = await suite.proto.sendBmp(drawn.bmp);
        return {
          'shown': shown,
          'ms': sw.elapsedMilliseconds,
          'lines': drawn.lines,
          'linesShown': drawn.linesShown,
          'chars': text.length,
          'charsShown': drawn.charsShown,
        };
      case 'bmp':
        needGlasses();
        final bytes = base64.decode(s('base64').trim());
        if (bytes.length < 62 || bytes[0] != 0x42 || bytes[1] != 0x4D) throw ArgumentError('base64 is not a BMP');
        final sw = Stopwatch()..start();
        return {'shown': await suite.proto.sendBmp(Uint8List.fromList(bytes)), 'ms': sw.elapsedMilliseconds};
      case 'raw':
        needGlasses();
        final hex = s('hex').replaceAll(RegExp('[^0-9a-fA-F]'), '');
        if (hex.isEmpty || hex.length.isOdd) throw ArgumentError('hex: an even number of hex digits');
        final data = Uint8List.fromList([
          for (var i = 0; i < hex.length; i += 2) int.parse(hex.substring(i, i + 2), radix: 16),
        ]);
        final side = s('side', 'both').toUpperCase();
        final arms = side == 'BOTH' ? const ['L', 'R'] : [side];
        if (arms.any((a) => a != 'L' && a != 'R')) throw ArgumentError('side: L, R or both');
        final out = <String, Object?>{};
        for (final lr in arms) {
          if (b('reply', true)) {
            final r = await ble.request(data, lr: lr, timeoutMs: n('timeoutMs', 1500).toInt());
            out[lr] = r.isTimeout ? null : _hex(r.data);
          } else {
            await ble.sendData(data, lr: lr);
            out[lr] = 'sent';
          }
        }
        return out;
      case 'clear':
        await suite.display.clear();
        return 'cleared';
      case 'notice':
        needGlasses();
        suite.display.notice(s('text'), duration: Duration(milliseconds: (n('seconds', 4) * 1000).round()));
        return 'shown';
      case 'events':
        final cat = s('cat');
        final all = EventLog.instance.recent.where((e) => cat.isEmpty || e.cat == cat).toList();
        final k = n('n', 30).toInt().clamp(1, 200);
        return [for (final e in all.skip(max(0, all.length - k))) e.toJson()];
      default:
        throw ArgumentError('No command "$cmd". These exist: ${commands.keys.join(', ')}');
    }
  }

  static String _hex(Uint8List data) => data.map((v) => v.toRadixString(16).padLeft(2, '0')).join(' ');

  /// [text] drawn [size] px high into the glasses' 576x136 picture, lit where
  /// the text is, and how much of it fitted.
  static Future<DrawnText> drawText(String text,
      {double size = 14,
      bool mono = false,
      bool bold = false,
      double lineHeight = 1.15,
      double x = 8,
      double y = 4,
      double width = 560}) async {
    const w = MonoEncoder.width, h = MonoEncoder.height;
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: const ui.Color(0xFFFFFFFF),
          fontSize: size,
          height: lineHeight,
          fontFamily: mono ? 'monospace' : null,
          fontWeight: bold ? ui.FontWeight.w700 : ui.FontWeight.w400,
        ),
      ),
      textDirection: ui.TextDirection.ltr,
    )..layout(maxWidth: min(max(16.0, width), w - x));
    final metrics = painter.computeLineMetrics();
    var shown = 0;
    while (shown < metrics.length && y + metrics[shown].baseline + metrics[shown].descent <= h) {
      shown++;
    }
    var charsShown = text.length;
    if (shown < metrics.length) {
      final first = metrics[shown];
      charsShown = painter.getPositionForOffset(ui.Offset(0, first.baseline - first.ascent + 1)).offset;
    }
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    canvas.drawRect(ui.Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()), ui.Paint()..color = const ui.Color(0xFF000000));
    painter.paint(canvas, ui.Offset(x, y));
    painter.dispose();
    final image = await recorder.endRecording().toImage(w, h);
    final rgba = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    final lit = Uint8List(w * h);
    if (rgba != null) {
      for (var i = 0; i < w * h; i++) {
        lit[i] = rgba.getUint8(i * 4) > 110 ? 1 : 0;
      }
    }
    return DrawnText(MonoEncoder.bmp(lit), lines: metrics.length, linesShown: shown, charsShown: charsShown);
  }

  @override
  void dispose() {
    _idle?.cancel();
    _on = false;
    super.dispose();
  }
}

/// Text drawn as a picture for the glasses.
class DrawnText {
  const DrawnText(this.bmp, {required this.lines, required this.linesShown, required this.charsShown});

  final Uint8List bmp;
  final int lines;
  final int linesShown;
  final int charsShown;

  bool get cut => linesShown < lines;
}
