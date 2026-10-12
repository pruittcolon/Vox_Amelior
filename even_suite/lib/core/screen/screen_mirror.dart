import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../ble/proto.dart';
import '../log.dart';
import 'mono_encoder.dart';

/// Where screen frames come from (Android screen capture, or a test fake).
abstract class ScreenSource {
  /// Asks Android for permission and starts capturing.
  /// Returns 'ok', 'denied', 'busy' or 'error: ...'.
  Future<String> start(Map<String, Object> config);
  Future<void> stop();
  Future<void> configure(Map<String, Object> config);

  /// Frames: {gray, x0, y0, x1, y1, ms}, {same: true} when nothing changed,
  /// or {stopped: true, reason} when sharing ended.
  Stream<Map> get frames;
}

class PlatformScreen implements ScreenSource {
  static const _methods = MethodChannel('suite/screen');
  static final Stream<Map> _frames =
      const EventChannel('suite/screenFrames').receiveBroadcastStream().map((e) => e as Map);

  @override
  Stream<Map> get frames => _frames;

  @override
  Future<String> start(Map<String, Object> config) async {
    try {
      return await _methods.invokeMethod<String>('start', config) ?? 'error';
    } on MissingPluginException {
      return 'error: only on the phone';
    }
  }

  @override
  Future<void> stop() => _methods.invokeMethod('stop');

  @override
  Future<void> configure(Map<String, Object> config) => _methods.invokeMethod('configure', config);
}

/// Phone screen on the glasses, about one frame a second.
///
/// Android captures a small grayscale frame; here it becomes a 1-bit BMP and
/// goes out through the fast image path. Only the newest frame is ever sent,
/// a frame identical to the one on the glasses is skipped, and a frame the
/// glasses did not confirm is retried on the next tick. It turns itself off
/// after [autoOff], like hands-free.
class ScreenMirror extends ChangeNotifier {
  ScreenMirror(this.source, this.proto) {
    _sub = source.frames.listen(_onFrame, onError: (_) {});
  }

  final ScreenSource source;
  final Proto proto;
  StreamSubscription<Map>? _sub;

  // Settings (pushed from the app's Settings)
  String region = 'strip';
  int strip = 0;
  // Custom area (percent of the screen) and how it maps onto the glasses.
  int cx = 0, cy = 0, cw = 100, ch = 30;
  String fit = 'fit';
  int rows = 136;
  int fpsTenths = 10;
  String style = 'text';
  String invert = 'auto';
  Duration autoOff = const Duration(minutes: 10);

  /// Clock for the frame-rate figure (tests use fake time).
  DateTime Function() now = DateTime.now;

  /// Waits for the glasses (see [Proto.sendBmp]); shorter in tests.
  int endTimeoutMs = 3000;
  int crcTimeoutMs = 2000;

  /// Pause after a missed frame before the next one (off: the sender
  /// already retries the arm that rejected a picture, more slowly).
  Duration coolDown = Duration.zero;

  /// A frame counts once both arms accepted it (CRC match) - the glasses
  /// only display it then.
  bool requireCrc = true;

  // State
  bool running = false;
  bool starting = false;
  String? error;
  Uint8List? preview; // the last 1-bit frame, as a BMP the phone can draw
  int sent = 0;
  int skipped = 0;
  int failed = 0;
  int lastSendMs = 0;
  DateTime? offAt;

  Timer? _off;
  Uint8List? _pending;
  Uint8List? _shown;
  Uint8List? _lastFailed;
  bool _sending = false;
  final _times = <int>[];

  /// Frames per second actually reaching the glasses (last 10 frames).
  double get fps {
    if (_times.length < 2) return 0;
    final span = _times.last - _times.first;
    return span <= 0 ? 0 : (_times.length - 1) * 1000 / span;
  }

  Map<String, Object> get config => {
        'region': region,
        'strip': strip,
        'intervalMs': (10000 / fpsTenths).round(),
        'cx': cx,
        'cy': cy,
        'cw': cw,
        'ch': ch,
        'fit': fit,
        'rows': rows,
      };

  Future<void> start() async {
    if (running || starting) return;
    starting = true;
    error = null;
    notifyListeners();
    final r = await source.start(config);
    starting = false;
    EventLog.add('screen', 'start', {...config, 'style': style, 'invert': invert, 'answer': r});
    if (r == 'ok') {
      running = true;
      sent = skipped = failed = 0;
      _capturedInterval = (10000 / fpsTenths).round();
      _times.clear();
      _shown = null;
      _armOff();
    } else if (r == 'denied') {
      error = 'Screen sharing was not allowed';
    } else if (r != 'busy') {
      error = r.replaceFirst('error: ', '');
    }
    notifyListeners();
  }

  Future<void> stop({bool clearGlasses = true}) async {
    if (!running && !starting) return;
    EventLog.add('screen', 'stop', {'sent': sent, 'skipped': skipped, 'failed': failed});
    running = false;
    starting = false;
    _off?.cancel();
    offAt = null;
    _pending = null;
    notifyListeners();
    try {
      await source.stop();
    } catch (_) {}
    if (clearGlasses && proto.ble.isConnected) await proto.exit();
  }

  /// Apply changed settings while running.
  Future<void> reconfigure() async {
    _shown = null; // style changes must reach the glasses
    if (running) await source.configure(config);
  }

  void _armOff() {
    _off?.cancel();
    offAt = DateTime.now().add(autoOff);
    _off = Timer(autoOff, () {
      error = 'Screen sharing turned itself off after ${autoOff.inMinutes} min';
      stop();
    });
  }

  /// Keep sharing for another [autoOff].
  void extend() {
    if (running) {
      _armOff();
      notifyListeners();
    }
  }

  void _onFrame(Map f) {
    if (f['stopped'] == true) {
      final reason = '${f['reason'] ?? ''}';
      EventLog.add('screen', 'ended', {'reason': reason, 'sent': sent});
      if (reason.isNotEmpty) error = reason;
      if (running) {
        running = false;
        _off?.cancel();
        offAt = null;
        if (proto.ble.isConnected) proto.exit();
      }
      notifyListeners();
      return;
    }
    if (!running) return;
    if (f['same'] == true) {
      // Nothing new on screen; resend a frame the glasses missed.
      final missed = _lastFailed;
      if (missed != null) push(missed);
      return;
    }
    final gray = f['gray'];
    final h = (f['h'] as num?)?.toInt() ?? MonoEncoder.height;
    if (gray is! Uint8List || gray.length != MonoEncoder.width * h) return;
    final enc = Stopwatch()..start();
    final frame = MonoEncoder.encode(gray,
        rows: h,
        x0: (f['x0'] as num?)?.toInt() ?? 0,
        y0: (f['y0'] as num?)?.toInt() ?? 0,
        x1: (f['x1'] as num?)?.toInt() ?? MonoEncoder.width,
        y1: (f['y1'] as num?)?.toInt() ?? h,
        style: style,
        invert: invert);
    preview = frame.bmp;
    EventLog.add('screen', 'capture', {
      'captureMs': f['ms'],
      'encodeMs': enc.elapsedMilliseconds,
      'lit': (frame.litShare * 100).round(),
      'flipped': frame.flipped,
      'style': style,
      'busy': _sending,
    });
    notifyListeners();
    push(frame.bmp);
  }

  /// Queue a frame; only the newest waiting frame is kept.
  @visibleForTesting
  void push(Uint8List bmp) {
    _pending = bmp;
    if (!_sending) _loop();
  }

  Future<void> _loop() async {
    _sending = true;
    try {
      while (_pending != null && running) {
        final frame = _pending!;
        _pending = null;
        if (_shown != null && listEquals(frame, _shown)) {
          skipped++;
          continue;
        }
        if (!proto.ble.isConnected) {
          _lastFailed = frame;
          continue;
        }
        final t0 = now().millisecondsSinceEpoch;
        final ok = await proto.sendBmp(frame,
            endTimeoutMs: endTimeoutMs, crcTimeoutMs: crcTimeoutMs, requireCrc: requireCrc);
        final t1 = now().millisecondsSinceEpoch;
        lastSendMs = t1 - t0;
        EventLog.add('screen', 'sent', {'ok': ok, 'ms': lastSendMs, 'sent': sent + (ok ? 1 : 0)});
        _adapt();
        if (ok) {
          sent++;
          _shown = frame;
          _lastFailed = null;
          _times.add(t1);
          if (_times.length > 10) _times.removeAt(0);
        } else {
          failed++;
          _shown = null;
          _lastFailed = frame;
          notifyListeners();
          if (coolDown > Duration.zero) await Future.delayed(coolDown);
        }
        notifyListeners();
      }
    } finally {
      _sending = false;
    }
  }

  int _capturedInterval = 0;
  final _recentSends = <int>[];

  /// Do not capture much faster than the glasses take frames: pace capture to
  /// the median of the last few send times (one slow frame does not slow
  /// everything down), never more than twice the chosen interval.
  void _adapt() {
    final want = (10000 / fpsTenths).round();
    _recentSends.add(lastSendMs);
    if (_recentSends.length > 5) _recentSends.removeAt(0);
    final sorted = [..._recentSends]..sort();
    final median = sorted[sorted.length ~/ 2];
    var need = median > want ? ((median * 1.1) ~/ 100 + 1) * 100 : want;
    if (need > want * 2) need = want * 2;
    if (need != _capturedInterval) {
      _capturedInterval = need;
      EventLog.add('screen', 'interval', {'ms': need, 'sendMs': lastSendMs});
      if (running) source.configure({...config, 'intervalMs': need});
    }
  }

  @override
  void dispose() {
    _sub?.cancel();
    _off?.cancel();
    super.dispose();
  }
}
