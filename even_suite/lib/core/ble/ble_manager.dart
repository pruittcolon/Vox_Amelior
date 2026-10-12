import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../log.dart';
import 'ble_receive.dart';

enum LinkState { disconnected, scanning, connecting, connected }

/// A left + right arm pair found by scanning (or remembered from last time).
class GlassesPair {
  final String channel;
  final String leftName;
  final String rightName;
  final String leftAddress;
  final String rightAddress;

  const GlassesPair({
    required this.channel,
    required this.leftName,
    required this.rightName,
    this.leftAddress = '',
    this.rightAddress = '',
  });

  factory GlassesPair.fromMap(Map map) => GlassesPair(
        channel: '${map['channelNumber'] ?? ''}',
        leftName: '${map['leftDeviceName'] ?? ''}',
        rightName: '${map['rightDeviceName'] ?? ''}',
        leftAddress: '${map['leftAddress'] ?? ''}',
        rightAddress: '${map['rightAddress'] ?? ''}',
      );

  Map<String, String> toMap() => {
        'channelNumber': channel,
        'leftDeviceName': leftName,
        'rightDeviceName': rightName,
        'leftAddress': leftAddress,
        'rightAddress': rightAddress,
      };

  /// "G1 · 45"
  String get label {
    final model = leftName.split('_').first;
    return '${model.isEmpty ? 'G1' : model} · $channel';
  }
}

/// Packet transport to the glasses. The real one is the Android channel; tests
/// plug in simulated glasses.
abstract class BleTransport {
  Stream<BleReceive> get receive;

  /// Write [data] to one arm ('L'/'R'), or both when [lr] is null.
  Future<void> send(Uint8List data, String? lr);

  /// Write many packets to one arm, [gapMs] apart; returns how many were written.
  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0});
}

/// Scanning and connecting (Android side of the demo's BleManager).
abstract class BleControl {
  Future<String?> startScan();
  Future<void> stopScan();
  Future<String?> connect(GlassesPair pair);
  Future<void> disconnect();
  void setHandler(Future<void> Function(String method, Map args) handler);
}

class PlatformBle implements BleTransport, BleControl {
  static const _channel = MethodChannel('method.bluetooth');
  static final Stream<BleReceive> _receive = const EventChannel('eventBleReceive')
      .receiveBroadcastStream('eventBleReceive')
      .map((e) => BleReceive.fromMap(e as Map));

  @override
  Stream<BleReceive> get receive => _receive;

  @override
  Future<void> send(Uint8List data, String? lr) =>
      _channel.invokeMethod('send', {'data': data, 'lr': lr});

  @override
  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0}) async =>
      await _channel.invokeMethod<int>('sendBatch', {'lr': lr, 'packets': packets, 'gapMs': gapMs}) ?? 0;

  @override
  Future<String?> startScan() => _call('startScan');

  @override
  Future<void> stopScan() => _call('stopScan');

  /// Ask for the faster Bluetooth 5 radio mode on the next connect.
  bool phy2m = true;

  @override
  Future<String?> connect(GlassesPair pair) => _call('connectToGlasses', {
        'channel': pair.channel,
        'saved': pair.toMap(),
        'phy2m': phy2m,
      });

  @override
  Future<void> disconnect() => _call('disconnectFromGlasses');

  @override
  void setHandler(Future<void> Function(String method, Map args) handler) {
    _channel.setMethodCallHandler(
        (call) => handler(call.method, (call.arguments as Map?) ?? const {}));
  }

  /// Returns an error message, or null.
  Future<String?> _call(String method, [Object? args]) async {
    try {
      await _channel.invokeMethod(method, args);
      return null;
    } on PlatformException catch (e) {
      return e.message ?? e.code;
    } on MissingPluginException {
      return 'Bluetooth is only available on the phone';
    }
  }
}

/// Request/response layer from the demo's BleManager, plus link state.
///
/// Replies are matched to requests by arm + command byte, exactly like the
/// demo. `0xF5` packets are TouchBar events and go to [events].
class BleManager extends ChangeNotifier {
  BleManager(this.transport, {this.control}) {
    _sub = transport.receive.listen(_onReceive, onError: (_) {});
    control?.setHandler(_onNative);
  }

  final BleTransport transport;
  final BleControl? control;
  StreamSubscription<BleReceive>? _sub;

  LinkState state = LinkState.disconnected;
  final List<GlassesPair> found = [];
  GlassesPair? current;
  bool leftUp = false;
  bool rightUp = false;
  String? lastError;
  Timer? _scanTimer;

  /// Called when both arms link up (the app saves the pair here).
  void Function(GlassesPair pair)? onConnected;

  final _events = StreamController<GlassesEvent>.broadcast();
  Stream<GlassesEvent> get events => _events.stream;

  bool get isConnected => state == LinkState.connected;

  // ------------------------------------------------------------ link state

  Future<void> startScan() async {
    final c = control;
    if (c == null) return;
    lastError = null;
    found.clear();
    final error = await c.startScan();
    if (error != null) {
      lastError = error;
      notifyListeners();
      return;
    }
    if (state != LinkState.connected) state = LinkState.scanning;
    notifyListeners();
    _scanTimer?.cancel();
    _scanTimer = Timer(const Duration(seconds: 15), stopScan);
  }

  Future<void> stopScan() async {
    _scanTimer?.cancel();
    await control?.stopScan();
    if (state == LinkState.scanning) {
      state = LinkState.disconnected;
      notifyListeners();
    }
  }

  Future<void> connect(GlassesPair pair) async {
    final c = control;
    if (c == null) return;
    _scanTimer?.cancel();
    lastError = null;
    current = pair;
    state = LinkState.connecting;
    notifyListeners();
    final error = await c.connect(pair);
    if (error != null) {
      lastError = error;
      state = LinkState.disconnected;
      notifyListeners();
    }
  }

  Future<void> disconnect() async {
    await control?.disconnect();
    state = LinkState.disconnected;
    leftUp = rightUp = false;
    notifyListeners();
  }

  Future<void> _onNative(String method, Map args) async {
    if (method != 'foundPairedGlasses') {
      EventLog.add('glasses', method, {'arm': args['arm'], 'reason': args['reason']});
    }
    switch (method) {
      case 'foundPairedGlasses':
        final pair = GlassesPair.fromMap(args);
        if (!found.any((p) => p.channel == pair.channel)) {
          found.add(pair);
          notifyListeners();
        }
      case 'glassesConnecting':
        if (state != LinkState.connected) state = LinkState.connecting;
        notifyListeners();
      case 'glassesConnected':
        final pair = GlassesPair.fromMap(args);
        current = pair;
        state = LinkState.connected;
        leftUp = rightUp = true;
        lastError = null;
        _scanTimer?.cancel();
        notifyListeners();
        onConnected?.call(pair);
      case 'glassesDisconnected':
        final arm = args['arm'];
        if (arm == 'L') leftUp = false;
        if (arm == 'R') rightUp = false;
        if (arm == null) leftUp = rightUp = false;
        // The native side keeps reconnecting unless the user disconnected.
        state = args['reason'] == 'user' || args['reason'] == 'bluetooth-off'
            ? LinkState.disconnected
            : LinkState.connecting;
        if (args['reason'] == 'bluetooth-off') lastError = 'Bluetooth is off';
        notifyListeners();
    }
  }

  // ------------------------------------------------------- request/response

  final _reqListen = <String, Completer<BleReceive>>{};
  final _reqTimeout = <String, Timer>{};
  Completer<BleReceive>? _nextReceive;

  void _onReceive(BleReceive res) {
    if (res.data.isEmpty || res.type == 'VoiceChunk') return;
    if (res.data[0] == 0xF5 && res.data.length > 1) {
      EventLog.add('glasses', 'touch', {'code': res.data[1], 'lr': res.lr});
      _events.add(GlassesEvent(res.data[1], res.lr));
      return;
    }
    final cmd = _key(res.lr, res.data[0]);
    _reqListen.remove(cmd)?.complete(res);
    _reqTimeout.remove(cmd)?.cancel();
    final next = _nextReceive;
    if (next != null && !next.isCompleted) {
      _nextReceive = null;
      next.complete(res);
    }
  }

  static String _key(String lr, int cmd) =>
      '$lr${cmd.toRadixString(16).padLeft(2, '0')}';

  Future<void> sendData(Uint8List data, {String? lr}) =>
      transport.send(data, lr);

  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0}) async {
    final sw = Stopwatch()..start();
    final sent = await transport.sendBatch(packets, lr, gapMs: gapMs);
    EventLog.add('ble', 'batch', {
      'lr': lr,
      'n': packets.length,
      'sent': sent,
      'gapMs': gapMs,
      'bytes': packets.fold<int>(0, (a, p) => a + p.length),
      'ms': sw.elapsedMilliseconds,
    });
    return sent;
  }

  /// Send [data] and wait for the reply with the same command byte.
  Future<BleReceive> request(Uint8List data,
      {String? lr, int timeoutMs = 1000, bool useNext = false}) async {
    final lr0 = lr ?? 'R';
    final completer = Completer<BleReceive>();
    final cmd = _key(lr0, data[0]);

    if (useNext) {
      _nextReceive = completer;
    } else {
      final old = _reqListen.remove(cmd);
      if (old != null && !old.isCompleted) old.complete(BleReceive.timeout());
      _reqTimeout.remove(cmd)?.cancel();
      _reqListen[cmd] = completer;
    }

    if (timeoutMs > 0) {
      _reqTimeout[cmd] = Timer(Duration(milliseconds: timeoutMs), () {
        _reqTimeout.remove(cmd);
        final cb = useNext ? completer : _reqListen.remove(cmd);
        if (cb != null && !cb.isCompleted) cb.complete(BleReceive.timeout());
        if (useNext && identical(_nextReceive, completer)) _nextReceive = null;
      });
    }

    final sw = Stopwatch()..start();
    try {
      await transport.send(data, lr0).timeout(const Duration(seconds: 2));
    } catch (_) {
      _reqTimeout.remove(cmd)?.cancel();
      final cb = _reqListen.remove(cmd);
      if (cb != null && !cb.isCompleted) cb.complete(BleReceive.timeout());
    }
    final r = await completer.future;
    EventLog.add('ble', 'req', {
      'cmd': data[0].toRadixString(16).padLeft(2, '0'),
      'lr': lr0,
      'bytes': data.length,
      'ms': sw.elapsedMilliseconds,
      'ok': !r.isTimeout,
      // Raw reply (not for the 8 s heartbeat), to learn what the glasses answer.
      if (!r.isTimeout && data[0] != 0x25)
        'reply': r.data.take(12).map((b) => b.toRadixString(16).padLeft(2, '0')).join(' '),
    });
    return r;
  }

  Future<BleReceive> requestRetry(Uint8List data,
      {String? lr, int timeoutMs = 200, int retry = 3}) async {
    for (var i = 0; i <= retry; i++) {
      final ret = await request(data, lr: lr, timeoutMs: timeoutMs);
      if (!ret.isTimeout) return ret;
    }
    return BleReceive.timeout();
  }

  /// Left first; the right arm only after the left acknowledged (demo order).
  Future<bool> sendBoth(Uint8List data,
      {int timeoutMs = 250, bool Function(Uint8List)? isSuccess, int retry = 0}) async {
    final ret = await requestRetry(data, lr: 'L', timeoutMs: timeoutMs, retry: retry);
    if (ret.isTimeout) return false;
    if (isSuccess != null) {
      if (!isSuccess(ret.data)) return false;
      final retR = await requestRetry(data, lr: 'R', timeoutMs: timeoutMs, retry: retry);
      return !retR.isTimeout && isSuccess(retR.data);
    }
    if (ret.data.length > 1 && ret.data[1] == 0xc9) {
      final retR = await requestRetry(data, lr: 'R', timeoutMs: timeoutMs, retry: retry);
      if (retR.isTimeout) return false;
    }
    return true;
  }

  /// Multi-packet transfer: every packet must be acknowledged (0xC9/0xCB).
  Future<bool> requestList(List<Uint8List> sendList, {String? lr, int? timeoutMs}) async {
    if (lr != null) return _requestList(sendList, lr, timeoutMs: timeoutMs);
    final rets = await Future.wait([
      _requestList(sendList, 'L', keepLast: true, timeoutMs: timeoutMs),
      _requestList(sendList, 'R', keepLast: true, timeoutMs: timeoutMs),
    ]);
    if (rets[0] && rets[1]) {
      return sendBoth(sendList.last, timeoutMs: timeoutMs ?? 250);
    }
    return false;
  }

  Future<bool> _requestList(List<Uint8List> sendList, String lr,
      {bool keepLast = false, int? timeoutMs}) async {
    final len = keepLast ? sendList.length - 1 : sendList.length;
    for (var i = 0; i < len; i++) {
      final resp = await request(sendList[i], lr: lr, timeoutMs: timeoutMs ?? 350);
      if (resp.isTimeout) return false;
      if (resp.data.length < 2 || (resp.data[1] != 0xc9 && resp.data[1] != 0xcb)) {
        return false;
      }
    }
    return true;
  }

  @override
  void dispose() {
    _sub?.cancel();
    _scanTimer?.cancel();
    for (final t in _reqTimeout.values) {
      t.cancel();
    }
    _events.close();
    super.dispose();
  }
}
