import 'dart:typed_data';

/// One packet from the glasses (same shape as the demo's BleReceive).
class BleReceive {
  String lr = '';
  Uint8List data = Uint8List(0);
  String type = '';
  bool isTimeout = false;

  BleReceive();

  BleReceive.timeout() : isTimeout = true;

  factory BleReceive.fromMap(Map map) => BleReceive()
    ..lr = map['lr'] as String? ?? ''
    ..data = map['data'] as Uint8List? ?? Uint8List(0)
    ..type = map['type'] as String? ?? '';

  int getCmd() => data.isEmpty ? -1 : data[0];

  String get hexString =>
      data.map((e) => e.toRadixString(16).padLeft(2, '0')).join(' ');
}

/// TouchBar and system events the glasses push as `0xF5 <code>`.
class GlassesEvent {
  static const int exit = 0; // double tap: back to the dashboard
  static const int tap = 1; // single tap: left = up, right = down
  static const int evenAiStart = 23; // long press left: start talking
  static const int evenAiEnd = 24; // long press released

  final int code;
  final String lr;

  const GlassesEvent(this.code, this.lr);

  bool get isTapLeft => code == tap && lr == 'L';
  bool get isTapRight => code == tap && lr == 'R';

  @override
  String toString() => 'GlassesEvent($code, $lr)';
}
