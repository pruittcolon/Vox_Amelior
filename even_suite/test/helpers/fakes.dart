import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/voice/speech.dart';

/// Simulated G1: acknowledges commands the way the real arms do and records
/// every packet per arm.
class FakeGlasses implements BleTransport {
  final _rx = StreamController<BleReceive>.broadcast();
  final Map<String, List<Uint8List>> sent = {'L': [], 'R': []};

  /// Arms that ignore everything (to test timeouts).
  final Set<String> deaf = {};

  /// Commands to ignore (e.g. {0x4E}).
  final Set<int> ignoreCmds = {};

  @override
  Stream<BleReceive> get receive => _rx.stream;

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      sent[arm]!.add(data);
      if (deaf.contains(arm) || ignoreCmds.contains(data[0])) continue;
      final reply = _reply(data);
      if (reply != null) {
        scheduleMicrotask(() => _rx.add(BleReceive()
          ..lr = arm
          ..data = reply
          ..type = 'Receive'));
      }
    }
  }

  @override
  Future<int> sendBatch(List<Uint8List> packets, String lr) async {
    if (deaf.contains(lr)) return 0;
    sent[lr]!.addAll(packets);
    return packets.length;
  }

  Uint8List? _reply(Uint8List d) {
    switch (d[0]) {
      case 0x25: // heartbeat echoes its sequence
        return Uint8List.fromList([0x25, 6, 0, d[3], 0x04, d[5]]);
      case 0x16: // CRC: status in byte 5
        return Uint8List.fromList([0x16, d[1], d[2], d[3], d[4], 0xC9]);
      case 0x15: // image chunks are not acknowledged
        return null;
      default:
        return Uint8List.fromList([d[0], 0xC9]);
    }
  }

  /// Simulate a TouchBar event.
  void touch(int code, String lr) => _rx.add(BleReceive()
    ..lr = lr
    ..data = Uint8List.fromList([0xF5, code])
    ..type = 'Receive');

  /// 0x4E screens received by [arm], decoded to text.
  List<String> screens([String arm = 'L']) => [
        for (final p in sent[arm]!)
          if (p[0] == 0x4E) String.fromCharCodes(p.sublist(9)),
      ];

  void clear() {
    sent['L']!.clear();
    sent['R']!.clear();
  }
}

/// Scanning/connecting stand-in; [emit] plays native callbacks.
class FakeControl implements BleControl {
  Future<void> Function(String method, Map args)? _handler;
  final calls = <String>[];

  @override
  void setHandler(Future<void> Function(String method, Map args) handler) => _handler = handler;

  Future<void> emit(String method, Map args) => _handler!(method, args);

  @override
  Future<String?> startScan() async {
    calls.add('startScan');
    return null;
  }

  @override
  Future<void> stopScan() async => calls.add('stopScan');

  @override
  Future<String?> connect(GlassesPair pair) async {
    calls.add('connect ${pair.channel}');
    return null;
  }

  @override
  Future<void> disconnect() async => calls.add('disconnect');

  static const pairInfo = {
    'channelNumber': '45',
    'leftDeviceName': 'G1_45_L_92333',
    'rightDeviceName': 'G1_45_R_92334',
    'leftAddress': 'AA:BB:CC:DD:EE:01',
    'rightAddress': 'AA:BB:CC:DD:EE:02',
  };

  Future<void> connectNow() => emit('glassesConnected', pairInfo);
}

/// Speech recognizer stand-in: tests push partial/final results.
class FakeSpeech implements SpeechEngine {
  final _events = StreamController<SpeechEvent>.broadcast();
  int starts = 0;
  int stops = 0;
  int finishes = 0;
  bool running = false;

  @override
  Stream<SpeechEvent> get events => _events.stream;

  @override
  Future<bool> available() async => true;

  @override
  Future<void> start({
    bool continuous = true,
    String language = '',
    bool preferOffline = true,
    bool quiet = true,
    String engine = 'auto',
  }) async {
    starts++;
    running = true;
    _events.add(const SpeechEvent('state', listening: true));
  }

  @override
  Future<void> stop() async {
    stops++;
    running = false;
    _events.add(const SpeechEvent('state', listening: false));
  }

  @override
  Future<void> finish() async => finishes++;

  void partial(String text) => _events.add(SpeechEvent('partial', text: text));

  /// A finished utterance (the recognizer then starts the next one).
  void say(String text, {bool finishing = false}) {
    _events.add(SpeechEvent('final', text: text, finishing: finishing));
    _events.add(SpeechEvent('end', finishing: finishing));
  }

  void silence({bool finishing = false}) => _events.add(SpeechEvent('end', finishing: finishing));

  void error(String message, {bool fatal = false}) =>
      _events.add(SpeechEvent('error', message: message, fatal: fatal));
}

/// In-memory bridge for widget tests: no sockets, no timers. Long polls wait
/// on a completer until something changes.
class FakeBridgeClient extends BridgeClient {
  FakeBridgeClient() : super('http://fake');

  int version = 1;
  final messages = <BridgeMessage>[];
  final sentTexts = <String>[];
  final pasted = <String>[];
  final keysSent = <String>[];
  String screen = r'~ $ claude' '\n' '* Ready. Ask me anything.\n> ';
  bool online = true;
  Completer<void> _changed = Completer();

  void _bump() {
    version++;
    final c = _changed;
    _changed = Completer();
    c.complete();
  }

  void addReply(String text, {String status = 'done'}) {
    messages.add(BridgeMessage(
        id: 'r$version', who: 'claude', text: text, status: status, ts: DateTime.now().millisecondsSinceEpoch / 1000));
    _bump();
  }

  @override
  Future<Map<String, bool>> status() async {
    if (!online) throw const BridgeException('Termux bridge is not running');
    return {'claude': true, 'shell': true, 'agy': true};
  }

  @override
  Future<BridgeSnapshot> snapshot({int version = -1, int waitSec = 0, int tail = 60}) async {
    if (!online) throw const BridgeException('Termux bridge is not running');
    if (version == this.version && waitSec > 0) await _changed.future;
    return BridgeSnapshot(this.version, List.of(messages), const []);
  }

  @override
  Future<void> send(String text, {String to = 'claude', int turns = 0}) async {
    sentTexts.add(text);
    messages.add(BridgeMessage(id: 'u$version', who: 'you', text: text, status: 'done'));
    _bump();
  }

  @override
  Future<TermScreen> capture(String agent, {int lines = 120, int waitSec = 0, String hash = ''}) async {
    final h = '${screen.hashCode}';
    if (hash == h && waitSec > 0) await _changed.future;
    return TermScreen(screen, '${screen.hashCode}');
  }

  @override
  Future<void> paste(String agent, String text, {bool enter = true}) async {
    pasted.add(text);
    screen = '$screen$text\n';
    _bump();
  }

  @override
  Future<void> keys(String agent, List<String> keys) async => keysSent.addAll(keys);

  @override
  Future<void> stop() async {}

  @override
  Future<void> reset() async {
    messages.clear();
    _bump();
  }

  @override
  void abortAll() {}
}
