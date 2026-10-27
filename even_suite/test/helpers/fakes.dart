import 'dart:async';
import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/ble_receive.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:even_suite/core/system.dart';
import 'package:even_suite/core/voice/speaker.dart';
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

  /// Ignore the first N packets of a command (e.g. {0x20: 2}), then answer.
  final Map<int, int> ignoreFirst = {};

  /// Simulated Bluetooth round trip (0 = reply at once).
  Duration replyDelay = Duration.zero;

  @override
  Stream<BleReceive> get receive => _rx.stream;

  @override
  Future<void> send(Uint8List data, String? lr) async {
    for (final arm in lr == null ? ['L', 'R'] : [lr]) {
      sent[arm]!.add(data);
      if (deaf.contains(arm) || ignoreCmds.contains(data[0])) continue;
      final skip = ignoreFirst[data[0]] ?? 0;
      if (skip > 0) {
        ignoreFirst[data[0]] = skip - 1;
        continue;
      }
      final reply = _reply(data);
      if (reply != null) {
        void deliver() => _rx.add(BleReceive()
          ..lr = arm
          ..data = reply
          ..type = 'Receive');
        if (replyDelay == Duration.zero) {
          scheduleMicrotask(deliver);
        } else {
          Timer(replyDelay, deliver);
        }
      }
    }
  }

  final batchGaps = <int>[];

  @override
  Future<int> sendBatch(List<Uint8List> packets, String lr, {int gapMs = 0}) async {
    batchGaps.add(gapMs);
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

  /// Deliver a raw reply from one arm (for custom fakes).
  void injectReply(String lr, Uint8List data) => _rx.add(BleReceive()
    ..lr = lr
    ..data = data
    ..type = 'Receive');

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
  String lastSource = '';

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
    String audioSource = 'phone',
    int silenceMs = 1500,
  }) async {
    starts++;
    running = true;
    lastSource = audioSource;
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

  String? lastSystem;
  Map<String, String>? lastOptions;
  int warms = 0;

  @override
  Future<void> send(String text,
      {String to = 'claude', int turns = 0, String? system, Map<String, String>? options}) async {
    sentTexts.add(text);
    lastSystem = system;
    lastOptions = options;
    messages.add(BridgeMessage(id: 'u$version', who: 'you', text: text, status: 'done'));
    _bump();
  }

  /// Claude's reply on its way: a stage, a tool step, or text so far.
  void progress({String stage = 'thinking', String tool = '', String live = '', int block = 0}) {
    messages.removeWhere((m) => m.id == 'p$_turn');
    messages.add(BridgeMessage(
        id: 'p$_turn',
        who: 'claude',
        text: '',
        status: 'pending',
        stage: stage,
        tool: tool,
        live: live,
        block: block,
        steps: [if (tool.isNotEmpty) tool],
        ts: DateTime.now().millisecondsSinceEpoch / 1000));
    _bump();
  }

  /// Finish the reply started with [progress].
  void finish(String text, {int block = 1}) {
    messages.removeWhere((m) => m.id == 'p$_turn');
    messages.add(BridgeMessage(
        id: 'p$_turn', who: 'claude', text: text, status: 'done', stage: 'done', block: block,
        ts: DateTime.now().millisecondsSinceEpoch / 1000));
    _turn++;
    _bump();
  }

  int _turn = 0;

  @override
  Future<void> warm({String? system, Map<String, String>? options}) async => warms++;

  bool hook = true;

  @override
  Future<HubInfo> ping() async => HubInfo(3, const {'live', 'warm', 'work'}, hook: hook);

  @override
  Future<bool> installHook() async => hook = true;

  final workItems = <WorkItem>[];
  final workJobs = <WorkJob>[];
  int workVersion = 1;

  void addWork(WorkItem item) {
    workItems.add(item);
    workVersion++;
    _bump();
  }

  @override
  Future<WorkSnapshot> work({int version = -1, int waitSec = 0}) async {
    if (!online) throw const BridgeException('Termux bridge is not running');
    if (version == workVersion && waitSec > 0) await _changed.future;
    return WorkSnapshot(workVersion, List.of(workItems), List.of(workJobs));
  }

  @override
  Future<void> clearWork() async {
    workItems.clear();
    workVersion++;
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

  final logged = <Map<String, Object?>>[];

  @override
  Future<void> log(List<Map<String, Object?>> events, {String session = '', int dropped = 0}) async =>
      logged.addAll(events);

  @override
  void abortAll() {}
}

/// Android side of the app (permissions, settings, service) answered at once.
class FakeSystem extends SystemBridge {
  final prefsStore = <String, Object?>{};
  final serviceCalls = <String>[];
  bool micGranted = true;
  bool volumeKeysOn = false;

  @override
  void init() {}

  @override
  Future<Map<String, Object?>> prefs() async => Map.of(prefsStore);

  @override
  Future<void> setPref(String key, Object? value) async => prefsStore[key] = value;

  @override
  Future<Map<String, bool>> permissions(List<String> kinds) async =>
      {for (final k in kinds) k: k != 'microphone' || micGranted};

  @override
  Future<String?> service({required bool listening, required bool glasses, bool mic = false, String text = ''}) async {
    serviceCalls.add('listening=$listening glasses=$glasses');
    return null;
  }

  @override
  Future<bool> termuxInstalled() async => true;

  @override
  Future<String> runTermux(String command) async => 'ok';

  @override
  Future<bool> openTermux() async => true;

  @override
  Future<void> openAppSettings() async {}

  @override
  Future<void> volumeKeys(bool enabled) async => volumeKeysOn = enabled;

  final notifications = StreamController<Map>.broadcast();
  final posted = <String>[];
  bool noteAccess = false;

  @override
  Stream<Map> get phoneNotifications => notifications.stream;

  @override
  Future<void> notify(int id, String title, String text, {String tab = 'work'}) async => posted.add('$title|$text');

  @override
  Future<bool> notificationAccess() async => noteAccess;

  @override
  Future<void> openNotificationAccess() async => noteAccess = true;

  @override
  Future<(double, double)?> location() async => null;

  @override
  Future<String?> launchTab() async => null;

  int? batteryLevel = 82;

  @override
  Future<int?> battery() async => batteryLevel;
}

/// Text-to-speech stand-in: records what was said; finishes each piece at
/// once (or when [finishAll] is called, with [manual]).
class FakeSpeaker implements Speaker {
  final _events = StreamController<SpeakerEvent>.broadcast();
  final said = <String>[];
  final _open = <String>[];
  bool manual = false;
  int stops = 0;
  Map<String, Object> config = {};

  @override
  Stream<SpeakerEvent> get events => _events.stream;

  @override
  Future<void> speak(String text, String id, {bool flush = false}) async {
    said.add(text);
    if (manual) {
      _open.add(id);
    } else {
      scheduleMicrotask(() => _events.add(SpeakerEvent('done', id: id)));
    }
  }

  void finishAll() {
    for (final id in _open) {
      _events.add(SpeakerEvent('done', id: id));
    }
    _open.clear();
  }

  @override
  Future<void> stop() async {
    stops++;
    _open.clear();
  }

  @override
  Future<void> configure({String engine = '', String voice = '', double rate = 1, double pitch = 1}) async =>
      config = {'engine': engine, 'voice': voice, 'rate': rate, 'pitch': pitch};

  @override
  Future<List<TtsVoice>> voices() async =>
      const [TtsVoice('en-us-x-iom-local', 'en-US', 400, false), TtsVoice('en-gb-x-rjs-network', 'en-GB', 500, true)];

  @override
  Future<List<TtsEngineInfo>> engines() async => const [TtsEngineInfo('com.google.android.tts', 'Speech Services by Google')];
}

/// Screen capture stand-in: tests push grayscale frames.
class FakeScreen implements ScreenSource {
  final _frames = StreamController<Map>.broadcast();
  String answer = 'ok';
  final configs = <Map<String, Object>>[];
  int stops = 0;

  @override
  Stream<Map> get frames => _frames.stream;

  @override
  Future<String> start(Map<String, Object> config) async {
    configs.add(config);
    return answer;
  }

  @override
  Future<void> stop() async => stops++;

  @override
  Future<void> configure(Map<String, Object> config) async => configs.add(config);

  /// A frame: [lit] decides which pixels are bright (255) or dark (0).
  void frame(bool Function(int x, int y) lit, {int rows = 136}) {
    final gray = Uint8List(576 * rows);
    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < 576; x++) {
        gray[y * 576 + x] = lit(x, y) ? 255 : 0;
      }
    }
    _frames.add({'gray': gray, 'x0': 0, 'y0': 0, 'x1': 576, 'y1': rows, 'w': 576, 'h': rows, 'ms': 5});
  }

  void same() => _frames.add({'same': true});
  void stopped([String reason = '']) => _frames.add({'stopped': true, 'reason': reason});
}
