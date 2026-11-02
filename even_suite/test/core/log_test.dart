import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:even_suite/core/log.dart';
import 'package:even_suite/core/screen/screen_mirror.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final log = EventLog.instance;

  setUp(() {
    log
      ..clear()
      ..client = null
      ..nativeStats = null
      ..enabled = true;
  });

  tearDown(() => log
    ..stop()
    ..client = null);

  Future<void> settle() async {
    for (var i = 0; i < 30; i++) {
      await Future<void>.delayed(Duration.zero);
    }
  }

  test('sends to the glasses are logged with their timing', () async {
    final glasses = FakeGlasses();
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    final display = GlassesDisplay(Proto(ble));
    await control.connectNow();
    display.showText('Hello from Claude', title: 'Claude');
    await settle();
    final kinds = log.recent.map((e) => '${e.cat}.${e.name}').toSet();
    expect(kinds, containsAll(['glasses.glassesConnected', 'ble.req', 'display.screen']));
    final screen = log.recent.lastWhere((e) => e.name == 'screen');
    expect(screen.data['ok'], isTrue);
    expect(screen.data['status'], '70');
    expect(screen.data['ms'], isA<int>());
    display.dispose();
    ble.dispose();
  });

  test('uploads in one batch with native counters; nothing is lost when the bridge is down', () async {
    final bridge = FakeBridgeClient();
    log
      ..client = bridge
      ..nativeStats = () async => {
            'ble': {'L': {'writes': 3}}
          };
    for (var i = 0; i < 5; i++) {
      EventLog.add('test', 'tick', {'i': i});
    }
    expect(log.pending, 5);
    expect(await log.flush(), isTrue);
    expect(bridge.logged.length, 6, reason: '5 events + native stats');
    expect(bridge.logged.last['c'], 'native');
    expect(log.pending, 0);

    bridge.online = false;
    final down = _DownBridge();
    log.client = down;
    EventLog.add('test', 'offline');
    expect(await log.flush(), isFalse);
    expect(log.pending, greaterThanOrEqualTo(1), reason: 'kept for the next try');
  });

  test('turning the log off records nothing', () {
    log.enabled = false;
    EventLog.add('test', 'ignored');
    expect(log.recent, isEmpty);
  });

  testWidgets('slow glasses: screen capture slows down to the send rate', (tester) async {
    final glasses = FakeGlasses()..replyDelay = const Duration(milliseconds: 700);
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    final screen = FakeScreen();
    final clock = tester.binding.clock;
    final mirror = ScreenMirror(screen, Proto(ble))
      ..autoOff = const Duration(minutes: 5)
      ..now = (() => clock.now());
    await control.connectNow();
    await mirror.start();
    screen.frame((x, y) => x < 100);
    await tester.pump(const Duration(seconds: 3));
    expect(mirror.sent, 1);
    expect(mirror.lastSendMs, greaterThanOrEqualTo(1400), reason: 'end + CRC each wait 700 ms');
    final interval = screen.configs.last['intervalMs'] as int;
    expect(interval, greaterThan(1400), reason: 'no point capturing faster than frames can go out');
    final adapt = log.recent.lastWhere((e) => e.cat == 'screen' && e.name == 'interval');
    expect(adapt.data['ms'], interval);
    await mirror.stop(clearGlasses: false);
    mirror.dispose();
    ble.dispose();
  });
}

class _DownBridge extends FakeBridgeClient {
  @override
  Future<void> log(List<Map<String, Object?>> events, {String session = '', int dropped = 0}) async =>
      throw Exception('Termux bridge is not running');
}
