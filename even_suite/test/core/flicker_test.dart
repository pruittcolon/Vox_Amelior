import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/glasses/flicker.dart';
import 'package:even_suite/core/log.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

class _Rig {
  _Rig(WidgetTester tester) {
    flicker = Flicker(Proto(ble))
      ..now = (() => tester.binding.clock.now())
      ..topText = 'UP'
      ..bottomText = 'DOWN'
      ..topMs = 100
      ..bottomMs = 100
      ..topHeight = 8
      ..bottomHeight = 0
      ..depth = 4
      ..homeHeight = 5
      ..homeDepth = 2;
  }

  final glasses = FakeGlasses();
  final control = FakeControl();
  late final ble = BleManager(glasses, control: control);
  late final Flicker flicker;

  List<List<int>> moves(String lr) => [
        for (final p in glasses.sent[lr]!)
          if (p[0] == 0x26) p,
      ];

  void dispose() {
    flicker.dispose();
    ble.dispose();
  }
}

Future<void> _run(WidgetTester tester, int ms) async {
  for (var t = 0; t < ms; t += 50) {
    await tester.pump(const Duration(milliseconds: 50));
  }
}

Iterable<LogEvent> _events(String name) => EventLog.instance.recent.where((e) => e.cat == 'flicker' && e.name == name);

void main() {
  setUp(() => EventLog.instance.recent.clear());

  test('position packet is the Even app display setting', () {
    expect(Proto.positionPacket(0x105, height: 7, depth: 3), [0x26, 0x08, 0x00, 0x05, 0x02, 0x00, 0x07, 0x03]);
    expect(Proto.positionPacket(1, height: 12, depth: 4, preview: true), [0x26, 8, 0, 1, 2, 1, 8, 4]);
  });

  testWidgets('switches spot and text on its timer, then puts things back', (tester) async {
    final r = _Rig(tester);
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    expect(r.flicker.steps, inInclusiveRange(9, 11), reason: '0.1 s per spot = 10 switches a second');

    final moves = r.moves('L');
    expect(moves.take(4).map((p) => p[6]), [8, 0, 8, 0], reason: 'high, low, high, low');
    expect(moves.first.sublist(4), [0x02, 0x00, 8, 4], reason: 'display sub-command, keep, height, depth');
    expect(r.moves('R').length, moves.length, reason: 'moves go to both arms');
    expect(r.glasses.screens('L').take(4), ['UP', 'DOWN', 'UP', 'DOWN']);
    expect(r.glasses.screens('R').take(4), ['UP', 'DOWN', 'UP', 'DOWN']);
    final sentL = r.glasses.sent['L']!;
    expect(sentL.indexWhere((p) => p[0] == 0x4E), lessThan(sentL.indexWhere((p) => p[0] == 0x26)),
        reason: 'text first, then the move (default order)');

    await r.flicker.stop();
    await tester.pump();
    expect(r.flicker.running, isFalse);
    expect(r.moves('R').last.sublist(4), [0x02, 0x00, 5, 2], reason: 'normal spot restored');
    expect(r.glasses.sent['R']!.last[0], 0x18, reason: 'then back to the dashboard');

    expect(_events('step').length, inInclusiveRange(9, 11));
    final step = _events('step').first.data;
    expect(step['side'], 'top');
    expect(step['target'], 100);
    expect(step['h'], 8);
    final stop = _events('stop').single.data;
    expect(stop['reason'], 'user');
    expect(stop['rate'] as num, closeTo(10, 1.5));
    expect(_events('start').single.data['topMs'], 100);
    expect(_events('reset').single.data['ok'], isTrue);
    r.dispose();
  });

  testWidgets('different times per spot, and changes apply while running', (tester) async {
    final r = _Rig(tester)
      ..flicker.topMs = 300
      ..flicker.bottomMs = 100;
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 2000);
    expect(r.flicker.steps, inInclusiveRange(9, 11), reason: '0.4 s per round = 5 rounds in 2 s');
    r.flicker.topHeight = 6;
    await _run(tester, 1000);
    expect(r.moves('L').map((p) => p[6]), contains(6), reason: 'new top height used on the next switch');
    await r.flicker.stop();
    r.dispose();
  });

  testWidgets('stops by itself', (tester) async {
    final r = _Rig(tester)..flicker.autoStopSec = 2;
    await r.control.connectNow();
    await r.flicker.start();
    expect(r.flicker.offAt, isNotNull);
    await _run(tester, 2300);
    expect(r.flicker.running, isFalse);
    expect(_events('stop').single.data['reason'], 'auto');
    expect(_events('stats'), isNotEmpty, reason: 'a summary every second');
    r.dispose();
  });

  testWidgets('spot only and text only', (tester) async {
    final a = _Rig(tester)..flicker.what = 'move';
    await a.control.connectNow();
    await a.flicker.start();
    await _run(tester, 500);
    await a.flicker.stop(restore: false);
    expect(a.glasses.screens('L'), isEmpty);
    expect(a.moves('L').length, greaterThanOrEqualTo(4));
    a.dispose();

    final b = _Rig(tester)..flicker.what = 'text';
    await b.control.connectNow();
    await b.flicker.start();
    await _run(tester, 500);
    await b.flicker.stop();
    await tester.pump();
    expect(b.moves('L'), isEmpty, reason: 'never moved, so nothing to put back');
    expect(b.moves('R'), isEmpty);
    expect(b.glasses.screens('L').take(2), ['UP', 'DOWN']);
    b.dispose();
  });

  testWidgets('same text at both spots is sent once', (tester) async {
    final r = _Rig(tester)..flicker.bottomText = 'UP';
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    await r.flicker.stop(restore: false);
    expect(r.glasses.screens('L'), ['UP']);
    expect(r.moves('L').length, greaterThanOrEqualTo(9));
    r.dispose();
  });

  testWidgets('move first, together, right arm only', (tester) async {
    final r = _Rig(tester)
      ..flicker.order = 'move'
      ..flicker.arms = 'R';
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 400);
    await r.flicker.stop(restore: false);
    final sentR = r.glasses.sent['R']!;
    expect(sentR.indexWhere((p) => p[0] == 0x26), lessThan(sentR.indexWhere((p) => p[0] == 0x4E)));
    expect(r.moves('L'), isEmpty, reason: 'the right arm passes moves on itself');
    r.dispose();

    final t = _Rig(tester)..flicker.order = 'together';
    await t.control.connectNow();
    await t.flicker.start();
    await _run(tester, 400);
    await t.flicker.stop(restore: false);
    expect(t.flicker.steps, inInclusiveRange(3, 5));
    expect(t.moves('L').length, t.flicker.steps);
    t.dispose();
  });

  testWidgets('glasses that never answer moves: stops waiting and keeps going', (tester) async {
    final r = _Rig(tester);
    r.glasses.ignoreCmds.add(0x26);
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 2000);
    expect(r.flicker.moveSilent, isTrue);
    expect(_events('silent').single.data['what'], 'move');
    expect(r.flicker.steps, greaterThanOrEqualTo(12), reason: 'only the first 3 moves wait for the timeout');
    expect(_events('step').first.data['moveTimeout'], isTrue);
    final done = r.flicker.stop();
    await _run(tester, 400);
    await done;
    expect(_events('reset').single.data['ok'], isFalse);
    r.dispose();
  });

  testWidgets('not waiting for answers', (tester) async {
    final r = _Rig(tester)..flicker.waitForReplies = false;
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    await r.flicker.stop(restore: false);
    expect(r.flicker.steps, inInclusiveRange(9, 11));
    expect(r.flicker.failures, 0);
    r.dispose();
  });

  testWidgets('one step by hand, logged with the glasses answers', (tester) async {
    final r = _Rig(tester);
    await r.control.connectNow();
    await r.flicker.showOnce(false);
    expect(r.moves('L').single[6], 0);
    expect(r.glasses.screens('L'), ['DOWN']);
    final e = _events('test').single.data;
    expect(e['side'], 'bottom');
    expect(e['moveReply'], 'R:26 c9');
    expect(e['textReply'], 'R:4e c9');
    expect(r.flicker.moveReply, 'R:26 c9');
    r.dispose();
  });

  testWidgets('does nothing without glasses', (tester) async {
    final r = _Rig(tester);
    await r.flicker.start();
    expect(r.flicker.running, isFalse);
    expect(r.flicker.error, contains('Connect'));
    r.dispose();
  });
}
