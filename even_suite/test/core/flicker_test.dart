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
      ..homeDepth = 2
      ..clearMs = 0
      ..settleMs = 0;
  }

  final glasses = FakeGlasses();
  final control = FakeControl();
  late final ble = BleManager(glasses, control: control);
  late final Flicker flicker;

  List<List<int>> moves(String lr) => [
        for (final p in glasses.sent[lr]!)
          if (p[0] == 0x26) p,
      ];

  /// 0x26 and 0x4E packets to [lr], in order, as short labels.
  List<String> seq(String lr) => [
        for (final p in glasses.sent[lr]!)
          if (p[0] == 0x26)
            'move${p[5]}:${p[6]}'
          else if (p[0] == 0x4E)
            'text:${String.fromCharCodes(p.sublist(9))}'
          else if (p[0] == 0x18)
            'exit',
      ];

  /// Stop, letting the reset's pauses run.
  Future<void> stop(WidgetTester tester, {bool restore = true}) async {
    final done = flicker.stop(restore: restore);
    await _run(tester, 1200);
    await done;
  }

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

  test('position packet: state 1 moves (height preview), state 0 only closes it', () {
    expect(Flicker(Proto(BleManager(FakeGlasses(), control: FakeControl()))).arms, 'both');
    expect(Proto.positionPacket(0x105, height: 7, depth: 3), [0x26, 0x08, 0x00, 0x05, 0x02, 0x00, 0x07, 0x03]);
    expect(Proto.positionPacket(1, height: 12, depth: 4, preview: true), [0x26, 8, 0, 1, 2, 1, 8, 4]);
  });

  testWidgets('clears, moves with state 1, then redraws the text, on its timer; then puts things back', (tester) async {
    final r = _Rig(tester);
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    expect(r.flicker.steps, inInclusiveRange(9, 11), reason: '0.1 s per spot = 10 switches a second');

    expect(r.seq('R').take(6), ['exit', 'move1:8', 'text:UP', 'exit', 'move1:0', 'text:DOWN'],
        reason: 'the glasses ignore moves while text is up: clear, move (state 1), then the text at the new height');
    expect(r.seq('L').take(6), ['exit', 'move1:8', 'text:UP', 'exit', 'move1:0', 'text:DOWN'],
        reason: 'both arms work out their own height');
    expect(r.moves('R').first, [0x26, 8, 0, 0, 2, 1, 8, 4]);
    expect(r.glasses.screens('L').take(4), ['UP', 'DOWN', 'UP', 'DOWN'], reason: 'text goes to both arms');

    await r.stop(tester);
    expect(r.flicker.running, isFalse);
    expect(r.seq('R').reversed.take(4).toList().reversed, ['exit', 'move1:5', 'move0:5', 'exit'],
        reason: 'normal spot: clear, move there, close the height preview, clear');
    expect(r.moves('R').last.sublist(4), [0x02, 0x00, 5, 2]);

    expect(_events('step').length, inInclusiveRange(9, 11));
    final step = _events('step').first.data;
    expect(step['side'], 'top');
    expect(step['target'], 100);
    expect(step['h'], 8);
    final stop = _events('stop').single.data;
    expect(stop['reason'], 'user');
    expect(stop['rate'] as num, closeTo(10, 1.5));
    expect(_events('start').single.data['method'], 'live');
    expect(_events('reset').single.data['ok'], isTrue);
    r.dispose();
  });

  testWidgets('the Even app way: state 1, wait, state 0, then the text', (tester) async {
    final r = _Rig(tester)
      ..flicker.method = 'apply'
      ..flicker.previewMs = 200
      ..flicker.topMs = 500
      ..flicker.bottomMs = 500;
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    expect(r.seq('R').take(8),
        ['exit', 'move1:8', 'move0:8', 'text:UP', 'exit', 'move1:0', 'move0:0', 'text:DOWN']);
    expect(_events('step').first.data['moveMs'], greaterThanOrEqualTo(200), reason: 'includes the preview time');
    await r.stop(tester);
    r.dispose();
  });

  testWidgets('waits after clearing, and after the move before the text', (tester) async {
    final r = _Rig(tester)
      ..flicker.clearMs = 50
      ..flicker.settleMs = 60
      ..flicker.topMs = 500
      ..flicker.bottomMs = 500;
    await r.control.connectNow();
    await r.flicker.start();
    await tester.pump(const Duration(milliseconds: 10));
    expect(r.seq('R'), ['exit'], reason: 'the move waits for the glasses to go idle');
    await tester.pump(const Duration(milliseconds: 50));
    expect(r.seq('R'), ['exit', 'move1:8'], reason: 'the text waits for the redraw at the new height');
    await tester.pump(const Duration(milliseconds: 60));
    expect(r.seq('R'), ['exit', 'move1:8', 'text:UP']);
    await r.stop(tester, restore: false);
    r.dispose();
  });

  testWidgets('clearing can be turned off', (tester) async {
    final r = _Rig(tester)..flicker.clearFirst = false;
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 500);
    await r.stop(tester, restore: false);
    expect(r.seq('R').take(4), ['move1:8', 'text:UP', 'move1:0', 'text:DOWN']);
    expect(r.seq('R'), isNot(contains('exit')));
    expect(_events('start').single.data['clear'], isFalse);
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
    expect(r.moves('R').map((p) => p[6]), contains(6), reason: 'new top height used on the next switch');
    await r.stop(tester, restore: false);
    r.dispose();
  });

  testWidgets('stops by itself', (tester) async {
    final r = _Rig(tester)..flicker.autoStopSec = 2;
    await r.control.connectNow();
    await r.flicker.start();
    expect(r.flicker.offAt, isNotNull);
    await _run(tester, 2800);
    expect(r.flicker.running, isFalse);
    expect(_events('stop').single.data['reason'], 'auto');
    expect(_events('stats'), isNotEmpty, reason: 'a summary every second');
    r.dispose();
  });

  testWidgets('spot only, text only', (tester) async {
    final a = _Rig(tester)..flicker.what = 'move';
    await a.control.connectNow();
    await a.flicker.start();
    await _run(tester, 500);
    await a.stop(tester, restore: false);
    expect(a.glasses.screens('L'), isEmpty);
    expect(a.moves('R').length, greaterThanOrEqualTo(4));
    expect(a.seq('R').where((p) => p == 'exit').length, 1,
        reason: 'cleared once at the start; with no text the preview just keeps moving');
    a.dispose();

    final b = _Rig(tester)..flicker.what = 'text';
    await b.control.connectNow();
    await b.flicker.start();
    await _run(tester, 500);
    expect(b.seq('R'), isNot(contains('exit')), reason: 'no moves, so nothing to clear for');
    await b.stop(tester);
    expect(b.moves('R'), isEmpty, reason: 'never moved, so nothing to put back');
    expect(b.glasses.screens('L').take(2), ['UP', 'DOWN']);
    b.dispose();
  });

  testWidgets('same text: redrawn after each move, sent once in text-only mode', (tester) async {
    final r = _Rig(tester)..flicker.bottomText = 'UP';
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 500);
    await r.stop(tester, restore: false);
    expect(r.glasses.screens('L').length, r.flicker.steps, reason: 'drawn again at the new height');
    r.dispose();

    final t = _Rig(tester)
      ..flicker.bottomText = 'UP'
      ..flicker.what = 'text';
    await t.control.connectNow();
    await t.flicker.start();
    await _run(tester, 500);
    await t.stop(tester, restore: false);
    expect(t.glasses.screens('L'), ['UP']);
    t.dispose();
  });

  testWidgets('moves to the right arm only when asked', (tester) async {
    final r = _Rig(tester)..flicker.arms = 'R';
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 400);
    await r.stop(tester, restore: false);
    expect(r.moves('L'), isEmpty);
    expect(r.moves('R'), isNotEmpty);
    expect(r.seq('L').take(2), ['exit', 'text:UP'], reason: 'clearing and text still go to both arms');
    r.dispose();
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
    await r.stop(tester);
    expect(_events('reset').single.data['ok'], isFalse);
    r.dispose();
  });

  testWidgets('not waiting for answers', (tester) async {
    final r = _Rig(tester)..flicker.waitForReplies = false;
    await r.control.connectNow();
    await r.flicker.start();
    await _run(tester, 1000);
    await r.stop(tester, restore: false);
    expect(r.flicker.steps, inInclusiveRange(9, 11));
    expect(r.flicker.failures, 0);
    r.dispose();
  });

  testWidgets('one step by hand, logged with the glasses answers', (tester) async {
    final r = _Rig(tester);
    await r.control.connectNow();
    await r.flicker.showOnce(false);
    expect(r.seq('R'), ['exit', 'move1:0', 'text:DOWN']);
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
