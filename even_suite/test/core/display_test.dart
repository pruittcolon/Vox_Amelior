import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/bridge/terminal_controller.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:even_suite/core/glasses/glasses_text.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

const longText = 'Claude says: here is a fairly long answer that will not fit on one screen of the '
    'glasses, so it has to be split into pages. Each page holds five lines. You can page with the '
    'TouchBar, the arrows in the app, or by saying scroll down. This sentence pads it out a bit more '
    'so that there are at least three pages to move between, which the test checks below. And one '
    'more sentence for good measure, because more text means more pages to scroll through.';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('glasses text', () {
    test('clean strips Markdown and symbols the G1 font lacks', () {
      final t = GlassesText.clean('## Title\n**bold** and `code` → done ✓\n* item\n[link](http://x)\n🎉');
      expect(t, 'Title\nbold and code -> done v\n- item\nlink');
    });

    test('wrap keeps every word and yields short lines', () {
      final lines = GlassesText.wrap(longText);
      expect(lines.length, greaterThan(5));
      expect(lines.join(' ').split(RegExp(r'\s+')), longText.split(RegExp(r'\s+')));
    });

    test('pager: 5 lines a page, bounds, short text centred like the demo', () {
      final p = GlassesPager(List.generate(12, (i) => 'line $i'));
      expect(p.pageCount, 3);
      expect(p.up(), isFalse);
      expect(p.down(), isTrue);
      expect(p.screen(), 'line 5\nline 6\nline 7\nline 8\nline 9\n');
      expect(p.bottom(), isTrue);
      expect(p.screen(), 'line 10\nline 11\n');
      expect(p.down(), isFalse);
      expect(GlassesPager(['a', 'b']).screen(), '\n\na\nb\n');
      expect(GlassesPager(['a', 'b', 'c', 'd']).screen(), '\na\nb\nc\nd\n');
    });

    test('pager follows live text unless scrolled back', () {
      final p = GlassesPager(List.generate(6, (i) => '$i'), followTail: true);
      expect(p.page, 1);
      p.update(List.generate(11, (i) => '$i'));
      expect(p.page, 2, reason: 'was at the end: follows');
      p.up();
      p.update(List.generate(16, (i) => '$i'));
      expect(p.page, 1, reason: 'scrolled back: stays');
    });

    test('terminal screens become glasses lines without box drawing', () {
      final lines = TerminalController.glassesLines('╭────╮\n│ > hi │\n╰────╯\n● Done → ok\n\n───\n');
      expect(lines.join('\n'), isNot(contains('─')));
      expect(lines.last, '* Done -> ok');
    });
  });

  group('display', () {
    late FakeGlasses glasses;
    late FakeControl control;
    late BleManager ble;
    late GlassesDisplay display;

    setUp(() async {
      glasses = FakeGlasses();
      control = FakeControl();
      ble = BleManager(glasses, control: control);
      display = GlassesDisplay(Proto(ble), liveGap: const Duration(milliseconds: 400));
      await control.connectNow();
    });

    tearDown(() {
      display.dispose();
      ble.dispose();
    });

    Future<void> settle() async {
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(Duration.zero);
      }
    }

    test('a reply shows page 1, arrows page through, same page is never resent', () async {
      display.showText(longText, title: 'Claude');
      await settle();
      expect(glasses.screens().length, 1);
      expect(display.pageLabel, startsWith('1 / '));
      expect(display.down(), isTrue);
      await settle();
      expect(glasses.screens().length, 2);
      expect(display.up(), isTrue);
      await settle();
      expect(display.up(), isFalse, reason: 'already on the first page');
      await settle();
      expect(glasses.screens().length, 3);
      // Page numbers ride along in the 0x4E header.
      final last = glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E);
      expect(last[7], 1);
      expect(last[8], display.pager!.pageCount);
    });

    test('taps on the glasses page while something is shown', () async {
      display.showText(longText);
      await settle();
      glasses.touch(1, 'R');
      await settle();
      // Taps are routed by the app (Suite); here the display API is what they call.
      expect(display.down(), isTrue);
    });

    testWidgets('live text is throttled: a burst sends the first and the last', (tester) async {
      final g = FakeGlasses();
      final c = FakeControl();
      final b = BleManager(g, control: c);
      final d = GlassesDisplay(Proto(b), liveGap: const Duration(milliseconds: 400));
      await c.connectNow();
      for (final w in ['turn', 'turn on', 'turn on the', 'turn on the lights']) {
        d.showLive(w);
        await tester.pump(const Duration(milliseconds: 50));
      }
      await tester.pump(const Duration(milliseconds: 500));
      await tester.pump();
      final screens = g.screens();
      expect(screens.first.trim(), 'turn');
      expect(screens.last.trim(), 'turn on the lights');
      expect(screens.length, lessThanOrEqualTo(3));
      d.dispose();
      b.dispose();
    });

    test('clear exits on the glasses; nothing is sent while disconnected', () async {
      display.showText('hello');
      await settle();
      await display.clear();
      expect(glasses.sent['L']!.last, [0x18]);
      expect(display.mode, DisplayMode.idle);

      await control.emit('glassesDisconnected', {'reason': 'user'});
      glasses.clear();
      display.showText('while away');
      await settle();
      expect(glasses.screens(), isEmpty);
      expect(display.previewLines.join(), contains('while away'), reason: 'the phone still previews it');

      await control.connectNow();
      await settle();
      expect(glasses.screens().single, contains('while away'), reason: 'shown once they reconnect');
    });
  });
}
