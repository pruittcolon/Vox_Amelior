// Renders every screen on a 360x780 phone with real fonts, at normal and
// large (1.3x) text. Any overflow fails the test. PNGs land in
// test/screenshots/out/ (CI uploads them).
import 'dart:io';
import 'dart:ui' as ui;

import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/glasses/dashboard.dart';
import 'package:even_suite/core/suite.dart';
import 'package:even_suite/features/flicker/flicker_page.dart';
import 'package:even_suite/features/tools/tools_page.dart';
import 'package:even_suite/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

final _shotKey = GlobalKey();

Future<void> _loadFonts() async {
  final candidates = <String>[
    if (Platform.environment['FLUTTER_ROOT'] != null)
      '${Platform.environment['FLUTTER_ROOT']}/bin/cache/artifacts/material_fonts',
    // flutter_tester lives in <flutter>/bin/cache/artifacts/engine/<platform>/
    '${File(Platform.resolvedExecutable).parent.parent.parent.path}/material_fonts',
  ];
  final dir = candidates.map(Directory.new).where((d) => d.existsSync()).firstOrNull;
  if (dir == null) return;
  final files = dir.listSync().whereType<File>().toList();
  Future<ByteData> bytes(File f) async => ByteData.sublistView(await f.readAsBytes());
  String name(File f) => f.uri.pathSegments.last.toLowerCase();
  final roboto = FontLoader('Roboto');
  final mono = FontLoader('monospace');
  for (final f in files) {
    final n = name(f);
    if (n.startsWith('roboto-') && !n.contains('italic')) roboto.addFont(bytes(f));
    if (n == 'roboto-regular.ttf') mono.addFont(bytes(f));
  }
  await roboto.load();
  await mono.load();
  for (final f in files.where((f) => name(f).startsWith('materialicons'))) {
    await (FontLoader('MaterialIcons')..addFont(bytes(f))).load();
  }
}

Future<void> _shot(WidgetTester tester, String name) async {
  for (var i = 0; i < 4; i++) {
    await tester.pump(const Duration(milliseconds: 150));
  }
  final boundary = tester.renderObject<RenderRepaintBoundary>(find.byKey(_shotKey));
  await tester.runAsync(() async {
    final image = await boundary.toImage(pixelRatio: 2);
    final png = await image.toByteData(format: ui.ImageByteFormat.png);
    final file = File('test/screenshots/out/$name.png')..createSync(recursive: true);
    file.writeAsBytesSync(png!.buffer.asUint8List());
  });
}

class _World {
  final system = FakeSystem();
  final glasses = FakeGlasses();
  final control = FakeControl();
  final speech = FakeSpeech();
  final bridge = FakeBridgeClient();
  final screen = FakeScreen();
  final speaker = FakeSpeaker();
  late final Suite suite = Suite(
    system: system,
    transport: glasses,
    control: control,
    speech: speech,
    client: bridge,
    screen: screen,
    speaker: speaker,
    weather: WeatherService(location: () async => null),
  )..conversation.gotItTime = Duration.zero; // test time is not wall-clock time
}

const _reply = 'You have three things today:\n'
    '- 10:00 stand-up with the team\n'
    '- 13:30 dentist on Main Street\n'
    '- 18:00 dinner with Ericah\n'
    'Want me to set a reminder for the dentist?';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(_loadFonts);

  for (final scale in [1.0, 1.3]) {
    final tag = scale == 1.0 ? '' : '_large_text';

    testWidgets('every screen fits a 360 dp phone at ${scale}x text', (tester) async {
      tester.view.physicalSize = const Size(1080, 2340);
      tester.view.devicePixelRatio = 3;
      tester.platformDispatcher.textScaleFactorTestValue = scale;
      addTearDown(tester.view.reset);
      addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);

      final w = _World();
      final suite = w.suite;
      await w.bridge.send('What’s on my calendar today?');
      w.bridge.addReply(_reply);
      await w.control.connectNow();
      await suite.chat.check();
      suite.display.showText(_reply, title: 'Claude');

      await tester.pumpWidget(RepaintBoundary(key: _shotKey, child: EvenSuiteApp(suite: suite)));
      await _shot(tester, '1_home$tag');

      // Scroll down to the quick actions.
      await tester.drag(find.byType(ListView).first, const Offset(0, -700));
      await _shot(tester, '2_home_actions$tag');

      // Claude tab, hands-free on, in the middle of a message.
      await tester.tap(find.byIcon(Icons.auto_awesome_outlined));
      await tester.pump();
      await suite.setHandsFree(true);
      await tester.pump();
      w.speech.partial('Claude remind me about the dentist at');
      await tester.pump();
      await _shot(tester, '3_claude_dictating$tag');

      w.speech.say('Claude remind me about the dentist at noon send');
      await tester.pump();
      expect(w.bridge.sentTexts.last, 'remind me about the dentist at noon');
      expect(w.bridge.lastSystem, contains('characters'), reason: 'glasses formatting goes with every message');
      expect(w.bridge.lastOptions, {'model': 'opus', 'effort': 'medium', 'access': 'auto'});
      // Claude works on it: the steps show on the phone and the glasses.
      w.bridge.progress(stage: 'tool', tool: 'Reading calendar.ics');
      await tester.pump();
      await tester.pump(const Duration(seconds: 2));
      expect(w.glasses.screens().last, contains('Reading calendar.ics'));
      await _shot(tester, '4a_claude_working$tag');
      w.bridge.progress(stage: 'writing', live: 'Done. I will remind', block: 1);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 500));
      expect(w.glasses.screens().last, contains('Done. I will remind'));
      w.bridge.finish('Done. I will remind you at 12:00.');
      await tester.pump();
      await tester.pump();
      expect(w.speaker.said.join(' '), contains('I will remind you at 12:00'), reason: 'asked by voice: read aloud');
      await tester.pump();
      expect(suite.voice.followUp, isTrue, reason: 'then your turn: a follow-up without the wake word');
      await _shot(tester, '4_claude_chat$tag');

      // Claude Code finished a job in Termux: Work tab, phone and glasses notes.
      final now = DateTime.now().millisecondsSinceEpoch / 1000;
      suite.work.apply(const WorkSnapshot(1, []));
      suite.work.apply(WorkSnapshot(2, const [], [
        WorkJob(
            session: 'a1',
            project: 'vox_mobile',
            prompt: 'add mood charts to the Insights tab',
            step: 'Running flutter test',
            steps: 14,
            started: now - 260),
      ]));
      expect(suite.dashboard.jobs.single.step, 'Running flutter test', reason: 'live on the look-up screen');
      suite.work.apply(WorkSnapshot(3, [
        WorkItem(
            id: 'w1',
            ts: now,
            project: 'even-suite',
            summary: 'Pushed the fix; CI is green.',
            text: 'Pushed the fix; CI is green. The APK is published.',
            prompt: 'fix the failing screenshot test',
            ms: 134000,
            tools: 6),
      ], [
        WorkJob(
            session: 'a1',
            project: 'vox_mobile',
            prompt: 'add mood charts to the Insights tab',
            step: 'Running flutter test',
            steps: 14,
            started: now - 260),
      ]));
      await tester.pump();
      expect(w.system.posted.single, 'Claude finished · even-suite|Pushed the fix; CI is green.');
      await tester.tap(find.byIcon(Icons.task_alt_outlined));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));
      expect(w.glasses.sent['L']!.any((p) => p[0] == 0x4B), isTrue, reason: 'short note on the glasses');
      await _shot(tester, '4b_work$tag');

      // Terminal tab.
      w.bridge.screen = 'pruitt@phone:~\$ claude\n'
          '╭───────────────────────────────╮\n'
          '│ ✻ Welcome to Claude Code!     │\n'
          '╰───────────────────────────────╯\n'
          '> fix the failing test in vox_mobile\n'
          '● Reading test/search_test.dart…\n'
          '● Found it: the index runs outside runAsync.\n'
          '  ⎿ Updated 1 file\n';
      await tester.tap(find.byIcon(Icons.terminal_outlined));
      await tester.pump();
      await _shot(tester, '5_terminal$tag');

      // Screen tab, sharing, with a frame on the glasses.
      await tester.tap(find.byIcon(Icons.cast_outlined));
      await tester.pump();
      await suite.mirror.start();
      w.screen.frame((x, y) => (y ~/ 12).isOdd && (x ~/ 6) % 7 != 0 && x > 40 && x < 536);
      await tester.pump();
      await tester.runAsync(() => precacheImage(
          MemoryImage(suite.mirror.preview!), tester.element(find.byType(EvenSuiteApp))));
      await _shot(tester, '5b_screen$tag');
      suite.settings.set('mirrorRegion', 'custom');
      await tester.pump();
      await tester.drag(find.byType(ListView).first, const Offset(0, -500));
      await _shot(tester, '5c_screen_custom$tag');
      await suite.mirror.stop();
      await tester.pump();

      // Tools.
      await tester.tap(find.byIcon(Icons.widgets_outlined));
      await tester.pump();
      await _shot(tester, '6_tools$tag');
      final toolsList = find.descendant(of: find.byType(ToolsPage), matching: find.byType(Scrollable)).first;
      Future<void> openTool(String label) async {
        final tile = find.text(label);
        // From the top each time: scrollUntilVisible only scrolls one way.
        await tester.drag(toolsList, const Offset(0, 3000));
        await tester.pump();
        await tester.scrollUntilVisible(tile, 150, scrollable: toolsList);
        await tester.ensureVisible(tile);
        await tester.pump();
        await tester.tap(tile);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      }

      // Flicker (lab), running: text and display height switch on the glasses.
      await openTool('Flicker (lab)');
      await tester.tap(find.text('Start flicker'));
      await tester.pump();
      await _shot(tester, '5d_flicker$tag');
      expect(suite.flicker.running, isTrue);
      final flickerList = find.descendant(of: find.byType(FlickerPage), matching: find.byType(Scrollable)).first;
      String? boxIn(String card) {
        final f = find.descendant(
            of: find.byKey(ValueKey(card), skipOffstage: false),
            matching: find.byType(EditableText),
            skipOffstage: false);
        return f.evaluate().isEmpty ? null : tester.widget<EditableText>(f.first).controller.text.split('\n').first;
      }

      // Each text box keeps its own text when rows above it appear (running).
      await tester.scrollUntilVisible(find.text('Top text'), 150, scrollable: flickerList);
      await tester.pump();
      expect(boxIn('flick-top'), 'TOP');
      await tester.scrollUntilVisible(find.text('Bottom text'), 150, scrollable: flickerList);
      await tester.pump();
      expect(boxIn('flick-bottom'), 'BOTTOM');
      await tester.drag(flickerList, const Offset(0, 3000));
      for (var i = 0; i < 20; i++) {
        await tester.pump(const Duration(milliseconds: 50));
      }
      expect(suite.flicker.steps, greaterThan(2));
      expect(w.glasses.sent['R']!.where((p) => p[0] == 0x26).map((p) => '${p[5]}:${p[6]}').take(2), ['1:8', '1:0'],
          reason: 'display moved high, then low (state 1 = it moves)');
      await tester.tap(find.descendant(of: find.byType(FlickerPage), matching: find.byIcon(Icons.stop_rounded)));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));
      expect(suite.flicker.running, isFalse);
      await tester.drag(find.descendant(of: find.byType(FlickerPage), matching: find.byType(ListView)), const Offset(0, -900));
      await _shot(tester, '5e_flicker_tuning$tag');
      await tester.pageBack();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));

      // Each tool page.
      for (final (label, file) in [
        ('Look up screen', '13_lookup'),
        ('Send text', '7_text'),
        ('Notifications', '8_notifications'),
        ('How to talk to it', '9_help'),
        ('Settings', '10_settings'),
        ('Diagnostics', '12_diagnostics'),
        ('Images', '11_images'),
      ]) {
        await openTool(label);
        await _shot(tester, '$file$tag');
        await tester.pageBack();
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      }

      // Claude and voice: a sheet over the Tools tab.
      await openTool('Claude and voice');
      await tester.pump(const Duration(milliseconds: 400));
      await _shot(tester, '14_claude_settings$tag');
      expect(find.text('Effort (how hard it thinks)'), findsOneWidget);
      await tester.tapAt(const Offset(180, 30));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));

      expect(tester.takeException(), isNull);
      await suite.setHandsFree(false);
      await tester.pumpWidget(const SizedBox());
      suite.dispose();
    }, timeout: const Timeout(Duration(minutes: 3)));
  }

  testWidgets('glasses TouchBar taps page the reply; hold left to talk', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    suite.display.showText(List.generate(30, (i) => 'Line number $i of the answer').join('\n'));
    await tester.pump();
    expect(suite.display.pager!.page, 0);
    w.glasses.touch(1, 'R');
    await tester.pump();
    expect(suite.display.pager!.page, 1);
    w.glasses.touch(1, 'L');
    await tester.pump();
    expect(suite.display.pager!.page, 0);

    // Long press left (Even AI gesture): listen through the glasses' mic.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    expect(suite.voice.holding, isTrue);
    expect(w.speech.lastSource, 'glasses');
    bool micPacket(int on) => w.glasses.sent['R']!.any((p) => p.length == 2 && p[0] == 0x0E && p[1] == on);
    expect(micPacket(1), isTrue, reason: 'glasses microphone on');
    w.speech.partial('what is the weather');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.glasses.screens().join(), isNot(contains('what is the weather')),
        reason: 'the glasses show their own listening screen while held');
    w.glasses.touch(24, 'L');
    await tester.pump();
    expect(micPacket(0), isTrue, reason: 'microphone off on release');
    w.speech.say('what is the weather', finishing: true);
    await tester.pump();
    expect(w.bridge.sentTexts, ['what is the weather']);
    expect(w.bridge.lastSystem, contains('G1'));
    await tester.pump(const Duration(milliseconds: 100));
    expect(w.glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E)[4], 0x31,
        reason: 'thinking screen uses the Even AI state');
    w.bridge.addReply('Sunny and 22 degrees.');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last, contains('Sunny'));
    expect(w.speaker.said.join(' '), contains('Sunny and 22 degrees.'), reason: 'read aloud (asked by voice)');
    await tester.pump();
    expect(suite.voice.followUp, isTrue, reason: 'then it listens for a follow-up');
    expect(w.speech.running, isTrue);
    await tester.pump(const Duration(seconds: 9));
    expect(suite.voice.followUp, isFalse);
    expect(w.speech.running, isFalse, reason: 'mic off again after the follow-up window (hands-free is off)');

    // Double tap: the glasses left our screen.
    w.glasses.touch(0, 'L');
    await tester.pump();
    expect(suite.display.pager, isNull);

    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('volume keys and the notification Stop button are wired', (tester) async {
    final w = _World();
    final suite = w.suite;
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await suite.setHandsFree(true);
    expect(suite.voice.handsFree, isTrue);
    suite.system.onStopListening!();
    await tester.pump();
    expect(suite.voice.handsFree, isFalse);
    expect(w.speech.running, isFalse);

    suite.display.showText(List.generate(12, (i) => 'row $i').join('\n'));
    suite.system.onVolumeKey!('down');
    expect(suite.display.pager!.page, 1);
    suite.system.onVolumeKey!('up');
    expect(suite.display.pager!.page, 0);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
}
