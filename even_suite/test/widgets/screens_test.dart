// Renders every screen on a 360x780 phone with real fonts, at normal and
// large (1.3x) text. Any overflow fails the test. PNGs land in
// test/screenshots/out/ (CI uploads them).
import 'dart:io';
import 'dart:ui' as ui;

import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/glasses/claude_panel.dart';
import 'package:even_suite/core/glasses/dashboard.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:even_suite/core/glasses/look_layout.dart';
import 'package:even_suite/core/settings.dart';
import 'package:even_suite/core/suite.dart';
import 'package:even_suite/features/flicker/flicker_page.dart';
import 'package:even_suite/features/lookup/look_designer.dart';
import 'package:even_suite/features/screen/screen_page.dart';
import 'package:even_suite/features/tools/tools_page.dart';
import 'package:even_suite/main.dart';
import 'package:even_suite/widgets/glasses_widgets.dart';
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
      expect(w.bridge.lastOptions, {'model': 'sonnet', 'effort': 'low', 'access': 'auto', 'lane': 'quick'},
          reason: 'the quick Claude by default');
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
      // Mid-conversation (the answer is up, it listens for a follow-up): nothing pops up over it;
      // the finished job waits on the look-up screen.
      expect(w.glasses.screens().last.trim(), isNot('*'));
      expect(w.glasses.sent['L']!.any((p) => p[0] == 0x4B), isFalse, reason: 'no popup over the conversation');
      expect(suite.dashboard.lastWork?.id, 'w1');
      await _shot(tester, '4b_work$tag');

      // More: every tool, grouped.
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

      Future<void> back() async {
        await tester.pageBack();
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      }

      // Terminal (from More).
      w.bridge.screen = 'pruitt@phone:~\$ claude\n'
          '╭───────────────────────────────╮\n'
          '│ ✻ Welcome to Claude Code!     │\n'
          '╰───────────────────────────────╯\n'
          '> fix the failing test in vox_mobile\n'
          '● Reading test/search_test.dart…\n'
          '● Found it: the index runs outside runAsync.\n'
          '  ⎿ Updated 1 file\n';
      await openTool('Terminal');
      await _shot(tester, '5_terminal$tag');
      await back();

      // Phone screen, sharing, with a frame on the glasses.
      await openTool('Phone screen');
      await suite.mirror.start();
      w.screen.frame((x, y) => (y ~/ 12).isOdd && (x ~/ 6) % 7 != 0 && x > 40 && x < 536);
      await tester.pump();
      await tester.runAsync(() => precacheImage(
          MemoryImage(suite.mirror.preview!), tester.element(find.byType(EvenSuiteApp))));
      await _shot(tester, '5b_screen$tag');
      suite.settings.set('mirrorRegion', 'custom');
      await tester.pump();
      await tester.drag(
          find.descendant(of: find.byType(ScreenPage), matching: find.byType(ListView)).first, const Offset(0, -500));
      await _shot(tester, '5c_screen_custom$tag');
      await suite.mirror.stop();
      await tester.pump();
      await back();

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
        ('Glasses layout', '15_layout'),
        ('Look up screen', '13_lookup'),
        ('TouchBar', '17_touchbar'),
        ('Speech engine', '19_speech'),
        ('Phone actions', '20_phone'),
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

  testWidgets('look-up screen: Claude on the right, drawn as the glasses lay it out', (tester) async {
    // 576 x 136 glasses pixels, drawn at 2x; each state twice: with the
    // firmware's boxes outlined (to check spacing) and as you would see it.
    tester.view.physicalSize = const Size(1152, 272);
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.reset);
    final now = DateTime(2026, 10, 27, 14, 20);
    double ago(int s) => now.subtract(Duration(seconds: s)).millisecondsSinceEpoch / 1000;
    final states = <String, ClaudePanel>{
      for (final st in ['working', 'answering', 'done']) st: ClaudePanel.example(st, PanelLayout.standard, now: now),
      'custom_layout': ClaudePanel.example('done', PanelLayout.parse('claude,asked:0,meta:1'), now: now),
      'longest': ClaudePanel.build(now: now, jobs: [
        WorkJob(
            session: 'a',
            project: 'a-really-long-project-folder-name-that-goes-on',
            prompt: 'W' * 30 + ' mmmm wwww MMMM ' * 6,
            step: 'Searching the web: ' + 'wide words ' * 8,
            steps: 120,
            recent: ['M' * 40, 'w' * 40, 'Searching the web'],
            started: ago(7300)),
        const WorkJob(session: 'b'),
      ], layout: PanelLayout.standard.copyWith(metaLines: 1)),
    };
    for (final e in states.entries) {
      for (final guides in [true, false]) {
        final key = GlobalKey();
        await tester.pumpWidget(MaterialApp(
          debugShowCheckedModeBanner: false,
          home: RepaintBoundary(
            key: key,
            child: GlassesDashboardView(
              now: now,
              temp: '68°F',
              title: e.value.title,
              rows: e.value.rows,
              guides: guides,
            ),
          ),
        ));
        await tester.pump();
        final boundary = tester.renderObject<RenderRepaintBoundary>(find.byKey(key));
        await tester.runAsync(() async {
          final image = await boundary.toImage(pixelRatio: 2);
          final png = await image.toByteData(format: ui.ImageByteFormat.png);
          File('test/screenshots/out/16_lookup_${e.key}${guides ? '_guides' : ''}.png')
            ..createSync(recursive: true)
            ..writeAsBytesSync(png!.buffer.asUint8List());
        });
      }
    }
    expect(tester.takeException(), isNull);
  });

  testWidgets('my look-up layouts, drawn as the glasses will (G1 font spacing)', (tester) async {
    tester.view.physicalSize = const Size(1152, 272);
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.reset);
    final data = LookData.sample(DateTime(2026, 10, 27, 14, 20));
    for (final e in LookLayout.presets.entries) {
      final key = GlobalKey();
      await tester.pumpWidget(MaterialApp(
        debugShowCheckedModeBanner: false,
        home: RepaintBoundary(
          key: key,
          child: ColoredBox(color: const Color(0xFF050A08), child: LookScreen(layout: e.value, data: data)),
        ),
      ));
      await tester.pump();
      final boundary = tester.renderObject<RenderRepaintBoundary>(find.byKey(key));
      await tester.runAsync(() async {
        final image = await boundary.toImage(pixelRatio: 2);
        final png = await image.toByteData(format: ui.ImageByteFormat.png);
        File('test/screenshots/out/18_look_${e.key}.png')
          ..createSync(recursive: true)
          ..writeAsBytesSync(png!.buffer.asUint8List());
      });
    }
    expect(tester.takeException(), isNull);
  });

  testWidgets('look up: my layout replaces the dashboard, looking down closes it', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    expect(suite.settings.lookMode, 'custom');

    w.glasses.touch(2, 'R'); // head up
    await tester.pump();
    await tester.pump();
    final shown = w.glasses.screens().last;
    expect(shown.split('\n').length, greaterThanOrEqualTo(5), reason: 'all five rows, laid out with spaces');
    expect(shown, contains(LookRenderer.date(DateTime.now(), 'short')), reason: 'the Split layout has the date');
    expect(w.glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E)[4], 0x71, reason: 'plain text over the dashboard');
    await tester.pump(const Duration(seconds: 3));
    expect(suite.display.title, Suite.lookTitle, reason: 'kept up (and fresh) while you look');

    w.glasses.touch(3, 'R'); // head down
    await tester.pump();
    await tester.pump();
    expect(suite.display.mode, DisplayMode.idle);
    expect(w.glasses.sent['L']!.last, [0x18]);
    // The glasses follow a head-down with an F5 00 of their own: not a double tap, so no listening.
    w.glasses.touch(31, 'R');
    w.glasses.touch(0, 'R');
    await tester.pump();
    expect(suite.voice.dictating, isFalse);
    await tester.pump(const Duration(seconds: 1));

    // Not over something you are reading.
    suite.display.showText('An answer you are reading');
    await tester.pump();
    final before = w.glasses.screens().length;
    w.glasses.touch(2, 'R');
    await tester.pump();
    expect(w.glasses.screens().length, before);
    await suite.display.clear();

    // The glasses' own dashboard instead: nothing of ours.
    suite.settings.set('lookMode', 'dashboard');
    final count = w.glasses.screens().length;
    w.glasses.touch(2, 'R');
    await tester.pump();
    expect(w.glasses.screens().length, count);

    // The glasses answer our own exit with a "double tap": that one is not you.
    w.glasses.touch(0, 'R');
    await tester.pump();
    expect(suite.voice.dictating, isFalse, reason: 'an echo, not a double tap');
    await tester.pump(const Duration(seconds: 1));

    // Double tap set to switch chat and terminal.
    suite.settings.set('doubleTapAction', 'switch');
    w.glasses.touch(0, 'R');
    await tester.pump();
    expect(suite.settings.target, Target.terminal);
    w.glasses.touch(0, 'R');
    await tester.pump();
    expect(suite.settings.target, Target.claude);
    await tester.pump(const Duration(seconds: 50));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

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

    // Long press left: "Termux-Claude listening" at once (instead of Even AI's
    // own listening screen), the phone mic takes it down, the words show live.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    expect(suite.voice.pauseSends, isTrue, reason: 'a pause (or the release, when reported) sends it');
    expect(w.speech.lastSource, 'phone');
    // (The test font is wider than the glasses' font, so the header may wrap here.)
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('Termux-Claude listening'));
    expect(w.glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E)[4], 0x71,
        reason: 'plain text: no Even AI fade-in or scroll bar on every update');
    w.speech.partial('what is the weather');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.glasses.screens().last, contains('what is the weather'), reason: 'what you say, live');
    w.glasses.touch(24, 'L');
    await tester.pump();
    w.speech.say('what is the weather', finishing: true);
    await tester.pump();
    expect(w.bridge.sentTexts, ['what is the weather']);
    expect(w.bridge.lastSystem, contains('G1'));
    await tester.pump(const Duration(milliseconds: 100));
    expect(w.glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E)[4], 0x71, reason: 'thinking screen: plain text too');
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

    // Let go before saying anything: never stuck on a listening screen.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    w.glasses.touch(24, 'L');
    await tester.pump();
    w.speech.silence(finishing: true);
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last, contains("Didn't catch that"));
    expect(w.bridge.sentTexts, ['what is the weather'], reason: 'nothing sent');
    await tester.pump(const Duration(seconds: 3));
    expect(suite.display.mode, DisplayMode.idle, reason: 'the glasses are given back');

    // Double tap with nothing on the glasses: talk - no Even AI screen at all.
    await tester.pump(const Duration(seconds: 1)); // past the glasses' answer to that close
    w.glasses.touch(0, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('Termux-Claude listening'));
    // Double tap again: thrown away.
    w.glasses.touch(0, 'L');
    await tester.pump();
    expect(suite.voice.dictating, isFalse);
    await tester.pump(const Duration(seconds: 1));

    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('say "terminal": Claude Code by voice, "start" and "send" on the glasses', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await suite.chat.check(); // Claude Code's hooks are set up
    suite.work.apply(const WorkSnapshot(1, []));
    await tester.pump();
    String last() => w.glasses.screens().last.replaceAll('\n', ' ');

    // Hold and say "terminal" (the pause after it sends it): messages now go to Claude Code in Termux.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    w.speech.say('terminal');
    await tester.pump();
    await tester.pump();
    expect(suite.settings.target, Target.terminal);
    expect(last(), contains('Terminal mode'));
    expect(w.bridge.sentTexts, isEmpty);
    expect(w.bridge.pasted, isEmpty, reason: 'switching is not typed anywhere');
    await tester.pump(const Duration(seconds: 5));
    expect(suite.display.mode, DisplayMode.idle, reason: 'the notice goes away by itself');

    // Hold again: the screen asks for "start", then shows it was heard.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(last(), contains('say "start"'));
    expect(last(), contains('"send"'));
    w.speech.partial('start list the files');
    await tester.pump(const Duration(milliseconds: 700));
    expect(last(), contains('Terminal [start]'));
    expect(last(), contains('list the files'));
    expect(last(), isNot(contains('start list')));
    // "send" types it straight away.
    w.speech.say('start list the files send');
    await tester.pump();
    await tester.pump();
    expect(w.bridge.pasted, ['list the files']);
    expect(last().toLowerCase(), contains('typed'));
    expect(last(), contains('[send]'));
    w.glasses.touch(24, 'L');
    await tester.pump();

    // Claude Code's steps (from its hooks), then its answer, like a chat answer.
    suite.work.apply(const WorkSnapshot(2, [], [
      WorkJob(session: 's1', prompt: 'fix the build', step: 'Editing a.dart', steps: 3),
      WorkJob(session: 's2', prompt: 'list the files', step: 'Running ls', steps: 1),
    ]));
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(last(), contains('Running ls'));
    expect(last(), isNot(contains('Editing a.dart')), reason: 'another session\'s job is not yours');
    expect(last(), contains('Claude Code'));
    suite.work.apply(const WorkSnapshot(3, [
      WorkItem(id: 'j1', ts: 1, prompt: 'list the files', summary: '3 files', text: 'Three files: a, b, c.', ms: 4000),
    ]));
    await tester.pump();
    await tester.pump();
    expect(last(), contains('Three files'));
    expect(w.speaker.said.join(' '), contains('Three files'), reason: 'read aloud, it was spoken');
    expect(w.bridge.sentTexts, isEmpty, reason: 'nothing went to the chat');

    // "chat" switches back.
    await suite.submit('back to chat', spoken: true);
    await tester.pump();
    expect(suite.settings.target, Target.claude);
    expect(w.bridge.sentTexts, isEmpty);
    await tester.pump(const Duration(seconds: 20));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('TouchBar: what you say goes when you pause; tap sends, double tap cancels', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();

    // The glasses never say the TouchBar was let go while our screen is up: a pause sends it.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('pause or tap to send'));
    w.speech.say('what time is it');
    await tester.pump();
    await tester.pump();
    expect(w.bridge.sentTexts, ['what time is it']);
    w.bridge.addReply('Noon.');
    await tester.pump();
    await tester.pump();

    // Tap: sent at once, mid-sentence. Every press starts fresh.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.draft, isEmpty);
    w.speech.partial('turn on the lights');
    await tester.pump();
    w.glasses.touch(1, 'R');
    await tester.pump();
    await tester.pump();
    expect(w.bridge.sentTexts.last, 'turn on the lights');
    w.bridge.addReply('Done.');
    await tester.pump();
    await tester.pump();

    // Double tap: thrown away, now and later.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    w.speech.partial('never mind');
    await tester.pump();
    w.glasses.touch(0, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isFalse);
    await tester.pump(const Duration(seconds: 10));
    expect(w.bridge.sentTexts.length, 2);

    // Nothing said: "Didn't catch that", then the glasses are given back.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    await tester.pump(const Duration(seconds: 7));
    await tester.pump();
    expect(w.glasses.screens().last, contains("Didn't catch that"));
    await tester.pump(const Duration(seconds: 3));
    expect(suite.display.mode, DisplayMode.idle);
    expect(w.bridge.sentTexts.length, 2);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('phone actions: instant ones, Claude\'s, and a reminder going off', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();

    // Said plainly: done at once, Claude is not asked.
    await suite.submit('Flashlight on', spoken: true);
    await tester.pump();
    expect(w.system.phoneCalls.single, {'do': 'torch', 'on': true});
    expect(w.bridge.sentTexts, isEmpty);
    expect(w.glasses.screens().last, contains('Done'));
    await tester.pump(const Duration(seconds: 4));

    // Through Claude: its action line is done and never shown.
    await suite.submit('remind me tomorrow at 9 to call the bank', spoken: false);
    await tester.pump();
    expect(w.bridge.lastSystem, contains('[[phone:'), reason: 'Claude knows it can');
    w.bridge.addReply('Reminder set for 9 AM.\n[[phone:{"do":"remind","at":"2030-01-02T09:00","text":"Call the bank"}]]');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last, contains('Reminder set for 9 AM.'));
    expect(w.glasses.screens().join(), isNot(contains('[[phone')));
    final remind = w.system.phoneCalls.last;
    expect(remind['do'], 'remind');
    expect(remind['text'], 'Call the bank');
    expect(remind['at'], DateTime(2030, 1, 2, 9).millisecondsSinceEpoch);
    await tester.pump(const Duration(seconds: 30));

    // The reminder goes off: on the glasses.
    await suite.display.clear();
    await tester.pump(const Duration(seconds: 1));
    suite.system.onReminder!('r1', 'Call the bank');
    await tester.pump();
    expect(w.glasses.screens().last, contains('Call the bank'));
    expect(w.glasses.screens().last, contains('Reminder'));
    await tester.pump(const Duration(seconds: 30));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('quick answers: a light Claude first, Claude Code when it needs tools', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();

    await suite.submit('what is in my notes folder', spoken: false);
    await tester.pump();
    expect(w.bridge.lastOptions?['lane'], 'quick');
    expect(w.bridge.lastSystem, contains('[[full]]'));
    // It needs tools: handed to Claude Code by itself, the same question.
    w.bridge.addReply('[[full]]');
    await tester.pump();
    await tester.pump();
    expect(w.bridge.sentTexts, ['what is in my notes folder', 'what is in my notes folder']);
    expect(w.bridge.lastOptions?['lane'], 'full');
    expect(w.bridge.lastOptions?['model'], 'opus');
    expect(w.glasses.screens().join(), isNot(contains('[[full]]')));
    expect(suite.conversation.busy, isTrue, reason: 'still waiting, now for Claude Code');
    w.bridge.addReply('Three files: a.md, b.md, c.md');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last, contains('Three files'));

    // "Think harder": straight to Claude Code.
    await tester.pump(const Duration(seconds: 30));
    await suite.submit('think harder about the plan', spoken: false);
    await tester.pump();
    expect(w.bridge.lastOptions?['lane'], 'full');
    w.bridge.addReply('Done.');
    await tester.pump();
    await tester.pump(const Duration(seconds: 30));

    // TouchBar: the words stopped changing for 0.9 s - sent without waiting for the recognizer.
    await tester.pump(const Duration(seconds: 1));
    w.glasses.touch(0, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.pauseSends, isTrue);
    w.speech.partial('what day is it');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.bridge.sentTexts.last, isNot('what day is it'));
    await tester.pump(const Duration(milliseconds: 500));
    await tester.pump();
    expect(w.bridge.sentTexts.last, 'what day is it');
    w.bridge.addReply('Tuesday.');
    await tester.pump(const Duration(seconds: 30));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('a tiny "*" for 3 seconds when Claude is done (can be switched off)', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    suite.work.apply(const WorkSnapshot(1, []));
    await tester.pump();

    const a = WorkItem(id: 'a', ts: 1, prompt: 'fix the bug', summary: 'Fixed it', ms: 60000);
    suite.work.apply(const WorkSnapshot(2, [a]));
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last.trim(), '*');
    expect(w.glasses.sent['L']!.where((p) => p[0] == 0x4B), isEmpty, reason: 'no popup on top of the *');
    expect(w.system.posted.join(), contains('Claude finished'), reason: 'the phone still says so');
    await tester.pump(const Duration(milliseconds: 3100));
    await tester.pump();
    expect(suite.display.mode, DisplayMode.idle);
    expect(w.glasses.sent['L']!.lastWhere((p) => p[0] == 0x4E || p[0] == 0x18), [0x18],
        reason: 'gone after 3 seconds');

    // Sent away with a double tap while Claude works: no screens come back, the answer is a "*".
    await tester.pump(const Duration(seconds: 1)); // past the glasses' answer to that exit
    await suite.submit('what time is it', spoken: false);
    await tester.pump();
    expect(suite.conversation.busy, isTrue);
    w.glasses.touch(0, 'L');
    await tester.pump();
    final shown = w.glasses.screens().length;
    await tester.pump(const Duration(seconds: 3));
    expect(w.glasses.screens().length, shown, reason: 'the working screen stays away');
    w.bridge.addReply('It is noon.');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last.trim(), '*');
    await tester.pump(const Duration(seconds: 4));

    // Switched off: nothing.
    suite.settings.set('doneBlink', false);
    final before = w.glasses.screens().length;
    suite.work.apply(const WorkSnapshot(3, [a, WorkItem(id: 'b', ts: 2, summary: 'Done again', ms: 1000)]));
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().length, before);
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
