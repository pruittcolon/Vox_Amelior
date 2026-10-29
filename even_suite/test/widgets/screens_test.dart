// Renders every screen on a 360x780 phone with real fonts, at normal and
// large (1.3x) text. Any overflow fails the test. PNGs land in
// test/screenshots/out/ (CI uploads them).
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/glasses/claude_panel.dart';
import 'package:even_suite/core/glasses/dashboard.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:even_suite/core/glasses/hud_frame.dart';
import 'package:even_suite/core/glasses/look_layout.dart';
import 'package:even_suite/core/home/vizio.dart';
import 'package:even_suite/core/home/voice_monkey.dart';
import 'package:even_suite/core/settings.dart';
import 'package:even_suite/core/suite.dart';
import 'package:even_suite/core/voice/voice_commands.dart';
import 'package:even_suite/features/claude/claude_page.dart';
import 'package:even_suite/features/home/home_page.dart';
import 'package:even_suite/features/lookup/look_designer.dart';
import 'package:even_suite/features/lookup/lookup_page.dart';
import 'package:even_suite/features/screen/screen_page.dart';
import 'package:even_suite/features/settings/settings_page.dart';
import 'package:even_suite/features/tools/tools_page.dart';
import 'package:even_suite/main.dart';
import 'package:even_suite/widgets/glasses_widgets.dart';
import 'package:even_suite/widgets/terminal_view.dart';
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
  )
    ..conversation.gotItTime = Duration.zero // test time is not wall-clock time
    // Claude Code's answers in terminal mode like chat ones (the tests below
    // that check its steps); the "Termux on the glasses" test turns it on.
    ..settings.set('termShowSec', 0)
    ..settings.set('lookFrame', 'off'); // the HUD frame test turns it on
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
      suite.settings.set('talkMode', 'chat'); // back and forth: the "your turn" screen
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

      // Terminal sessions (from More): the Claude tab on Claude Code, live and in colour.
      w.bridge.screen = '\x1b[38;2;215;119;87m ▐▛███▜▌\x1b[0m   Claude Code v2.1\n'
          '\x1b[2m   /root/even-suite\x1b[0m\n'
          '\n'
          '\x1b[1m❯ fix the failing test in vox_mobile\x1b[0m\n'
          '\x1b[38;2;215;119;87m●\x1b[0m Reading test/search_test.dart…\n'
          '\x1b[38;2;215;119;87m●\x1b[0m Found it: the index runs outside runAsync.\n'
          '  ⎿ \x1b[32mUpdated 1 file\x1b[0m\n'
          '\x1b[38;5;244m────────────────────────────────────────\x1b[0m\n'
          '❯ \n'
          '\x1b[38;5;244m────────────────────────────────────────\x1b[0m\n'
          '  \x1b[38;5;141m⏵⏵ auto mode on\x1b[0m (shift+tab to cycle)\n';
      await openTool('Terminal sessions');
      await tester.pump(const Duration(milliseconds: 400));
      expect(suite.settings.target, Target.terminal);
      expect(find.byType(TerminalScreen), findsOneWidget);
      await tester.tap(find.byTooltip('Esc, arrows, Tab ...'));
      await tester.pump();
      expect(find.text('Esc'), findsOneWidget, reason: 'the keys a phone keyboard lacks');
      await _shot(tester, '5_terminal$tag');
      suite.useChat();
      await tester.tap(find.byIcon(Icons.widgets_outlined));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));

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

      // Each tool page.
      for (final (label, file) in [
        ('Glasses layout', '15_layout'),
        ('Look up screen', '13_lookup'),
        ('TouchBar', '17_touchbar'),
        ('Speech engine', '19_speech'),
        ('Phone actions', '20_phone'),
        ('Text height', '21_text_height'),
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
      expect(find.text('Answer speed'), findsOneWidget);
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
      // And with each HUD outline over it, as the glasses get it.
      for (final (style, _) in HudFrame.styles.where((s) => s.$1 != 'off')) {
        final hudKey = GlobalKey();
        await tester.pumpWidget(MaterialApp(
          debugShowCheckedModeBanner: false,
          home: RepaintBoundary(
            key: hudKey,
            child: ColoredBox(
                color: const Color(0xFF050A08), child: LookScreen(layout: e.value, data: data, frame: style)),
          ),
        ));
        await tester.pump();
        final b = tester.renderObject<RenderRepaintBoundary>(find.byKey(hudKey));
        await tester.runAsync(() async {
          final image = await b.toImage(pixelRatio: 2);
          final png = await image.toByteData(format: ui.ImageByteFormat.png);
          File('test/screenshots/out/18_look_${e.key}_$style.png')
            ..createSync(recursive: true)
            ..writeAsBytesSync(png!.buffer.asUint8List());
        });
      }
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
    suite.settings.set('talkMode', 'chat'); // back and forth
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

    // Long press left: the glasses' own "Even AI listening" stays - nothing of
    // ours over it, so they say when it is let go - and letting go sends it.
    final screensBefore = w.glasses.screens().length;
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    expect(suite.voice.pauseSends, isTrue, reason: 'a pause sends it too');
    expect(w.speech.lastSource, 'phone', reason: 'Android\'s recognizer here: the phone listens');
    expect(w.glasses.screens(), hasLength(screensBefore), reason: 'their Even AI screen stays');
    w.speech.partial('what is the weather');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.glasses.screens(), hasLength(screensBefore), reason: 'none while you talk either');
    w.glasses.touch(24, 'L');
    await tester.pump();
    expect(w.speech.finishes, greaterThan(0), reason: 'let go: the recognizer finishes the last words');
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
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('Listening'));
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
    expect(last(), contains('> Termux Claude'));
    expect(w.bridge.sentTexts, isEmpty);
    expect(w.bridge.pasted, isEmpty, reason: 'switching is not typed anywhere');
    await tester.pump(const Duration(seconds: 5));
    expect(suite.display.mode, DisplayMode.idle, reason: 'the notice goes away by itself');

    // Hold again: the glasses' own Even AI screen while you talk (nothing of ours).
    final held = w.glasses.screens().length;
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    w.speech.partial('start list the files');
    await tester.pump(const Duration(milliseconds: 700));
    expect(w.glasses.screens(), hasLength(held));
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
    expect(last(), contains('Termux Claude'));
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

    // Held, and a pause (not letting go) sends it. Their own Even AI screen stays meanwhile.
    final before = w.glasses.screens().length;
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens(), hasLength(before));
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
    w.speech.partial('what is on my calendar');
    await tester.pump();
    w.glasses.touch(1, 'R');
    await tester.pump();
    await tester.pump();
    expect(w.bridge.sentTexts.last, 'what is on my calendar');
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

  testWidgets('Antigravity answers when Claude has run out: say it, ask, same glasses, and back', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();

    // By voice: "use agy" is a command, not a question for anyone.
    await suite.submit('use agy', spoken: false);
    await tester.pump();
    expect(suite.settings.agent, 'agy');
    expect(w.bridge.sentTexts, isEmpty);
    expect(w.bridge.lastWarmTo, 'agy', reason: 'Antigravity starts before the question');
    await tester.pump(const Duration(seconds: 30));

    await suite.submit('what is in my notes folder', spoken: false);
    await tester.pump();
    expect(w.bridge.sentTo.last, 'agy');
    expect(w.bridge.lastOptions, {'effort': 'medium', 'access': 'auto', 'lane': 'full'},
        reason: 'no quick lane, no Claude model, and no commands unless allowed');
    expect(w.bridge.lastSystem, contains('Google Antigravity'));
    expect(w.bridge.lastSystem, isNot(contains('[[full]]')));
    w.bridge.progress(stage: 'tool', tool: 'Running ls');
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(w.glasses.screens().last, contains('Running ls'));
    w.bridge.finish('Three files: a.md, b.md, c.md');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last, contains('Three files'));
    await tester.pump(const Duration(seconds: 30));

    // Running commands is its own yes.
    suite.settings.set('agyCommands', true);
    await suite.submit('list my files', spoken: false);
    await tester.pump();
    expect(w.bridge.lastOptions?['agy_commands'], 'true');
    w.bridge.addReply('ok');
    await tester.pump();
    await tester.pump(const Duration(seconds: 30));

    // And back to Claude: the quick lane and the Claude model return.
    await suite.submit('use claude', spoken: false);
    await tester.pump();
    expect(suite.settings.agent, 'claude');
    await tester.pump(const Duration(seconds: 30));
    await suite.submit('how many moons does jupiter have', spoken: false);
    await tester.pump();
    expect(w.bridge.sentTo.last, 'claude');
    expect(w.bridge.lastOptions?['lane'], 'quick');
    expect(w.bridge.lastOptions?.containsKey('agy_commands'), isFalse);
    w.bridge.addReply('Eight pm.');
    await tester.pump();
    await tester.pump(const Duration(seconds: 30));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  });

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

  testWidgets('sessions: open another Claude Code, talk and type to it, the hold follows, end it', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    // Left over from testing: the hold fixed to chat.
    suite.settings.set('holdAction', 'chat');
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await suite.chat.check();
    suite.work.apply(const WorkSnapshot(1, []));
    await tester.pump();
    await tester.tap(find.byIcon(Icons.auto_awesome_outlined));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));
    Finder inClaude(Finder f) => find.descendant(of: find.byType(ClaudePage), matching: f);
    expect(inClaude(find.text('Chat')), findsOneWidget);
    expect(inClaude(find.text('Termux Claude')), findsOneWidget, reason: 'the one in Termux');
    expect(inClaude(find.text('Shell')), findsNothing, reason: 'shells are under Sessions');

    // Sessions: everything open (the shell too), past conversations, and New.
    await tester.tap(inClaude(find.text('Sessions')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(find.text('A command line'), findsOneWidget, reason: 'the shell is listed here');
    expect(find.text('fix the look up screen on the glasses'), findsOneWidget, reason: 'a past conversation');
    await tester.tap(find.text('New Claude Code'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(w.bridge.sessionList.map((x) => x.name), contains('claude-2'));
    expect(suite.settings.target, Target.terminal);
    expect(suite.terminal.session, 'claude-2');
    expect(suite.settings.holdAction, 'mode', reason: 'a hold fixed to chat would go somewhere else');
    expect(inClaude(find.text('Claude 2')), findsOneWidget);
    expect(find.byType(TerminalScreen), findsOneWidget);

    // Typed: word for word ("start" stays), into that session.
    await tester.enterText(inClaude(find.byType(TextField)), 'start the dev server');
    // The keyboard's send key (the "Opened Claude 2" toast covers the button here).
    await tester.testTextInput.receiveAction(TextInputAction.send);
    await tester.pump();
    await tester.pump();
    expect(w.bridge.pasted.last, 'start the dev server');
    expect(w.bridge.pastedTo.last, 'claude-2');

    // Spoken with the TouchBar: goes to the same one, a pause sends it.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    w.speech.say('list the files');
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(w.bridge.pasted.last, 'list the files');
    expect(w.bridge.pastedTo.last, 'claude-2');
    w.glasses.touch(24, 'L');
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));

    // Claude Code is asking a question: nothing typed, the glasses say so.
    w.bridge.pasteResult = 'menu';
    await suite.typeIntoTerminal('yes do it');
    await tester.pump();
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('asking you something'));
    w.bridge.pasteResult = 'sent';
    await tester.pump(const Duration(seconds: 7));

    // "next session" by voice: back to the Termux one.
    suite.runCommand(VoiceCommand.nextSession);
    await tester.pump();
    await tester.pump();
    expect(suite.terminal.session, isNot('claude-2'));
    await tester.pump(const Duration(seconds: 5));

    // Hold the chip: end it.
    suite.useSession('claude-2');
    await tester.pump();
    await tester.longPress(inClaude(find.text('Claude 2')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    await tester.tap(find.text('End this session'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    await tester.tap(find.text('End it'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(w.bridge.sessionList.map((x) => x.name), isNot(contains('claude-2')));
    await tester.pump(const Duration(seconds: 1));
    expect(suite.terminal.session, 'claude', reason: 'back to the Termux one');

    // A past conversation, picked up in a new session.
    await tester.tap(inClaude(find.text('Sessions')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    await tester.tap(find.text('fix the look up screen on the glasses'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(w.bridge.resumed.last, '0f1e2d3c-4b5a-4968-8776-655443322110');
    expect(suite.terminal.session, 'claude-2');

    // Chat again.
    await tester.tap(inClaude(find.text('Chat')));
    await tester.pump();
    expect(suite.settings.target, Target.claude);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('glasses battery shows in the app', (tester) async {
    tester.view.physicalSize = const Size(1080, 2340);
    tester.view.devicePixelRatio = 3;
    addTearDown(tester.view.reset);
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(RepaintBoundary(key: _shotKey, child: EvenSuiteApp(suite: suite)));
    await tester.pump();
    Finder onHome(String text) => find.descendant(of: find.byType(HomePage), matching: find.text(text));
    Finder inClaude(String text) => find.descendant(of: find.byType(ClaudePage), matching: find.text(text));
    // Connected, not told yet: still there, so you know where to look.
    expect(onHome('Glasses …'), findsOneWidget);

    await tester.pump(const Duration(seconds: 4)); // asked a moment after connecting
    expect(suite.ble.batteryL, 73);
    expect(suite.ble.batteryR, 77);
    expect(suite.ble.battery, 73, reason: 'the arm that runs out first');
    expect(onHome('Glasses 73%'), findsOneWidget);
    await _shot(tester, '22_battery_home');
    // And on the Claude tab, whatever it talks to.
    await tester.tap(find.byIcon(Icons.auto_awesome_outlined));
    await tester.pump();
    expect(inClaude('Glasses 73%'), findsOneWidget);
    await tester.tap(find.byIcon(Icons.visibility_outlined));
    await tester.pump();

    // Told by another app's question too.
    w.glasses.injectReply('L', Uint8List.fromList([0x2C, 0x66, 41, 0, 0]));
    await tester.pump();
    expect(suite.ble.battery, 41);
    expect(onHome('Glasses 41%'), findsOneWidget);

    // Tap it: both arms, and read again.
    await tester.tap(onHome('Glasses 41%'));
    for (var i = 0; i < 4; i++) {
      await tester.pump(const Duration(milliseconds: 200)); // the sheet slides up
    }
    expect(find.text('Battery: left 41%, right 77%'), findsOneWidget);
    w.glasses.battery['L'] = 15;
    await tester.tap(find.text('Check'));
    for (var i = 0; i < 6; i++) {
      await tester.pump(const Duration(milliseconds: 300));
    }
    expect(find.text('Battery: left 15%, right 77%'), findsOneWidget);
    await _shot(tester, '22_battery_sheet');
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('built-in look-up screen: put back when the Even app rewrites the dashboard', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('lookMode', 'dashboard');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump(const Duration(seconds: 10));
    int dashWrites() => w.glasses.sent['R']!.where((p) => p[0] == 0x06).length;
    final before = dashWrites();
    // The Even app's minute update: its acks (its own sequence numbers) reach us too.
    for (final p in [
      [0x06, 0x07, 0x00, 0x2c, 0x06, 0x00],
      [0x06, 0x16, 0x00, 0x2d, 0x01, 0x00],
      [0x06, 0x0c, 0x00, 0x2e, 0x03, 0x01, 0x00, 0x01, 0x00, 0x00],
    ]) {
      w.glasses.injectReply('R', Uint8List.fromList(p));
    }
    await tester.pump(const Duration(seconds: 1));
    await tester.pump(const Duration(seconds: 2));
    expect(dashWrites(), greaterThan(before), reason: 'ours goes back right after');
    expect(suite.dashboard.foreignWrites, greaterThan(0));
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    final nav = tester.state<NavigatorState>(find.byType(Navigator).first);
    nav.push(MaterialPageRoute<void>(builder: (_) => const LookUpPage()));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(find.text('Another app changes your dashboard'), findsOneWidget);
    expect(find.text('Stop the Even app'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('my look-up layout comes back when the glasses put their dashboard over it', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('lookMode', 'custom');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump(const Duration(seconds: 1));
    w.glasses.touch(2, 'R'); // head up
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));
    final shown = w.glasses.screens().length;
    expect(suite.display.title, Suite.lookTitle);
    w.glasses.touch(30, 'R'); // their dashboard came up over ours
    await tester.pump(const Duration(milliseconds: 300));
    expect(w.glasses.screens().length, greaterThan(shown), reason: 'the same screen, sent again');
    w.glasses.touch(3, 'R'); // head down
    await tester.pump(const Duration(seconds: 1));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('"flashlight on" works in terminal mode too - never typed into the terminal', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    suite.useSession('claude');
    await tester.pump();
    await suite.submit('Flashlight on', spoken: true);
    await tester.pump();
    expect(w.system.phoneCalls.single['do'], 'torch');
    expect(w.bridge.pasted, isEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('Termux on the glasses: its screen stays for the time you set, then goes', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('termShowSec', 10);
    suite.applySettings();
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await suite.chat.check();
    suite.useSession('claude');
    await tester.pump();
    w.bridge.screen = '> ready\n';
    await suite.typeIntoTerminal('list the files');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.bridge.pasted.last, 'list the files');
    expect(suite.terminal.mirroring, isTrue, reason: 'the terminal itself, not a chat-like answer');
    expect(w.glasses.screens().last, contains('list the files'));

    // Still there before the time is up, gone after it.
    await tester.pump(const Duration(seconds: 8));
    expect(suite.terminal.mirroring, isTrue);
    await tester.pump(const Duration(seconds: 3));
    expect(suite.terminal.mirroring, isFalse);

    // Until you close it: still there much later.
    suite.settings.set('termShowSec', -1);
    suite.applySettings();
    await suite.typeIntoTerminal('and again');
    await tester.pump(const Duration(minutes: 3));
    expect(suite.terminal.mirroring, isTrue);
    suite.terminal.setMirror(false);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('look up: the HUD frame picture goes over the text, once it is up', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('lookFrame', 'hud');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    final before = w.glasses.sent['L']!.length;
    w.glasses.touch(2, 'R'); // head up
    for (var i = 0; i < 10; i++) {
      await tester.pump(const Duration(milliseconds: 300));
    }
    final after = w.glasses.sent['L']!.sublist(before);
    final text = after.indexWhere((p) => p[0] == 0x4E);
    final picture = after.indexWhere((p) => p[0] == 0x15);
    expect(text, greaterThanOrEqualTo(0), reason: 'the layout text at once');
    expect(picture, greaterThan(text), reason: 'then the outline picture (it replaces the screen)');
    expect(after.indexWhere((p) => p[0] == 0x4E, picture), greaterThan(picture), reason: 'then the text again, over it');
    // Unchanged text: the frame is not sent again every refresh (once the
    // phone's battery, read just after the first draw, has filled in).
    await tester.pump(const Duration(seconds: 5));
    int frames() => w.glasses.sent['L']!.sublist(before).where((p) => p[0] == 0x15 && p[1] == 0).length;
    final settled = frames();
    await tester.pump(const Duration(seconds: 6));
    expect(frames(), settled);
    w.glasses.touch(3, 'R');
    await tester.pump(const Duration(seconds: 1));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('built-in look-up dark on the left: the startup mode goes back to 0, both screens reset', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('lookMode', 'dashboard');
    w.glasses.startupMode
      ..['L'] = 1
      ..['R'] = 1;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump(const Duration(seconds: 7)); // read once after connecting
    expect(w.glasses.sent['R']!.where((p) => p[0] == 0x08 && p[4] == 0x04), isNotEmpty);
    final dashBefore = w.glasses.sent['L']!.where((p) => p[0] == 0x06).length;

    // You look up: only the right arm opens its dashboard.
    w.glasses.touch(2, 'R');
    w.glasses.touch(30, 'R');
    for (var i = 0; i < 10; i++) {
      await tester.pump(const Duration(milliseconds: 300));
    }
    final set = w.glasses.sent['R']!.where((p) => p[0] == 0x08 && p[4] == 0x03).toList();
    expect(set, hasLength(1), reason: 'set once, on the right arm (it passes it on)');
    expect(set.single[5], 0);
    expect(w.glasses.startupMode, {'L': 0, 'R': 0});
    expect(w.glasses.sent['L']!.where((p) => p[0] == 0x18), isNotEmpty, reason: 'both screens back to idle');
    expect(w.glasses.sent['L']!.where((p) => p[0] == 0x06).length, greaterThan(dashBefore), reason: 'dashboard sent again');

    // Not again on the next look-up (once per connection).
    w.glasses.touch(30, 'R');
    await tester.pump(const Duration(seconds: 2));
    expect(w.glasses.sent['R']!.where((p) => p[0] == 0x08 && p[4] == 0x03), hasLength(1));
    expect(await suite.fixLeftEye(), contains('Startup mode 0 (left) / 0 (right)'));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('one question (standard): no listening after the answer, and it goes away by itself', (tester) async {
    final w = _World();
    final suite = w.suite;
    expect(suite.settings.talkMode, 'single');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    w.glasses.touch(23, 'L'); // hold left: talk
    await tester.pump();
    await tester.pump();
    w.speech.partial('how many moons does jupiter have');
    await tester.pump(const Duration(milliseconds: 500));
    w.glasses.touch(24, 'L');
    await tester.pump();
    w.speech.say('how many moons does jupiter have', finishing: true);
    await tester.pump();
    expect(w.bridge.sentTexts, ['how many moons does jupiter have']);
    await tester.pump(const Duration(milliseconds: 100));
    w.bridge.addReply('Jupiter has 95 known moons. Want to know about the biggest ones?');
    await tester.pump();
    await tester.pump();
    expect(w.glasses.screens().last.replaceAll('\n', ' '), contains('95 known moons'));
    await tester.pump(const Duration(seconds: 2));
    expect(suite.voice.followUp, isFalse, reason: 'one question: nothing listens afterwards');
    expect(suite.voice.dictating, isFalse);
    // The answer asks something back, but nobody is listening for it: it
    // stays for its reading time, then the glasses get their screen back.
    int exits() => w.glasses.sent['L']!.where((p) => p.length == 1 && p[0] == 0x18).length;
    final before = exits();
    await tester.pump(const Duration(seconds: 40));
    expect(suite.display.mode, DisplayMode.idle);
    expect(exits(), greaterThan(before), reason: 'the glasses get their screen back');
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('back and forth: follow-ups, but never on and on by itself', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.settings.set('talkMode', 'chat');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    w.speech.partial('what is on tv');
    await tester.pump(const Duration(milliseconds: 500));
    w.glasses.touch(24, 'L');
    await tester.pump();
    w.speech.say('what is on tv', finishing: true);
    await tester.pump();
    // A TV in the room keeps "answering": each one sent as a follow-up...
    var turns = 0;
    for (var i = 0; i < 10; i++) {
      w.bridge.addReply('Answer $i. Anything else?');
      await tester.pump();
      await tester.pump();
      await tester.pump(const Duration(seconds: 1));
      if (!suite.voice.followUp) break;
      turns++;
      w.speech.say('the tv says something $i', finishing: true);
      await tester.pump();
      await tester.pump(const Duration(seconds: 2)); // a pause sends it
    }
    // ...until the limit: then it stops listening.
    expect(turns, Suite.maxFollowUps);
    expect(suite.voice.dictating, isFalse);
    await tester.pump(const Duration(seconds: 40));
    expect(suite.display.mode, DisplayMode.idle);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('hands-free, no wake word: a timer, the lights, the TV; "TV remote", then just "up"', (tester) async {
    final w = _World();
    final suite = w.suite;
    final lights = <String>[];
    final tvKeys = <String>[];
    suite.voiceMonkey = VoiceMonkey(get: (u) async {
      lights.add(u.queryParameters['device']!);
      return (200, '{"success":true}');
    });
    suite.tv = VizioTv(call: (method, url, headers, body) async {
      if (url.path == '/key_command/') {
        final k = (((body as Map)['KEYLIST'] as List).first as Map);
        tvKeys.add('${k['CODESET']}/${k['CODE']}');
      }
      return {
        'STATUS': {'RESULT': 'SUCCESS'}
      };
    });
    suite.settings
      ..set('vmToken', 'tok')
      ..set('tvIp', '192.168.4.21')
      ..set('tvAuth', 'Z');
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await suite.setHandsFree(true);
    await tester.pump();
    Future<void> hear(String text) async {
      w.speech.say(text);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));
    }

    await hear('Set a timer for 5 minutes.');
    expect(w.system.phoneCalls.last['do'], 'remind', reason: 'no "Claude" needed');
    await hear('All lights off.');
    expect(lights, ['alllights-di8jv']);
    expect(w.glasses.screens().last, contains('All lights'));
    await hear('Turn off the TV');
    expect(tvKeys, ['11/0']);
    // Anything else without the wake word is still ignored.
    await hear('the lights in this movie are amazing');
    await hear('TV');
    expect(lights, hasLength(1));
    expect(suite.tvRemote, isFalse);
    expect(w.bridge.sentTexts, isEmpty, reason: 'nothing went to Claude');

    // The remote: then just the keys, until "done".
    await suite.submit('TV remote', spoken: true);
    await tester.pump();
    expect(suite.tvRemote, isTrue);
    expect(suite.voice.followUp, isTrue, reason: 'listening for the next key');
    w.speech.say('Up.', finishing: true);
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(tvKeys.last, '3/8');
    expect(suite.tvRemote, isTrue);
    w.speech.say('volume down', finishing: true);
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(tvKeys.sublist(tvKeys.length - 3), ['5/0', '5/0', '5/0']);
    w.speech.say('done', finishing: true);
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));
    expect(suite.tvRemote, isFalse);
    expect(w.bridge.sentTexts, isEmpty);
    await suite.setHandsFree(false);
    await tester.pump(const Duration(seconds: 3));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
  testWidgets('the link: heartbeats keep coming while the glasses chatter (looking up and down)', (tester) async {
    final w = _World();
    final suite = w.suite;
    suite.ble.clock = () => tester.binding.clock.now(); // (DateTime.now is not the test's time)
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump(const Duration(seconds: 12)); // the start-up writes are done
    int beats() => w.glasses.sent['L']!.where((p) => p[0] == 0x25).length;
    final before = beats();
    // The glasses answer their own head-up screen every few seconds (22 0a
    // on the right arm, as in the log before each drop). What they send is
    // not a sign of life they count: they hang up 30 s after our last write.
    for (var i = 0; i < 12; i++) {
      w.glasses.injectReply('R', Uint8List.fromList([0x22, 0x0a, 0, 0, 1, 0, 0, 1, 0, 1]));
      await tester.pump(const Duration(seconds: 3));
    }
    expect(beats() - before, greaterThanOrEqualTo(3), reason: '36 s of chatter: at least 3 heartbeats of ours');
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('the link: the background service follows what is wanted, not what is up', (tester) async {
    final w = _World();
    final suite = w.suite;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    final calls = w.system.serviceCalls;
    expect(calls.last, contains('glasses=true'));

    // The link drops (they stopped answering: GATT status 8): it is being rebuilt, so the
    // service stays - Android must not freeze the app while it reconnects.
    await w.control.emit('glassesDisconnected', {'arm': 'R', 'reason': 'link', 'gattStatus': 8});
    await tester.pump();
    expect(suite.ble.state, LinkState.connecting);
    expect(suite.ble.wantsLink, isTrue);
    expect(calls.where((c) => c.contains('glasses=false')), isEmpty, reason: 'not stopped by a drop');

    // Back up, then the user lets go of them: now it may stop.
    await w.control.connectNow();
    await tester.pump();
    await w.control.emit('glassesDisconnected', {'reason': 'user'});
    await tester.pump();
    expect(suite.ble.wantsLink, isFalse);
    expect(calls.last, contains('glasses=false'));
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));

  testWidgets('Settings: "Stay connected" shows what Android is allowed to do, and asks in one tap', (tester) async {
    final w = _World();
    final suite = w.suite;
    w.system.batteryOk = false;
    await w.control.connectNow();
    await tester.pumpWidget(EvenSuiteApp(suite: suite));
    await tester.pump();
    final nav = tester.state<NavigatorState>(find.byType(Navigator).first);
    nav.push(MaterialPageRoute<void>(builder: (_) => const SettingsPage()));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(find.text('Stay connected'), findsWidgets);
    expect(find.text('The glasses: Linked'), findsOneWidget);
    expect(find.text('Battery: restricted'), findsOneWidget);
    await tester.tap(find.text('Allow'));
    await tester.pump();
    expect(w.system.batteryRequests, 1);
    // Allowed (as the phone's dialog would leave it): back from it, the card says so.
    w.system.batteryOk = true;
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.paused);
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
    await tester.pump();
    await tester.pump();
    expect(find.text('Battery: not restricted'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  }, timeout: const Timeout(Duration(minutes: 2)));
}
