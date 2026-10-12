// Renders every screen on a 360x780 phone with real fonts, at normal and
// large (1.3x) text. Any overflow fails the test. PNGs land in
// test/screenshots/out/ (CI uploads them).
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:even_suite/core/suite.dart';
import 'package:even_suite/core/system.dart';
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
  final glasses = FakeGlasses();
  final control = FakeControl();
  final speech = FakeSpeech();
  final bridge = FakeBridgeClient();
  late final Suite suite = Suite(
    system: SystemBridge(),
    transport: glasses,
    control: control,
    speech: speech,
    client: bridge,
  );
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
      w.bridge.addReply('Done. I will remind you at 12:00.');
      await tester.pump();
      await _shot(tester, '4_claude_chat$tag');

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

      // Tools, then each tool page.
      await tester.tap(find.byIcon(Icons.widgets_outlined));
      await tester.pump();
      await _shot(tester, '6_tools$tag');
      for (final (label, file) in [
        ('Send text', '7_text'),
        ('Notifications', '8_notifications'),
        ('How to talk to it', '9_help'),
        ('Settings', '10_settings'),
        ('Images', '11_images'),
      ]) {
        await tester.tap(find.text(label).last);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
        await _shot(tester, '$file$tag');
        await tester.pageBack();
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      }

      expect(tester.takeException(), isNull);
      await suite.setHandsFree(false);
      await tester.pumpWidget(const SizedBox());
      suite.dispose();
    });
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

    // Long press left: talk; release: send.
    w.glasses.touch(23, 'L');
    await tester.pump();
    await tester.pump();
    expect(suite.voice.dictating, isTrue);
    expect(suite.voice.holding, isTrue);
    w.speech.partial('what is the weather');
    await tester.pump(const Duration(milliseconds: 500));
    expect(w.glasses.screens().last, contains('what is the weather'));
    w.glasses.touch(24, 'L');
    await tester.pump();
    w.speech.say('what is the weather', finishing: true);
    await tester.pump();
    expect(w.bridge.sentTexts, ['what is the weather']);

    // Double tap: the glasses left our screen.
    w.glasses.touch(0, 'L');
    await tester.pump();
    expect(suite.display.pager, isNull);

    await tester.pumpWidget(const SizedBox());
    suite.dispose();
  });

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
  });
}
