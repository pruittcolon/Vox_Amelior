// Renders every screen on a 360x780 dp phone with real fonts, at normal and
// large (1.3x) text. Any overflow fails the test. PNGs land in
// test/screenshots/out/ (CI uploads them).
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/app_state.dart';
import 'package:narrator_studio/core/audio/exporter.dart';
import 'package:narrator_studio/core/audio/player.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/ui/cast_sheet.dart';
import 'package:narrator_studio/ui/clone_screen.dart';
import 'package:narrator_studio/ui/export_sheet.dart';
import 'package:narrator_studio/ui/gallery_screen.dart';
import 'package:narrator_studio/ui/guide_screen.dart';
import 'package:narrator_studio/ui/home_shell.dart';
import 'package:narrator_studio/ui/mix_screen.dart';
import 'package:narrator_studio/ui/preview.dart';
import 'package:narrator_studio/ui/reader_screen.dart';
import 'package:narrator_studio/ui/scope.dart';
import 'package:narrator_studio/ui/settings_screen.dart';
import 'package:narrator_studio/ui/theme.dart';
import 'package:narrator_studio/ui/voice_picker.dart';

import 'helpers/fakes.dart';

final _shotKey = GlobalKey();

Future<void> _loadFonts() async {
  final candidates = <String>[
    if (Platform.environment['FLUTTER_ROOT'] != null)
      '${Platform.environment['FLUTTER_ROOT']}/bin/cache/artifacts/material_fonts',
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

Future<void> _settle(WidgetTester tester) async {
  for (var i = 0; i < 6; i++) {
    await tester.pump(const Duration(milliseconds: 120));
  }
}

/// Lets real file I/O started by the UI finish, then redraws. Each await on
/// I/O resumes only after a pump, so alternate a few times.
Future<void> _io(WidgetTester tester) async {
  for (var i = 0; i < 6; i++) {
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 60)));
    await tester.pump(const Duration(milliseconds: 50));
  }
  await _settle(tester);
}

Future<void> _shot(WidgetTester tester, String name) async {
  await _settle(tester);
  expect(tester.takeException(), isNull, reason: 'layout problem on $name');
  final boundary = tester.renderObject<RenderRepaintBoundary>(find.byKey(_shotKey));
  await tester.runAsync(() async {
    final image = await boundary.toImage(pixelRatio: 1.5);
    final png = await image.toByteData(format: ui.ImageByteFormat.png);
    final file = File('test/screenshots/out/$name.png')..createSync(recursive: true);
    file.writeAsBytesSync(png!.buffer.asUint8List());
  });
}

class _World {
  _World(this.app, this.native, this.synth);
  final AppState app;
  final FakeNative native;
  final FakeSynth synth;
  late final player = Player(app, native);
  late final exporter = Exporter(app, native);
  late final preview = VoicePreview(app, native, player);

  Widget wrap(Widget home, double scale, {Brightness brightness = Brightness.light}) => RepaintBoundary(
        key: _shotKey,
        child: AppScope(
          app: app,
          player: player,
          exporter: exporter,
          native: native,
          preview: preview,
          child: MaterialApp(
            debugShowCheckedModeBanner: false,
            theme: buildTheme(Brightness.light),
            darkTheme: buildTheme(Brightness.dark),
            themeMode: brightness == Brightness.dark ? ThemeMode.dark : ThemeMode.light,
            builder: (context, child) => MediaQuery(
              data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(scale)),
              child: child!,
            ),
            home: home,
          ),
        ),
      );
}

const _gallery = {
  'version': 1,
  'voices': [
    {
      'id': 'voice-zero-caro-davy', 'set': 'voice-zero', 'name': 'Caro (LibriVox narrator)', 'asset': 'voice--voice-zero-caro-davy.wav',
      'size': 100, 'sha256': '0000000000000000000000000000000000000000000000000000000000000000', 'seconds': 8.4,
      'tone': 'medium', 'license': 'CC0', 'source': 'LibriVox narrator (Voice-Zero)',
    },
    {
      'id': 'vctk-p234-023', 'set': 'vctk', 'name': 'VCTK p234', 'asset': 'voice--vctk-p234-023.wav', 'size': 100,
      'sha256': '0000000000000000000000000000000000000000000000000000000000000000', 'seconds': 10.0, 'tone': 'high',
      'license': 'CC-BY-4.0', 'source': 'VCTK corpus', 'gender': 'F', 'accent': 'Scottish', 'region': 'West Dumfries',
    },
    {
      'id': 'voice-donations-0a67', 'set': 'voice-donations', 'name': 'Donor 0a67', 'asset': 'voice--voice-donations-0a67.wav',
      'size': 100, 'sha256': '0000000000000000000000000000000000000000000000000000000000000000', 'seconds': 10.0,
      'tone': 'deep', 'license': 'CC0', 'source': 'Unmute voice donation',
    },
  ],
};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(_loadFonts);

  for (final scale in [1.0, 1.3]) {
    final tag = scale == 1.0 ? '' : '_large_text';

    testWidgets('every screen fits a 360 dp phone at ${scale}x text', (tester) async {
      tester.view.physicalSize = const Size(1080, 2340);
      tester.view.devicePixelRatio = 3;
      addTearDown(tester.view.reset);

      // First launch: no voices, empty library.
      final fresh = (await tester.runAsync(() => makeApp(packs: [])))!;
      final w0 = _World(fresh.$1, fresh.$3, fresh.$2);
      await tester.pumpWidget(w0.wrap(const HomeShell(), scale));
      await _shot(tester, '01_first_launch$tag');

      // With voices and a few documents.
      final ready = (await tester.runAsync(() => makeApp(packs: ['kokoro', 'supertonic'])))!;
      final w = _World(ready.$1, ready.$3, ready.$2);
      final app = w.app;
      final doc = (await tester.runAsync(() async {
        await app.library.add(title: 'The Origin of Species (chapter 1)', kind: 'pdf', text: 'Plain words. ' * 300);
        await app.library.add(title: 'The Library That Remembered Everything', kind: 'sample', text: sampleAudiobook);
        final d = await app.library.add(title: 'Why Is the Sky Blue?', kind: 'sample', text: samplePodcast);
        d.position = 12;
        d.words = 760;
        d.utterances = 80;
        return d;
      }))!;
      await tester.pumpWidget(w.wrap(const HomeShell(), scale));
      await _shot(tester, '02_library$tag');

      await tester.tap(find.text('Voices').last);
      await _shot(tester, '03_voices$tag');
      await tester.tap(find.text('Write').last);
      await _shot(tester, '04_write$tag');

      // Reader with the sample podcast, sentence 12 highlighted.
      await tester.pumpWidget(w.wrap(ReaderScreen(key: UniqueKey(), doc: doc), scale));
      await _io(tester);
      expect(find.textContaining('Podcast'), findsWidgets);
      await _shot(tester, '05_reader$tag');

      // Voices & style sheet.
      final session = (await tester.runAsync(() => app.openSession(doc)))!;
      await tester.pumpWidget(w.wrap(Builder(builder: (context) {
        return Scaffold(
          body: Center(
            child: FilledButton(onPressed: () => showCastSheet(context, session), child: const Text('open cast')),
          ),
        );
      }), scale));
      await tester.tap(find.text('open cast'));
      await _shot(tester, '06_cast_sheet$tag');

      await tester.pumpWidget(w.wrap(Builder(builder: (context) {
        return Scaffold(
          body: Center(
            child: FilledButton(onPressed: () => showExportSheet(context, session), child: const Text('open export')),
          ),
        );
      }), scale));
      await tester.tap(find.text('open export'));
      await _shot(tester, '07_export$tag');

      await tester.pumpWidget(w.wrap(const VoicePickerScreen(title: 'Voice for GUEST', current: 'char:professor'), scale));
      await _shot(tester, '08_voice_picker$tag');

      await tester.pumpWidget(w.wrap(const GuideScreen(), scale));
      await _shot(tester, '09_guide$tag');

      await tester.pumpWidget(w.wrap(const MixScreen(), scale));
      await _shot(tester, '10_mix$tag');

      await tester.pumpWidget(w.wrap(const CloneScreen(), scale));
      await _shot(tester, '11_clone$tag');

      await tester.runAsync(() async {
        final f = File(app.models.path('gallery', 'gallery.json'));
        await f.parent.create(recursive: true);
        await f.writeAsString(jsonEncode(_gallery));
      });
      await tester.pumpWidget(w.wrap(const GalleryScreen(), scale));
      await _io(tester);
      expect(find.text('Caro (LibriVox narrator)'), findsOneWidget);
      await _shot(tester, '12_gallery$tag');

      await tester.pumpWidget(w.wrap(const SettingsScreen(), scale));
      await _shot(tester, '13_settings$tag');

      if (scale == 1.0) {
        await tester.pumpWidget(w.wrap(ReaderScreen(key: UniqueKey(), doc: doc), scale, brightness: Brightness.dark));
        await _io(tester);
        await _shot(tester, '14_reader_dark');
      }
      await tester.pumpWidget(const SizedBox());
    });
  }
}
