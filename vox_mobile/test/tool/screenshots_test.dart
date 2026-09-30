// Renders the main screens to PNG files so the design can be looked at without a phone.
// Skipped unless VOX_SHOTS_DIR is set:
//   VOX_SHOTS_DIR=/tmp/shots VOX_FONTS=<flutter>/bin/cache/artifacts/material_fonts flutter test test/tool
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';
import 'package:vox_amelior_mobile/ui/listening_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/mic_tune.dart';
import 'package:vox_amelior_mobile/ui/more_screen.dart';
import 'package:vox_amelior_mobile/ui/now_screen.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/speakers_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

Future<void> _loadFonts(String dir) async {
  Future<ByteData> load(String f) async => ByteData.sublistView(File(p.join(dir, f)).readAsBytesSync());
  final roboto = FontLoader('Roboto')
    ..addFont(load('Roboto-Regular.ttf'))
    ..addFont(load('Roboto-Medium.ttf'))
    ..addFont(load('Roboto-Bold.ttf'))
    ..addFont(load('Roboto-Black.ttf'));
  await roboto.load();
  final icons = FontLoader('MaterialIcons')..addFont(load('MaterialIcons-Regular.otf'));
  await icons.load();
}

ThemeData _roboto(ThemeData t) => t.copyWith(textTheme: t.textTheme.apply(fontFamily: 'Roboto'));

void main() {
  final out = Platform.environment['VOX_SHOTS_DIR'];
  final fonts = Platform.environment['VOX_FONTS'];
  // Set VOX_SHOTS_DIR and VOX_FONTS to render screenshots.
  final skip = out == null || fonts == null;

  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  setUpAll(() async {
    if (!skip) await _loadFonts(fonts);
  });
  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_shots_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  final key = GlobalKey();

  Future<void> shot(WidgetTester tester, String name, Widget screen, {Brightness brightness = Brightness.light, double scale = 1.0}) async {
    tester.view
      ..physicalSize = const Size(1080, 2340)
      ..devicePixelRatio = 3; // 360 × 780
    addTearDown(tester.view.reset);
    await tester.pumpWidget(RepaintBoundary(
      key: key,
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: _roboto(VoxTheme.light()),
        darkTheme: _roboto(VoxTheme.dark()),
        themeMode: brightness == Brightness.dark ? ThemeMode.dark : ThemeMode.light,
        builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(scale)),
          child: DefaultTextStyle.merge(style: const TextStyle(fontFamily: 'Roboto'), child: child!),
        ),
        home: screen,
      ),
    ));
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      final boundary = key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final image = await boundary.toImage(pixelRatio: 2);
      final bytes = (await image.toByteData(format: ui.ImageByteFormat.png))!;
      Directory(out!).createSync(recursive: true);
      File(p.join(out, '$name.png')).writeAsBytesSync(bytes.buffer.asUint8List());
    });
  }

  testWidgets('settings screens', (tester) async {
    await shot(tester, 'settings_home', SettingsScreen(services: s));
    await shot(tester, 'settings_hearing', ListeningSettingsScreen(services: s));
    await shot(tester, 'settings_hearing_dark', ListeningSettingsScreen(services: s), brightness: Brightness.dark);
    await shot(tester, 'settings_speakers', SpeakersSettingsScreen(services: s));
    await shot(tester, 'more', MoreScreen(services: s));
  }, skip: skip);

  testWidgets('now screen and meter states', (tester) async {
    await shot(tester, 'now', NowScreen(services: s));
    await shot(
      tester,
      'meter_states',
      Scaffold(
        appBar: AppBar(title: const Text('Level meter')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              for (final r in const [
                MicReading(),
                MicReading(live: true, level: 0.12, peak: 0.2),
                MicReading(live: true, level: 0.62, peak: 0.72),
                MicReading(live: true, level: 0.97, peak: 1, clipping: true),
              ]) ...[
                LevelMeterBar(reading: r),
                const SizedBox(height: 18),
              ],
              const SettingsGroup(title: 'Example', children: [IconBadge(Icons.mic_rounded)]),
            ],
          ),
        ),
      ),
    );
  }, skip: skip);
}
