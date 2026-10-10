import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

/// Switching models in one place, on a small phone with large text.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_models_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  /// Puts [m]'s files in place as a finished download would.
  void install(ModelAsset m) {
    for (final name in m.installedFileNames) {
      s.models.file(m, name)
        ..createSync(recursive: true)
        ..writeAsStringSync(name);
    }
    s.models.markInstalled(m);
  }

  Future<void> show(WidgetTester tester) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3; // 360 wide, a small phone
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: VoxTheme.light(),
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: const TextScaler.linear(1.3)),
        child: child!,
      ),
      home: ModelsScreen(services: s),
    ));
    await tester.pumpAndSettle();
  }

  Future<void> reveal(WidgetTester tester, Finder f) async {
    final list = find.descendant(of: find.byKey(const ValueKey('models-list')), matching: find.byType(Scrollable)).first;
    await tester.scrollUntilVisible(f, 200, scrollable: list);
    await tester.pumpAndSettle();
  }

  testWidgets('each job with its choices, and which model is in use', (tester) async {
    install(ModelCatalog.textEmbedder);
    await show(tester);
    expect(find.text('Hearing speech'), findsOneWidget);
    expect(find.text('Standard · fp16'), findsOneWidget);
    expect(find.text('Small · int8'), findsOneWidget);
    await reveal(tester, find.text('Small · 4-bit'));
    expect(find.text('Search by meaning'), findsOneWidget);
    expect(find.text('Standard · 8-bit'), findsOneWidget);
    expect(find.text('In use'), findsOneWidget, reason: 'the standard search model');
    await reveal(tester, find.text('Gemma 4 E2B (faster)'));
    expect(find.text('Assistant'), findsOneWidget);
  });

  testWidgets('switching to a search size already downloaded is immediate', (tester) async {
    install(ModelCatalog.textEmbedder);
    install(ModelCatalog.textEmbedderSmall);
    await show(tester);
    await reveal(tester, find.text('Small · 4-bit'));
    await tester.tap(find.text('Small · 4-bit'));
    await tester.pumpAndSettle();
    expect(s.settings.value.searchModel, 'q4');
    expect(s.searchModel?.id, ModelCatalog.textEmbedderSmall.id);
    expect(s.searchModelId, ModelCatalog.textEmbedderSmall.id, reason: 'its own vectors');
    expect(find.text('In use'), findsOneWidget);
    expect(find.byTooltip('Delete to free space'), findsOneWidget, reason: 'the standard one, no longer in use');
  });

  testWidgets('a big download is confirmed first', (tester) async {
    await show(tester);
    final full = find.text('Full precision · fp32').last;
    await reveal(tester, full);
    await tester.tap(full);
    await tester.pumpAndSettle();
    expect(find.text('Download 1.2 GB?'), findsOneWidget);
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();
    expect(s.settings.value.searchModel, 'int8', reason: 'nothing changed');
  });

  testWidgets('search by meaning and tone of voice can be switched off', (tester) async {
    await show(tester);
    await reveal(tester, find.text('Off · words only'));
    await tester.tap(find.text('Off · words only'));
    await tester.pumpAndSettle();
    expect(s.settings.value.meaningSearch, isFalse);
    await reveal(tester, find.text('Off'));
    await tester.tap(find.text('Off'));
    await tester.pumpAndSettle();
    expect(s.settings.value.hearTone, isFalse);
  });
}
