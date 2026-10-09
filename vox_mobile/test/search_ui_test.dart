import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';
import 'package:vox_amelior_mobile/ui/timeline_screen.dart';

import 'support/fakes.dart';

/// Searching the Timeline by words and by meaning, on a small phone with
/// large text.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  Future<void> setUpServices({bool meaning = true}) async {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_search_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db, embedder: meaning ? () => InlineEmbedder(FakeTextEmbedder()) : null);
    final p = s.speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]).id;
    final e = s.speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]).id;
    final t = DateTime.now().subtract(const Duration(hours: 3));
    s.transcripts.addSegment(text: 'We cannot pay the electric bill this month', startedAt: t, duration: const Duration(seconds: 3), speakerId: e);
    s.transcripts.addSegment(
        text: 'Rent is due', startedAt: t.add(const Duration(seconds: 10)), duration: const Duration(seconds: 3), speakerId: p);
    s.transcripts.addSegment(
        text: 'The puppy chewed my shoe again', startedAt: t.add(const Duration(hours: 1)), duration: const Duration(seconds: 3), speakerId: p);
    // Every line embedded before the screen opens (no timers left behind).
    if (meaning) await s.indexer.run();
  }

  tearDown(() {
    s.indexer.dispose();
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> show(WidgetTester tester, Widget screen) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: VoxTheme.light(),
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: const TextScaler.linear(1.3)),
        child: child!,
      ),
      home: screen,
    ));
    await tester.pumpAndSettle();
  }

  Future<void> type(WidgetTester tester, String text) async {
    await tester.enterText(find.byType(TextField).first, text);
    await tester.pump(const Duration(milliseconds: 300)); // the search debounce
    await tester.pumpAndSettle();
  }

  testWidgets('smart search finds what was said in other words, and says how', (tester) async {
    // Indexing waits on real timers: outside the test's fake clock.
    await tester.runAsync(() => setUpServices());
    await show(tester, TimelineScreen(services: s));
    await type(tester, 'money');
    expect(find.text('Smart'), findsOneWidget);
    expect(find.text('Meaning'), findsOneWidget);
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget);
    expect(find.textContaining('Rent is due'), findsOneWidget);
    expect(find.textContaining('puppy'), findsNothing);
    expect(find.textContaining('· meaning'), findsNWidgets(2));
    expect(find.textContaining('Searched by words and meaning'), findsOneWidget);

    await tester.tap(find.text('Exact words'));
    await tester.pumpAndSettle();
    expect(find.text('No matches'), findsOneWidget, reason: 'no line contains "money"');

    await tester.tap(find.text('Meaning'));
    await tester.pumpAndSettle();
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget);
  });

  testWidgets('a line found both ways is labelled with both', (tester) async {
    // Indexing waits on real timers: outside the test's fake clock.
    await tester.runAsync(() => setUpServices());
    await show(tester, TimelineScreen(services: s));
    await type(tester, 'electric bill');
    expect(find.textContaining('words + meaning'), findsOneWidget);
    expect(find.textContaining('Ask Gemma: "electric bill"'), findsOneWidget, reason: 'a question-sized search offers an answer');
  });

  testWidgets('without the model: words only, and an offer to download meaning search', (tester) async {
    await setUpServices(meaning: false);
    await show(tester, TimelineScreen(services: s));
    await type(tester, 'electric');
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget);
    expect(find.text('Meaning'), findsNothing);
    expect(find.text('Also find what was said in other words'), findsOneWidget);
    await type(tester, 'money');
    expect(find.text('No matches'), findsOneWidget);
  });

  testWidgets('settings: search by meaning, with how many lines are ready', (tester) async {
    await setUpServices(meaning: false);
    await show(tester, Scaffold(body: ListView(children: [MeaningSearchSettings(services: s)])));
    expect(find.text('Search by meaning'), findsOneWidget);
    expect(find.text('Download search by meaning'), findsOneWidget);
    await tester.tap(find.text('Search by meaning'));
    await tester.pumpAndSettle();
    expect(s.settings.value.meaningSearch, isFalse);
    expect(find.text('Download search by meaning'), findsNothing, reason: 'switched off: nothing to download');
  });
}
