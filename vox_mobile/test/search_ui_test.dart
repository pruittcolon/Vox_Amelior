import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

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
  _SlowQueries? slow;

  Future<void> setUpServices({bool meaning = true}) async {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_search_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db, embedder: meaning ? () => slow = _SlowQueries(InlineEmbedder(FakeTextEmbedder())) : null);
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

  testWidgets('a line heard during a search: the results stay on screen while they refresh', (tester) async {
    await tester.runAsync(() => setUpServices());
    await show(tester, TimelineScreen(services: s));
    await type(tester, 'money');
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget);

    // Meaning search was unloaded meanwhile: the next refresh has to run the
    // model again, which takes a while.
    await tester.runAsync(() async {
      await s.updateSettings(s.settings.value.copyWith(meaningSearch: false));
      await s.updateSettings(s.settings.value.copyWith(meaningSearch: true));
    });
    final hold = Completer<void>();
    slow!.hold = hold.future;
    await tester.runAsync(() async {
      s.transcripts.addSegment(
          text: 'Did you pay the water bill yet',
          startedAt: DateTime.now().subtract(const Duration(minutes: 5)),
          duration: const Duration(seconds: 3));
      s.dataVersion.value++; // as when the listening service adds a line
      s.indexForSearch();
    });
    await tester.pump();
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget, reason: 'not emptied while refreshing');
    expect(find.byType(CircularProgressIndicator), findsNothing);

    await tester.runAsync(() async {
      hold.complete();
      await Future<void>.delayed(const Duration(milliseconds: 400));
    });
    await tester.pumpAndSettle();
    expect(find.textContaining('Did you pay the water bill yet'), findsOneWidget, reason: 'the new line joins once it is ready');
    expect(find.textContaining('We cannot pay the electric bill'), findsOneWidget);
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

/// Queries wait for [hold], as when the model has to load first.
class _SlowQueries implements AsyncEmbedder {
  _SlowQueries(this._inner);

  final AsyncEmbedder _inner;
  Future<void>? hold;

  @override
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task}) async {
    if (task == EmbedTask.query && hold != null) await hold;
    return _inner.embed(texts, task: task);
  }

  @override
  Future<void> close() => _inner.close();
}
