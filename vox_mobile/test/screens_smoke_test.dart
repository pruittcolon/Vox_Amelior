import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/appearance_screen.dart';
import 'package:vox_amelior_mobile/ui/ask_screen.dart';
import 'package:vox_amelior_mobile/ui/capacity_screen.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/more_screen.dart';
import 'package:vox_amelior_mobile/ui/people_screen.dart';
import 'package:vox_amelior_mobile/ui/reviews_screen.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';
import 'package:vox_amelior_mobile/ui/voice_clips_screen.dart';

import 'support/fakes.dart';

/// Renders every new screen on a small phone with large text, to catch
/// layout overflows and crashes that unit tests cannot see.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;
  late int reviewId;
  late int conversationId;
  final monday = DateTime.now().subtract(Duration(days: DateTime.now().weekday + 6));

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
    final far = voiceprint(600, noise: 0);
    final alex = s.speakers.create(name: 'Alexandria Montgomery', embeddingModel: 'f', samples: [
      for (var i = 0; i < 12; i++) voiceprint(1, variant: i),
      for (var i = 0; i < 12; i++)
        Float32List.fromList([for (var d = 0; d < kDim; d++) voiceprint(1, variant: 50 + i)[d] + 0.8 * far[d]]),
    ]);
    final sam = s.speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
    for (var i = 0; i < 40; i++) {
      final seg = s.transcripts.addSegment(
        text: 'Line $i about the garden, the budget and a rather long sentence that wraps across the screen',
        startedAt: monday.add(Duration(hours: 18, minutes: i)),
        duration: const Duration(seconds: 3),
        speakerId: i.isEven ? alex.id : sam.id,
        embedding: voiceprint(i.isEven ? 1 : 2, variant: 100 + i),
      );
      conversationId = seg.conversationId;
      if (i < 3) s.clips.maybeSave(const ClipPolicy(mode: ClipMode.everyone), seg, Float32List(16000));
    }
    s.speakers.markNotSpeaker(s.transcripts.recent(limit: 1).single.id);
    final llm = FakeLlm()
      ..responder = (p) => p.contains('overview')
          ? 'Mostly generalisations.'
          : '- [1] "a quote" — Kind (date, name, number, place or other) — a fairly long note explaining it\n- [2] "b" — Strawman — why';
    final engine = ReviewEngine(reviews: s.reviews, transcripts: s.transcripts, llm: llm);
    reviewId = engine.start(
      title: 'Logical fallacies',
      prompt: ReviewTemplate.builtIns.first.prompt,
      format: ReviewTemplate.builtIns.first.format,
      kind: ReviewKind.list,
      periodLabel: 'last week (Mon – Sun)',
      from: monday,
      to: monday.add(const Duration(days: 7)),
      budget: const ContextBudget(2048),
    )!;
    while (await engine.step(reviewId)) {}
    engine.start(
      title: 'Summary',
      prompt: 'x',
      format: 'y',
      kind: ReviewKind.summary,
      periodLabel: 'this week',
      from: monday,
      to: monday.add(const Duration(days: 7)),
      budget: const ContextBudget(2048),
    );
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> show(WidgetTester tester, Widget screen, {Brightness brightness = Brightness.light}) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3; // 360 × 760, a small phone
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      key: UniqueKey(), // a fresh app, so pages pushed by an earlier step are gone
      theme: VoxTheme.light(const Appearance(corners: 'square')),
      darkTheme: VoxTheme.dark(),
      themeMode: brightness == Brightness.dark ? ThemeMode.dark : ThemeMode.light,
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: const TextScaler.linear(1.3)),
        child: child!,
      ),
      home: screen,
    ));
    await tester.pumpAndSettle();
  }

  /// Scrolls the page's main list until [f] is built and visible.
  Future<void> reveal(WidgetTester tester, Finder f) async {
    final list = find.descendant(of: find.byType(ListView).last, matching: find.byType(Scrollable)).first;
    final position = tester.state<ScrollableState>(list).position;
    if (f.evaluate().isEmpty) {
      position.jumpTo(0);
      await tester.pump();
    }
    for (var i = 0; i < 60 && f.evaluate().isEmpty; i++) {
      position.jumpTo((position.pixels + 250).clamp(0, position.maxScrollExtent));
      await tester.pump();
    }
    await tester.ensureVisible(f.first);
    await tester.pumpAndSettle();
  }

  Future<void> tapText(WidgetTester tester, String text) async {
    final f = find.text(text);
    await reveal(tester, f);
    await tester.tap(f.first);
    await tester.pumpAndSettle();
  }

  testWidgets('reviews: list, detail with findings, and the creator', (tester) async {
    await show(tester, Scaffold(body: ReviewsView(services: s)));
    expect(find.text('Logical fallacies'), findsOneWidget);
    expect(find.text('Summary'), findsOneWidget);

    await tester.tap(find.text('Logical fallacies'));
    await tester.pumpAndSettle();
    expect(find.text('RESULT'), findsOneWidget);
    expect(find.textContaining('4 found'), findsOneWidget);
    expect(find.textContaining('Strawman'), findsWidgets);
    await tapText(tester, 'Strawman 2');
    await tapText(tester, '"b"');
    expect(find.byType(ConversationScreen), findsOneWidget);
    await tester.pageBack();
    await tester.pumpAndSettle();
    await tester.pageBack();
    await tester.pumpAndSettle();

    await tapText(tester, 'New review');
    expect(find.text('Start review'), findsOneWidget);
    await tapText(tester, 'Last week');
    await reveal(tester, find.textContaining('40 lines'));
    expect(find.textContaining('40 lines'), findsOneWidget);
    await tapText(tester, 'Summary');
    await tapText(tester, 'Your own');
    await reveal(tester, find.text('Count findings'));
    expect(find.text('Count findings'), findsOneWidget);
  });

  testWidgets('conversation: "Not [name]" and "New person…" in the menu', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversationId));
    // Lines are built as they scroll into view.
    final line = find.textContaining('Line 0 about').first;
    await tester.scrollUntilVisible(
      line,
      200,
      scrollable: find.descendant(of: find.byKey(const ValueKey('conversation-lines')), matching: find.byType(Scrollable)).first,
    );
    await tester.pumpAndSettle();
    await tester.tap(line);
    await tester.pumpAndSettle();
    expect(find.text('Not Alexandria Montgomery'), findsOneWidget);
    final sheet = find.descendant(of: find.byType(BottomSheet), matching: find.byType(Scrollable)).first;
    await tester.scrollUntilVisible(find.text('New person…'), 100, scrollable: sheet);
    expect(find.text('New person…'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Not Alexandria Montgomery'), -100, scrollable: sheet);
    await tester.tap(find.text('Not Alexandria Montgomery'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Guest'), findsWidgets);
  });

  testWidgets('people, voice clips, capacity, appearance, settings and more render', (tester) async {
    await show(tester, PeopleScreen(services: s));
    expect(find.textContaining('voice pattern'), findsWidgets);
    expect(find.byWidgetPredicate((w) => w is Text && (w.data ?? '').contains('not Sam')), findsOneWidget);

    await show(tester, VoiceClipsScreen(services: s));
    expect(find.text('3 clips · <1 min of speech'), findsOneWidget);
    await tester.tap(find.byType(Switch));
    await tester.pumpAndSettle();
    await tapText(tester, 'Only chosen');
    expect(s.settings.value.clipMode, ClipMode.chosen);
    await tapText(tester, 'Sam');
    expect(s.settings.value.clipPeople, isNotEmpty);

    await show(tester, CapacityScreen(services: s));
    await reveal(tester, find.text('Test this phone'));
    expect(find.text('Test this phone'), findsOneWidget);
    expect(find.textContaining('Recommended per review part'), findsOneWidget);
    expect(find.textContaining('2,048 tokens'), findsWidgets);

    await show(tester, AppearanceScreen(services: s), brightness: Brightness.dark);
    await tapText(tester, 'Dark');
    expect(s.settings.value.themeMode, 'dark');
    await tapText(tester, 'Soft');
    expect(s.settings.value.corners, 'soft');

    await show(tester, SettingsScreen(services: s));
    expect(find.text('Microphone & hearing'), findsOneWidget);
    expect(find.textContaining('Mic boost +15%'), findsOneWidget);
    await tapText(tester, 'Assistant');
    await reveal(tester, find.text('Gemma capacity'));
    await tester.pageBack();
    await tester.pumpAndSettle();
    await tapText(tester, 'Voices & speakers');
    await reveal(tester, find.text('Several voice patterns per person'));
    expect(find.text('Several voice patterns per person'), findsOneWidget);
    await show(tester, MoreScreen(services: s));
    await reveal(tester, find.text('Appearance'));
    await reveal(tester, find.text('Voice clips'));
    expect(find.text('Voice clips'), findsOneWidget);
  });

  testWidgets('ask: both tabs render on a small phone', (tester) async {
    await show(
      tester,
      AskScreen(
        ask: (_) => const Stream.empty(),
        requests: s.requests,
        assistantReady: () => false,
        onEditPrompt: () {},
        reviewsBuilder: (_) => ReviewsView(services: s),
        onReviewPeriod: (_) {},
      ),
    );
    await tapText(tester, 'This week');
    await reveal(tester, find.text('Go through everything from this week'));
    await tester.tap(find.text('Go through all'));
    await tester.pumpAndSettle();
    expect(find.text('New review'), findsOneWidget);
  });

  test('settings for tests', () {
    expect(const AppSettings().budget.recommendedChunkTokens, 2048);
  });
}
