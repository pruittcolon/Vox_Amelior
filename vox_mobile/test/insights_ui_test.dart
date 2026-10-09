import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/insights_screen.dart';
import 'package:vox_amelior_mobile/ui/people_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';
import 'package:vox_amelior_mobile/ui/timeline_screen.dart';

import 'support/fakes.dart';

/// The Insights tab, the sortable Timeline and People cards, on a small
/// phone with large text (overflows fail the test).
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;
  late String pruitt;
  late String ericah;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_insights_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
    pruitt = s.speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]).id;
    ericah = s.speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]).id;
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  void line(String text, DateTime at, String who, {String? emotion, String? sound, int seconds = 4}) {
    final seg = s.transcripts.addSegment(text: text, startedAt: at, duration: Duration(seconds: seconds), speakerId: who);
    if (emotion != null || sound != null) s.transcripts.setTone(seg.id, emotion: emotion, sound: sound);
  }

  /// Two conversations: a calm one three days ago, a heated one two hours ago.
  void twoConversations({bool tones = true}) {
    final calm = DateTime.now().subtract(const Duration(days: 3));
    line('The garden looks lovely this year', calm, pruitt, emotion: tones ? 'happy' : null);
    line('The garden needs water tomorrow', calm.add(const Duration(seconds: 10)), ericah, emotion: tones ? 'neutral' : null);
    final heated = DateTime.now().subtract(const Duration(hours: 2));
    line('Why is the garden hose still outside', heated, ericah, emotion: tones ? 'angry' : null, seconds: 6);
    line('Because you asked me to water the garden', heated.add(const Duration(seconds: 10)), pruitt, emotion: tones ? 'angry' : null);
    line('Fine, that is funny actually', heated.add(const Duration(seconds: 20)), ericah, emotion: tones ? 'happy' : null, sound: tones ? 'laughter' : null);
  }

  Future<void> show(WidgetTester tester, Widget screen, {Brightness brightness = Brightness.light}) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3; // 360 × 760, a small phone
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      key: UniqueKey(),
      theme: VoxTheme.light(),
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

  /// Scrolls the page (not a chip row) until [f] is on screen.
  Future<void> reveal(WidgetTester tester, Finder f) async {
    final page = find.byWidgetPredicate((w) => w is ListView && w.scrollDirection == Axis.vertical).first;
    await tester.scrollUntilVisible(f, 200, scrollable: find.descendant(of: page, matching: find.byType(Scrollable)).first);
    await tester.pumpAndSettle();
  }

  group('Insights', () {
    for (final brightness in Brightness.values) {
      testWidgets('every section renders ($brightness)', (tester) async {
        twoConversations();
        await show(tester, InsightsScreen(services: s), brightness: brightness);
        expect(find.text('Insights'), findsOneWidget);
        expect(find.text('Talk time'), findsOneWidget);
        expect(find.text('Conversations'), findsOneWidget);
        expect(find.text('2'), findsWidgets, reason: 'two conversations');
        for (final title in ['Mood over time', 'Who talks', 'When you talk', 'Together', 'Standout conversations', 'Most said']) {
          await reveal(tester, find.text(title));
          expect(find.text(title), findsOneWidget, reason: title);
        }
        expect(find.byType(StackedColumnChart), findsOneWidget);
        expect(find.byType(WeekHourHeatmap), findsOneWidget);
        expect(find.textContaining('garden · '), findsOneWidget, reason: 'the most said word');
      });
    }

    testWidgets('tapping a mood column shows that stretch; the legend opens those lines', (tester) async {
      twoConversations();
      await show(tester, InsightsScreen(services: s));
      await reveal(tester, find.byType(StackedColumnChart));
      expect(find.textContaining('had a clear feeling'), findsOneWidget);
      final chart = tester.getRect(find.byType(StackedColumnChart));
      await tester.tapAt(Offset(chart.right - 4, chart.top + 40));
      await tester.pumpAndSettle();
      expect(find.textContaining('had a clear feeling'), findsNothing, reason: 'the selected column is described instead');

      await reveal(tester, find.widgetWithText(ActionChip, 'Angry 2'));
      await tester.tap(find.widgetWithText(ActionChip, 'Angry 2'));
      await tester.pumpAndSettle();
      expect(find.byType(TimelineScreen), findsOneWidget);
      expect(find.textContaining('that sounded angry'), findsOneWidget);
      expect(find.text('Why is the garden hose still outside'), findsOneWidget);
    });

    testWidgets('one person: their numbers and who they talk with', (tester) async {
      twoConversations();
      await show(tester, InsightsScreen(services: s));
      await reveal(tester, find.text('Who talks'));
      await tester.tap(find.widgetWithText(RankBar, 'Ericah'));
      await tester.pumpAndSettle();
      expect(find.widgetWithText(AppBar, 'Ericah'), findsOneWidget);
      await reveal(tester, find.text('Talks most with'));
      expect(find.widgetWithText(RankBar, 'Pruitt'), findsOneWidget);
      expect(find.text('Who talks'), findsNothing);

      await tester.tap(find.widgetWithText(RankBar, 'Pruitt'));
      await tester.pumpAndSettle();
      expect(find.byType(TimelineScreen), findsOneWidget);
      expect(find.textContaining('2 conversations with'), findsOneWidget);
    });

    testWidgets('without tones, it offers to set up tone of voice', (tester) async {
      twoConversations(tones: false);
      await show(tester, InsightsScreen(services: s));
      await reveal(tester, find.text('Set up tone of voice'));
      expect(find.text('Set up tone of voice'), findsOneWidget);
      expect(find.byType(StackedColumnChart), findsNothing);
    });

    testWidgets('nothing heard yet', (tester) async {
      await show(tester, InsightsScreen(services: s));
      expect(find.text('Nothing to show yet'), findsOneWidget);
      await tester.tap(find.text('All'));
      await tester.pumpAndSettle();
      expect(find.text('Nothing to show yet'), findsOneWidget);
    });

    testWidgets('standouts switch between heated, laughter and longest', (tester) async {
      twoConversations();
      await show(tester, InsightsScreen(services: s));
      await reveal(tester, find.text('Most heated'));
      expect(find.text('😠 2'), findsOneWidget);
      await tester.tap(find.text('Most laughter'));
      await tester.pumpAndSettle();
      expect(find.text('😂 1'), findsOneWidget);
      await tester.tap(find.text('Longest'));
      await tester.pumpAndSettle();
      expect(find.text('No conversations longer than a minute.'), findsOneWidget);
    });
  });

  group('Timeline', () {
    testWidgets('opens already filtered, and filtered results can be sorted', (tester) async {
      twoConversations();
      await show(tester, TimelineScreen(services: s, initialPeople: {pruitt}));
      expect(find.textContaining('2 conversations with Pruitt'), findsOneWidget);
      // Newest first, under day headings.
      final heatedTop = tester.getTopLeft(find.text('Why is the garden hose still outside')).dy;
      final calmTop = tester.getTopLeft(find.text('The garden looks lovely this year')).dy;
      expect(heatedTop, lessThan(calmTop));

      await tester.tap(find.byTooltip('Sort'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Oldest').last);
      await tester.pumpAndSettle();
      expect(
        tester.getTopLeft(find.text('The garden looks lovely this year')).dy,
        lessThan(tester.getTopLeft(find.text('Why is the garden hose still outside')).dy),
      );

      await tester.tap(find.byTooltip('Sort'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Most heated').last);
      await tester.pumpAndSettle();
      expect(find.text('Sorted by most heated'), findsOneWidget);
      expect(
        tester.getTopLeft(find.text('Why is the garden hose still outside')).dy,
        lessThan(tester.getTopLeft(find.text('The garden looks lovely this year')).dy),
      );
    });

    testWidgets('search results show the searched words in bold', (tester) async {
      twoConversations();
      await show(tester, TimelineScreen(services: s, initialQuery: 'hose'));
      final rich = tester.widgetList<RichText>(find.byType(RichText)).where((r) => r.text.toPlainText() == 'Why is the garden hose still outside');
      expect(rich, isNotEmpty);
      final spans = <TextSpan>[];
      rich.first.text.visitChildren((span) {
        if (span is TextSpan && span.text != null) spans.add(span);
        return true;
      });
      final bold = spans.where((sp) => sp.style?.fontWeight == FontWeight.w800).map((sp) => sp.text).toList();
      expect(bold, ['hose']);
    });
  });

  testWidgets('People: when each person was last heard, tap for their statistics', (tester) async {
    twoConversations();
    await show(tester, PeopleScreen(services: s));
    expect(find.textContaining('Last heard'), findsWidgets);
    expect(find.byType(MoodStrip), findsWidgets);
    expect(find.byTooltip('Settings and more'), findsOneWidget);
    await tester.tap(find.text('Ericah'));
    await tester.pumpAndSettle();
    expect(find.byType(InsightsScreen), findsOneWidget);
    expect(find.widgetWithText(AppBar, 'Ericah'), findsOneWidget);
  });
}
