import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/reviews_screen.dart';
import 'package:vox_amelior_mobile/ui/speakers_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';
import 'package:vox_amelior_mobile/ui/timeline_screen.dart';

import 'support/fakes.dart';

/// Tone of voice on screen: chips on lines, mood filters, copying, and
/// reviews of "the last N lines" of chosen people.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;
  late int conversationId;
  late String pruitt;
  late String ericah;
  final clipboard = <String>[];

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_tone_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
    pruitt = s.speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]).id;
    ericah = s.speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]).id;
    final start = DateTime.now().subtract(const Duration(hours: 2));
    const script = [
      ('Did you take the trash out', 'neutral', null),
      ('I asked you three times already', 'angry', null),
      ('You never listen to me', 'angry', null),
      ('I am sorry, I forgot', 'sad', null),
      ('Okay, that was funny though', 'happy', 'laughter'),
      ('Pass the popcorn', 'neutral', null),
    ];
    for (var i = 0; i < script.length; i++) {
      final (text, emotion, sound) = script[i];
      final seg = s.transcripts.addSegment(
        text: text,
        startedAt: start.add(Duration(seconds: i * 10)),
        duration: const Duration(seconds: 3),
        speakerId: i.isEven ? pruitt : ericah,
      );
      s.transcripts.setTone(seg.id, emotion: emotion, sound: sound);
      conversationId = seg.conversationId;
    }
    clipboard.clear();
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> show(WidgetTester tester, Widget screen, {bool tall = false}) async {
    tester.view
      ..physicalSize = Size(1080, tall ? 4800 : 2280)
      ..devicePixelRatio = 3; // 360 wide, a small phone (tall: every line of a conversation on screen)
    addTearDown(tester.view.reset);
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(SystemChannels.platform, (call) async {
      if (call.method == 'Clipboard.setData') clipboard.add((call.arguments as Map)['text'] as String);
      return null;
    });
    addTearDown(() => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(SystemChannels.platform, null));
    await tester.pumpWidget(MaterialApp(
      key: UniqueKey(),
      theme: VoxTheme.light(),
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: const TextScaler.linear(1.3)),
        child: child!,
      ),
      home: screen,
    ));
    await tester.pumpAndSettle();
  }

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

  testWidgets('a conversation shows each line\'s tone and filters by it', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversationId), tall: true);
    expect(find.text('😠 Angry'), findsNWidgets(2), reason: 'shown on the two angry lines');
    expect(find.text('😂 Laughter'), findsOneWidget);
    expect(find.textContaining('Neutral'), findsNothing, reason: 'neutral is not worth a chip');

    await tester.tap(find.text('😠 Angry · 2'));
    await tester.pumpAndSettle();
    expect(find.text('You never listen to me'), findsOneWidget);
    expect(find.text('Pass the popcorn'), findsNothing);
    expect(find.text('Showing 2 of 6 lines'), findsOneWidget);

    await tester.tap(find.text('😢 Sad · 1'));
    await tester.pumpAndSettle();
    expect(find.text('Showing 3 of 6 lines'), findsOneWidget);
    expect(find.text('I am sorry, I forgot'), findsOneWidget);
  });

  testWidgets('long-press copies a line with name, time and tone', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversationId), tall: true);
    await tester.longPress(find.text('You never listen to me'));
    await tester.pumpAndSettle();
    expect(clipboard.single, endsWith('Pruitt [angry]: You never listen to me'));
    expect(find.text('Line copied'), findsOneWidget);
  });

  testWidgets('"Select text to copy" opens the conversation as selectable text', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversationId), tall: true);
    await tester.tap(find.byType(PopupMenuButton<String>));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Select text to copy'));
    await tester.pumpAndSettle();
    expect(find.byType(SelectTextScreen), findsOneWidget);
    expect(find.byType(SelectionArea), findsOneWidget);
    expect(find.textContaining('Pruitt [happy, laughter]: Okay, that was funny though'), findsOneWidget);
    await tester.tap(find.byTooltip('Copy all'));
    await tester.pumpAndSettle();
    expect(clipboard.single.split('\n'), hasLength(6));
  });

  testWidgets('the timeline finds conversations by mood', (tester) async {
    // A calm conversation the day before.
    s.transcripts.addSegment(
      text: 'Lovely weather',
      startedAt: DateTime.now().subtract(const Duration(days: 1)),
      duration: const Duration(seconds: 2),
      speakerId: pruitt,
    );
    await show(tester, TimelineScreen(services: s));
    expect(find.text('Any mood'), findsOneWidget);
    await tester.tap(find.text('😠 Angry'));
    await tester.pumpAndSettle();
    expect(find.textContaining('that sounded angry'), findsOneWidget);
    expect(find.text('Did you take the trash out'), findsOneWidget);
    expect(find.text('Lovely weather'), findsNothing);
    expect(find.textContaining('😠 2'), findsOneWidget, reason: 'the card shows how often each mood came up');

    // With a person as well: only their angry lines count.
    await tester.tap(find.widgetWithText(FilterChip, 'Ericah'));
    await tester.pumpAndSettle();
    expect(find.textContaining('with Ericah that sounded angry'), findsOneWidget);
    expect(find.text('Did you take the trash out'), findsOneWidget);
  });

  testWidgets('review the last lines of chosen people, only when they sounded angry', (tester) async {
    await show(tester, ReviewCreateScreen(services: s));
    await tester.tap(find.text('Fights & tension'));
    await tester.pumpAndSettle();
    await reveal(tester, find.text('Last lines'));
    await tester.tap(find.text('Last lines'));
    await tester.pumpAndSettle();
    await reveal(tester, find.text('Last 100'));
    expect(find.text('Last 100'), findsOneWidget);
    await reveal(tester, find.widgetWithText(FilterChip, 'Pruitt'));
    await tester.tap(find.widgetWithText(FilterChip, 'Pruitt'));
    await tester.pumpAndSettle();
    await reveal(tester, find.widgetWithText(FilterChip, '😠 Angry'));
    await tester.tap(find.widgetWithText(FilterChip, '😠 Angry'));
    await tester.pumpAndSettle();
    await reveal(tester, find.textContaining('lines →'));
    expect(find.textContaining('1 lines →'), findsOneWidget, reason: 'Pruitt said one angry line');

    final start = tester.widget<FilledButton>(find.ancestor(of: find.text('Start review'), matching: find.byWidgetPredicate((w) => w is FilledButton)));
    expect(start.onPressed, isNotNull);
    // (Starting it would load Gemma; the engine side is tested in tone_test.dart.)
  });

  testWidgets('settings offer the tone model and the switch', (tester) async {
    await show(tester, SpeakersSettingsScreen(services: s));
    await reveal(tester, find.text('Hear how things are said'));
    expect(find.text('Hear how things are said'), findsOneWidget);
    await reveal(tester, find.text('Download the tone model'));
    expect(find.text('Download the tone model'), findsOneWidget);
    await tester.tap(find.text('Hear how things are said'));
    await tester.pumpAndSettle();
    expect(s.settings.value.hearTone, isFalse);
  });
}
