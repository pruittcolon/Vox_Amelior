import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/name_voice.dart';
import 'package:vox_amelior_mobile/ui/people_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

import 'support/fakes.dart';

/// Naming voices on screen, on a small phone with large text: from a
/// conversation, from a line, and one voice after another.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;
  late String pruitt;
  late String guest1;
  late String guest2;
  late int conversation;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_name_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
    pruitt = s.speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]).id;
    s.speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]);
    String guest(String label, int speaker) {
      final c = UnknownCluster(id: SpeakerRepository.newId(), label: label, centroid: voiceprint(speaker), count: 1, updatedAt: DateTime.now());
      s.speakers.saveCluster(c);
      return c.id;
    }

    guest1 = guest('Guest 1', 3);
    guest2 = guest('Guest 2', 4);
    final t = DateTime.now().subtract(const Duration(hours: 3));
    SegmentView say(String text, int seconds, {String? who, String? voice}) => s.transcripts.addSegment(
          text: text,
          startedAt: t.add(Duration(seconds: seconds)),
          duration: const Duration(seconds: 3),
          speakerId: who,
          clusterId: voice,
          embedding: voiceprint(voice == guest2 ? 4 : 3, variant: seconds),
        );
    conversation = say('Did you see the electric bill that came today?', 0, who: pruitt).conversationId;
    say('It is almost twice what we paid last month, I do not know how we will cover it', 10, voice: guest1);
    say('We could stop eating out for a while', 20, who: pruitt);
    say('The rent is due on Friday as well', 30, voice: guest1);
    say('Hello, is anyone home?', 3600, voice: guest2);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> show(WidgetTester tester, Widget screen) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3; // 360 wide, a small phone
    addTearDown(tester.view.reset);
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

  List<SegmentView> linesOf(String voiceOrPerson) =>
      s.transcripts.recent().where((l) => l.clusterId == voiceOrPerson || l.speakerId == voiceOrPerson).toList();

  testWidgets('a conversation offers to name its voice: one tap names every line, Undo takes it back', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversation));
    expect(find.text('A voice without a name'), findsOneWidget);
    expect(find.text('Guest 1 · 2 lines here'), findsOneWidget);

    await tester.tap(find.widgetWithText(FilledButton, "Who's this?"));
    await tester.pumpAndSettle();
    expect(find.text('Who is Guest 1?'), findsOneWidget);
    expect(find.textContaining('almost twice what we paid'), findsWidgets, reason: 'what the voice said, to recognise it');
    expect(find.textContaining('with Pruitt'), findsOneWidget);

    await tester.tap(find.widgetWithText(ActionChip, 'Ericah'));
    await tester.pumpAndSettle();
    expect(linesOf(guest1), isEmpty);
    expect(s.transcripts.recent().where((l) => l.speakerName == 'Ericah'), hasLength(2));
    expect(find.text('A voice without a name'), findsNothing);
    expect(find.text('Guest 1 is Ericah: 2 lines named'), findsOneWidget);

    await tester.tap(find.text('Undo'));
    await tester.pumpAndSettle();
    expect(linesOf(guest1), hasLength(2));
    expect(find.text('A voice without a name'), findsOneWidget);
  });

  testWidgets('someone new: typed once, and the voice gets the name', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversation));
    // The guest's name on its lines is a button too.
    await tester.tap(find.text("Who's this?").last);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Someone new'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'Grandma');
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    final grandma = s.speakers.profiles().singleWhere((p) => p.name == 'Grandma');
    expect(linesOf(grandma.id), hasLength(2));
    expect(find.textContaining('Added Grandma: 2 lines named'), findsOneWidget);
  });

  testWidgets('typing a name that is already in the household picks that person', (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversation));
    await tester.tap(find.widgetWithText(FilledButton, "Who's this?"));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Someone new'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'pruitt ');
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect(s.speakers.profiles(), hasLength(2), reason: 'no second Pruitt');
    expect(linesOf(pruitt), hasLength(4));
  });

  testWidgets("a guest's line leads to naming the whole voice", (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversation));
    await tester.tap(find.textContaining('rent is due'));
    await tester.pumpAndSettle();
    expect(find.text('Who said this?'), findsOneWidget);
    expect(find.textContaining('Guest 1 said 2 lines in all'), findsOneWidget);
    await tester.tap(find.text('Name this voice'));
    await tester.pumpAndSettle();
    expect(find.text('Who is Guest 1?'), findsOneWidget);
  });

  testWidgets("a named person's line: one tap gives just that line to someone else", (tester) async {
    await show(tester, ConversationScreen(services: s, conversationId: conversation));
    await tester.tap(find.textContaining('stop eating out'));
    await tester.pumpAndSettle();
    expect(find.text("Said by Pruitt. If not, it's…"), findsOneWidget);
    expect(find.text('Not Pruitt'), findsOneWidget);
    await tester.tap(find.widgetWithText(ActionChip, 'Ericah'));
    await tester.pumpAndSettle();
    expect(s.transcripts.recent().singleWhere((l) => l.text.contains('stop eating out')).speakerName, 'Ericah');
    expect(linesOf(pruitt), hasLength(1));
  });

  testWidgets('"Who\'s this?" goes through every voice; "Not now" leaves one for later', (tester) async {
    await show(tester, NameVoicesScreen(services: s, openLine: (_, _) {}));
    expect(find.text('2 voices to name'), findsOneWidget);
    expect(find.text('Who is Guest 1?'), findsOneWidget, reason: 'the voice with the most lines first');

    await tester.tap(find.widgetWithText(ActionChip, 'Pruitt'));
    await tester.pumpAndSettle();
    expect(find.text('Who is Guest 2?'), findsOneWidget);
    expect(find.text('Last voice to name'), findsOneWidget);

    await tester.tap(find.text('Not now'));
    await tester.pumpAndSettle();
    expect(find.text('That was every voice for now'), findsOneWidget);
    expect(find.text('Named 1 voice. Thank you!'), findsOneWidget);
    expect(linesOf(guest2), hasLength(1), reason: 'skipped, still there');
  });

  testWidgets('People shows how many voices need a name, and what each said', (tester) async {
    await show(tester, PeopleScreen(services: s));
    expect(find.text('2 voices need a name'), findsOneWidget);
    expect(find.text('VOICES WITHOUT A NAME'), findsOneWidget);
    expect(find.textContaining('almost twice what we paid'), findsOneWidget);
    final button = find.widgetWithText(FilledButton, "Who's this?").first;
    await tester.ensureVisible(button);
    await tester.pumpAndSettle();
    await tester.tap(button);
    await tester.pumpAndSettle();
    expect(find.text('Who is Guest 1?'), findsOneWidget);
  });
}
