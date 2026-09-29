import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/ui/ask_screen.dart';
import 'package:vox_amelior_mobile/ui/rule_editor_screen.dart';
import 'package:vox_amelior_mobile/ui/transcript_list.dart';

import 'support/fakes.dart';

Widget host(Widget child) => MaterialApp(home: child);

/// The page's main list (text fields have their own inner scrollables).
Finder get listScroll => find.descendant(of: find.byType(ListView), matching: find.byType(Scrollable)).first;

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
  });
  tearDown(() => db.close());

  testWidgets('TranscriptList shows speakers, text and an empty state', (tester) async {
    await tester.pumpWidget(host(const Scaffold(body: TranscriptList(segments: []))));
    expect(find.text('Nothing heard yet.'), findsOneWidget);

    final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
    transcripts.addSegment(text: 'dinner at seven', startedAt: DateTime.now(), duration: const Duration(seconds: 2), speakerId: alex.id);
    transcripts.addSegment(text: 'sounds good', startedAt: DateTime.now(), duration: const Duration(seconds: 2));
    SegmentTapped? tapped;
    await tester.pumpWidget(host(Scaffold(
      body: TranscriptList(segments: transcripts.recent(), onTap: (s) => tapped = SegmentTapped(s.text)),
    )));
    expect(find.text('Alex'), findsOneWidget);
    expect(find.text('Unknown'), findsOneWidget);
    expect(find.text('dinner at seven'), findsOneWidget);
    expect(find.text('Today'), findsOneWidget);
    await tester.tap(find.text('sounds good'));
    expect(tapped?.text, 'sounds good');
  });

  testWidgets('RuleEditor blocks invalid rules and saves valid ones', (tester) async {
    AutomationRule? saved;
    await tester.pumpWidget(host(RuleEditorScreen(people: const ['Alex'], onSave: (r) => saved = r)));

    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect(saved, isNull);
    expect(find.textContaining('Give the rule a name'), findsOneWidget);
    expect(find.textContaining('fire on everything'), findsOneWidget);

    await tester.enterText(find.byKey(const Key('rule-name')), 'Shopping');
    await tester.enterText(find.byKey(const Key('rule-phrases')), 'shopping list, groceries');
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(saved!.name, 'Shopping');
    expect(saved!.trigger.phrases, ['shopping list', 'groceries']);
    expect(saved!.actions.single, isA<NotifyAction>());
  });

  testWidgets('RuleEditor webhook action requires https unless allowed', (tester) async {
    AutomationRule? saved;
    await tester.pumpWidget(host(RuleEditorScreen(people: const [], onSave: (r) => saved = r)));
    await tester.enterText(find.byKey(const Key('rule-name')), 'Lights');
    await tester.enterText(find.byKey(const Key('rule-phrases')), 'lights on');
    await tester.tap(find.byIcon(Icons.add));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Call a webhook'));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.byKey(const Key('webhook-url-1')), 200, scrollable: listScroll);
    await tester.enterText(find.byKey(const Key('webhook-url-1')), 'http://192.168.1.2:8123/api/webhook/x');
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect(saved, isNull);
    await tester.scrollUntilVisible(find.textContaining('Plain http:// is blocked'), -200, scrollable: listScroll);
    expect(find.textContaining('Plain http:// is blocked'), findsOneWidget);

    final allowHttp = find.widgetWithText(SwitchListTile, 'Allow plain http:// (home network devices)');
    await tester.scrollUntilVisible(allowHttp, 200, scrollable: listScroll);
    await tester.ensureVisible(allowHttp);
    await tester.pumpAndSettle();
    await tester.tap(find.descendant(of: allowHttp, matching: find.byType(Switch)));
    await tester.pumpAndSettle();
    expect(tester.widget<SwitchListTile>(allowHttp).value, isTrue);
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect((saved!.actions.last as WebhookAction).allowInsecureHttp, isTrue);
  });

  testWidgets('AskScreen streams an answer with its sources', (tester) async {
    final sam = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
    transcripts.addSegment(
      text: 'the plumber is coming at four',
      startedAt: DateTime.now().subtract(const Duration(hours: 1)),
      duration: const Duration(seconds: 3),
      speakerId: sam.id,
    );
    final llm = FakeLlm();
    await tester.pumpWidget(host(AskScreen(
      assistant: AssistantService(llm: llm, transcripts: transcripts, speakers: speakers),
      requests: AssistantRequestRepository(db),
      assistantReady: () => true,
    )));
    expect(find.text('Summarise today'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'When is the plumber coming?');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pumpAndSettle();
    expect(find.text('When is the plumber coming?'), findsOneWidget);
    expect(find.text('Sam said the plumber comes at four.'), findsOneWidget);
    expect(find.text('Based on 1 things said'), findsOneWidget);
  });

  testWidgets('AskScreen shows a clear message when the model is missing', (tester) async {
    final llm = FakeLlm()..unavailable = true;
    await tester.pumpWidget(host(AskScreen(
      assistant: AssistantService(llm: llm, transcripts: transcripts, speakers: speakers),
      requests: AssistantRequestRepository(db),
      assistantReady: () => false,
    )));
    expect(find.textContaining('Download Gemma'), findsOneWidget);
    await tester.enterText(find.byType(TextField), 'anything?');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pumpAndSettle();
    expect(find.text('Gemma is not downloaded'), findsOneWidget);
  });
}

class SegmentTapped {
  SegmentTapped(this.text);
  final String text;
}
