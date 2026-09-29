import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/ui/ask_screen.dart';
import 'package:vox_amelior_mobile/ui/rule_editor_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

import 'support/fakes.dart';

Widget host(Widget child) => MaterialApp(theme: VoxTheme.light(), home: child);

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
    await tester.tap(find.text('Save'));
    await tester.pumpAndSettle();
    expect((saved!.actions.last as WebhookAction).allowInsecureHttp, isTrue);
  });

  testWidgets('AskScreen streams an answer with its sources and tool activity', (tester) async {
    final sam = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
    transcripts.addSegment(
      text: 'the plumber is coming at four',
      startedAt: DateTime.now().subtract(const Duration(hours: 1)),
      duration: const Duration(seconds: 3),
      speakerId: sam.id,
    );
    final llm = FakeLlm()
      ..script.addAll([
        const LlmToolCall('search_conversations', {'query': 'plumber'}),
        'Sam said the plumber comes at four.',
      ]);
    final service = AssistantService(
      llm: llm,
      transcripts: transcripts,
      speakers: speakers,
      toolbox: toolboxFor(db, transcripts, speakers),
    );
    await tester.pumpWidget(host(AskScreen(ask: service.ask, requests: AssistantRequestRepository(db), assistantReady: () => true)));
    expect(find.text('Summarise today'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'When is the plumber coming?');
    await tester.tap(find.byIcon(Icons.arrow_upward_rounded));
    await tester.pumpAndSettle();
    expect(find.text('When is the plumber coming?'), findsOneWidget);
    expect(find.text('Sam said the plumber comes at four.'), findsOneWidget);
    expect(find.text('Searched conversations'), findsOneWidget);
    expect(find.text('Based on 1 thing said'), findsOneWidget);
  });

  testWidgets('AskScreen shows a clear message when the model is missing', (tester) async {
    final llm = FakeLlm()..unavailable = true;
    final service = AssistantService(llm: llm, transcripts: transcripts, speakers: speakers);
    await tester.pumpWidget(host(AskScreen(ask: service.ask, requests: AssistantRequestRepository(db), assistantReady: () => false)));
    expect(find.textContaining('Download Gemma 4'), findsOneWidget);
    await tester.enterText(find.byType(TextField), 'anything?');
    await tester.tap(find.byIcon(Icons.arrow_upward_rounded));
    await tester.pumpAndSettle();
    expect(find.text('Gemma is not downloaded'), findsOneWidget);
  });
}
