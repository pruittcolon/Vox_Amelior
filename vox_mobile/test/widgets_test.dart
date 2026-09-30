import 'dart:async';

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

  testWidgets('AskScreen offers questions for a chosen period and a way to go through all of it', (tester) async {
    String? reviewed;
    await tester.pumpWidget(host(AskScreen(
      ask: (_) => const Stream.empty(),
      requests: AssistantRequestRepository(db),
      assistantReady: () => true,
      reviewsBuilder: (_) => const Center(child: Text('REVIEWS LIST')),
      onReviewPeriod: (p) => reviewed = p,
    )));
    await tester.tap(find.text('Last week'));
    await tester.pumpAndSettle();
    expect(find.text('What did we decide last week?'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Go through everything from last week'), 200, scrollable: listScroll);
    await tester.tap(find.text('Go through everything from last week'));
    expect(reviewed, 'last week');

    await tester.tap(find.text('Go through all'));
    await tester.pumpAndSettle();
    expect(find.text('REVIEWS LIST'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
  });

  testWidgets('AskScreen explains failures plainly, retries, and can stop an answer', (tester) async {
    var calls = 0;
    StreamController<AnswerEvent>? pending;
    Stream<AnswerEvent> ask(String q) {
      calls++;
      if (calls == 1) return Stream.error(Exception('Failed to start streaming (code: 13)'));
      if (calls == 2) return Stream.fromIterable([const AnswerEvent.token('All good now.')]);
      pending = StreamController<AnswerEvent>();
      return pending!.stream;
    }

    await tester.pumpWidget(host(AskScreen(ask: ask, requests: AssistantRequestRepository(db), assistantReady: () => true)));
    await tester.enterText(find.byType(TextField), 'hi');
    await tester.tap(find.byIcon(Icons.arrow_upward_rounded));
    await tester.pumpAndSettle();
    expect(find.textContaining('Test this phone'), findsOneWidget);
    expect(find.textContaining('code: 13'), findsNothing);

    await tester.tap(find.text('Try again'));
    await tester.pumpAndSettle();
    expect(find.text('All good now.'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'long one');
    await tester.tap(find.byIcon(Icons.arrow_upward_rounded));
    await tester.pump();
    expect(pending!.hasListener, isTrue);
    await tester.tap(find.byIcon(Icons.stop_rounded));
    await tester.pumpAndSettle();
    expect(pending!.hasListener, isFalse);
    expect(find.text('Stopped'), findsOneWidget);
    expect(find.byIcon(Icons.arrow_upward_rounded), findsOneWidget);
  });
}
