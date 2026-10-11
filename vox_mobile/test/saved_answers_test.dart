import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/app/assistant_client.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/service/service_controller.dart';
import 'package:vox_amelior_mobile/ui/saved_answers_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

import 'support/fakes.dart';

/// Gemma's answers are kept in a list, however they were asked.
void main() {
  late AppDatabase db;
  late TranscriptRepository transcripts;
  late SpeakerRepository speakers;
  late AssistantRequestRepository requests;
  late FakeLlm llm;
  late AssistantClient client;

  setUp(() {
    db = AppDatabase.inMemory();
    transcripts = TranscriptRepository(db);
    speakers = SpeakerRepository(db);
    requests = AssistantRequestRepository(db);
    llm = FakeLlm()..tools = false;
    transcripts.addSegment(
      text: 'the plumber is coming at four',
      startedAt: DateTime.now().subtract(const Duration(hours: 1)),
      duration: const Duration(seconds: 3),
    );
    client = AssistantClient(
      service: ServiceController(),
      requests: requests,
      transcripts: transcripts,
      local: AssistantService(
        llm: llm,
        transcripts: transcripts,
        speakers: speakers,
        toolbox: toolboxFor(db, transcripts, speakers),
        budget: () => const ContextBudget(4096),
      ),
    );
  });
  tearDown(() => db.close());

  group('answers given in the app are kept', () {
    test('a finished answer is saved with its question and sources', () async {
      llm.reply = 'The plumber comes at four.';
      await client.ask('When is the plumber coming?').drain<void>();
      final saved = requests.recent().single;
      expect(saved.text, 'When is the plumber coming?');
      expect(saved.answer, 'The plumber comes at four.');
      expect(saved.status, RequestStatus.answered);
      expect(saved.source, RequestSource.app);
      expect(saved.sourceSegmentIds, isNotEmpty);
      expect(requests.takePending(), isEmpty, reason: 'never left waiting for the service to answer again');
    });

    test('a stopped answer is not saved', () async {
      llm.reply = List.filled(200, 'word').join(' ');
      // Stop at the first word, like the Stop button.
      final stopped = Completer<void>();
      late StreamSubscription<AnswerEvent> sub;
      sub = client.ask('Tell me everything').listen((e) {
        if (e.token != null && !stopped.isCompleted) stopped.complete(sub.cancel());
      });
      await stopped.future;
      await Future<void>.delayed(const Duration(milliseconds: 20));
      expect(requests.recent(), isEmpty);
    });

    test('a failed answer is not saved', () async {
      llm.failSends = 5;
      llm.canRecover = false;
      await expectLater(client.ask('When is the plumber coming?').drain<void>(), throwsA(anything));
      expect(requests.recent(), isEmpty);
    });
  });

  group('Saved answers screen', () {
    final copied = <String>[];

    Future<void> show(WidgetTester tester) async {
      tester.view
        ..physicalSize = const Size(1080, 2280)
        ..devicePixelRatio = 3;
      addTearDown(tester.view.reset);
      tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(SystemChannels.platform, (call) async {
        if (call.method == 'Clipboard.setData') copied.add((call.arguments as Map)['text'] as String);
        return null;
      });
      addTearDown(() => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(SystemChannels.platform, null));
      await tester.pumpWidget(MaterialApp(
        theme: VoxTheme.light(),
        builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context).copyWith(textScaler: const TextScaler.linear(1.3)),
          child: child!,
        ),
        home: SavedAnswersScreen(requests: requests),
      ));
      await tester.pumpAndSettle();
    }

    setUp(() {
      copied.clear();
      requests.answer(requests.add('When is the plumber coming?', source: RequestSource.app), 'At four, Sam said.');
      requests.answer(requests.add('Did we argue about the bills?'), 'Yes, twice this week: about the electric bill.');
      requests.fail(requests.add('What is the meaning of life?', source: RequestSource.app), 'Gemma is not downloaded');
    });

    testWidgets('lists every answer, newest first, with how it was asked', (tester) async {
      await show(tester);
      expect(find.text('What is the meaning of life?'), findsOneWidget);
      expect(find.text('Did we argue about the bills?'), findsOneWidget);
      expect(find.byIcon(Icons.mic_rounded), findsOneWidget);
      expect(find.byIcon(Icons.keyboard_rounded), findsNWidgets(2));
      expect(
        tester.getTopLeft(find.text('What is the meaning of life?')).dy,
        lessThan(tester.getTopLeft(find.text('When is the plumber coming?')).dy),
      );
    });

    testWidgets('search and filters narrow the list', (tester) async {
      await show(tester);
      await tester.enterText(find.byType(TextField), 'electric');
      await tester.pumpAndSettle();
      expect(find.text('Did we argue about the bills?'), findsOneWidget);
      expect(find.text('When is the plumber coming?'), findsNothing);

      await tester.enterText(find.byType(TextField), '');
      await tester.tap(find.text('Out loud'));
      await tester.pumpAndSettle();
      expect(find.text('Did we argue about the bills?'), findsOneWidget);
      expect(find.text('When is the plumber coming?'), findsNothing);

      await tester.tap(find.text('Typed'));
      await tester.pumpAndSettle();
      expect(find.text('Did we argue about the bills?'), findsNothing);
      expect(find.text('When is the plumber coming?'), findsOneWidget);

      await tester.enterText(find.byType(TextField), 'zebra');
      await tester.pumpAndSettle();
      expect(find.text('No matches'), findsOneWidget);
    });

    testWidgets('tap to open an answer, select it and copy it', (tester) async {
      await show(tester);
      expect(find.byType(SelectableText), findsNothing);
      await tester.tap(find.text('When is the plumber coming?'));
      await tester.pumpAndSettle();
      expect(find.byType(SelectableText), findsOneWidget);
      await tester.ensureVisible(find.text('Copy'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Copy'));
      await tester.pumpAndSettle();
      expect(copied.single, 'Q: When is the plumber coming?\nA: At four, Sam said.');
    });

    testWidgets('nothing asked yet', (tester) async {
      db.raw.execute('DELETE FROM assistant_requests');
      await show(tester);
      expect(find.text('No answers yet'), findsOneWidget);
    });
  });
}
