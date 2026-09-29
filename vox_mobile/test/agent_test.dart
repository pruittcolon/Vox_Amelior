import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  final now = DateTime(2026, 9, 30, 15, 30); // Wednesday
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late FakeLlm llm;
  String? ranAutomation;
  int? pausedMinutes;

  AgentToolbox toolbox() => toolboxFor(
        db,
        transcripts,
        speakers,
        clock: () => now,
        hooks: AgentHooks(
          runAutomation: (n) async => ranAutomation = n,
          pauseListening: (m) async => pausedMinutes = m,
        ),
      );

  AssistantService service({bool agent = true}) => AssistantService(
        llm: llm,
        transcripts: transcripts,
        speakers: speakers,
        toolbox: toolbox(),
        agentEnabled: () => agent,
        instructions: () => 'Answer like a pirate.',
        clock: () => now,
      );

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    llm = FakeLlm();
    ranAutomation = null;
    pausedMinutes = null;
    final sam = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
    transcripts.addSegment(text: 'the boiler service is booked for friday', startedAt: DateTime(2026, 9, 22, 10), duration: const Duration(seconds: 3), speakerId: sam.id);
    transcripts.addSegment(text: 'we need oat milk', startedAt: DateTime(2026, 9, 29, 18), duration: const Duration(seconds: 2), speakerId: sam.id);
  });
  tearDown(() => db.close());

  test('the model can search, get results back, then answer', () async {
    llm.script.addAll([
      const LlmToolCall('search_conversations', {'query': 'boiler', 'period': 'last week'}),
      'It is booked for Friday.',
    ]);
    final events = await service().ask('When is the boiler service?').toList();
    expect(events.where((e) => e.toolName == 'search_conversations').length, 1);
    expect(events.map((e) => e.token ?? '').join().trim(), 'It is booked for Friday.');
    final result = llm.toolResults.single;
    expect(result['period'], 'last week (Mon 21 Sep – Sun 27 Sep)');
    expect('${result['results']}', contains('boiler service is booked'));
    expect(llm.lastTools.map((t) => t.name), containsAll(['search_conversations', 'get_timeline', 'set_reminder', 'run_automation']));
    expect(llm.lastSystem, contains('Answer like a pirate.'));
    expect(llm.lastSystem, contains('You can call tools'));
  });

  test('reminders, notes, automations and pausing are real actions', () async {
    llm.script.addAll([
      const LlmToolCall('set_reminder', {'text': 'check the oven', 'minutes_from_now': 30}),
      const LlmToolCall('save_note', {'text': 'buy oat milk'}),
      const LlmToolCall('run_automation', {'name': 'Lights off'}),
      const LlmToolCall('pause_listening', {'minutes': 15}),
      'Done.',
    ]);
    await service().answer('remind me, note it, lights off, and stop listening for a bit');
    expect(ReminderRepository(db, clock: () => now.add(const Duration(minutes: 31))).takeDue().single.text, 'check the oven');
    expect(NoteRepository(db).all().single.text, 'buy oat milk');
    expect(ranAutomation, 'Lights off');
    expect(pausedMinutes, 15);
  });

  test('tool loops are capped', () async {
    for (var i = 0; i < 10; i++) {
      llm.script.add(const LlmToolCall('list_notes', {}));
    }
    llm.script.add('Final answer.');
    final events = await service().ask('loop please').toList();
    expect(events.where((e) => e.toolName != null).length, lessThanOrEqualTo(5));
    expect(llm.toolResults.last['note'], contains('No more tool calls'));
  });

  test('bad tool calls return errors to the model instead of crashing', () async {
    final tb = toolbox();
    expect((await tb.call('nope', {}))['error'], contains('unknown tool'));
    expect((await tb.call('set_reminder', {'text': 'x', 'minutes_from_now': -5}))['ok'], isFalse);
    expect((await tb.call('save_note', {'text': '  '}))['ok'], isFalse);
    expect((await tb.call('get_timeline', {'period': 'whenever'}))['error'], isNotNull);
  });

  test('timeline tool reads the requested period', () async {
    final r = await toolbox().call('get_timeline', {'period': 'yesterday'});
    expect(r['period'], startsWith('yesterday'));
    expect('${r['lines']}', contains('oat milk'));
    expect('${r['lines']}', isNot(contains('boiler')));
  });

  test('agent mode off, or a model without tools, answers from excerpts only', () async {
    await service(agent: false).answer('boiler?');
    expect(llm.lastTools, isEmpty);
    llm.tools = false;
    await service().answer('boiler?');
    expect(llm.lastTools, isEmpty);
    expect(llm.lastSystem, isNot(contains('You can call tools')));
  });

  test('requests remember their source and sources', () {
    final repo = AssistantRequestRepository(db);
    final id = repo.add('typed question', source: RequestSource.app);
    repo.answer(id, 'an answer', sources: [1, 2]);
    final r = repo.get(id)!;
    expect(r.source, RequestSource.app);
    expect(r.sourceSegmentIds, [1, 2]);
    expect(repo.recent(source: RequestSource.voice), isEmpty);
  });
}
