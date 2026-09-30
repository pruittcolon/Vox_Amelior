
import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/retriever.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  // Wednesday 2026-06-10, 15:30.
  final now = DateTime(2026, 6, 10, 15, 30);
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late SpeakerProfile alex;
  late SpeakerProfile sam;

  SegmentView say(String text, DateTime at, {String? speaker}) => transcripts.addSegment(
        text: text,
        startedAt: at,
        duration: const Duration(seconds: 3),
        speakerId: speaker,
      );

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
    sam = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
  });
  tearDown(() => db.close());

  group('QueryParser', () {
    QueryParser parser() => QueryParser(people: speakers.profiles());

    test('extracts speaker, keywords and drops filler words', () {
      final q = parser().parse('What did Sam say about the plumber?', now: now);
      expect(q.speaker?.name, 'Sam');
      expect(q.keywords, ['plumber']);
    });

    test('understands common time phrases', () {
      final y = parser().parse('what did we decide yesterday about the car', now: now);
      expect(y.from, DateTime(2026, 6, 9));
      expect(y.to, DateTime(2026, 6, 10));
      expect(y.keywords, ['decide', 'car']);
      expect(parser().parse('anything today?', now: now).from, DateTime(2026, 6, 10));
      expect(parser().parse('in the last 2 hours', now: now).from, DateTime(2026, 6, 10, 13, 30));
      expect(parser().parse('last week', now: now).from, DateTime(2026, 6, 1)); // previous Mon–Sun
      expect(parser().parse('this morning', now: now).to, DateTime(2026, 6, 10, 12));
      final mon = parser().parse('what happened on monday', now: now);
      expect(mon.from, DateTime(2026, 6, 8));
      expect(mon.timeLabel, 'Monday');
    });

    test('a weekday equal to today refers to last week, not today', () {
      final q = parser().parse('what did alex say on wednesday', now: now);
      expect(q.from, DateTime(2026, 6, 3));
    });

    test('possessives and case are handled; unknown names are just keywords', () {
      expect(parser().parse("Alex's idea about the garden", now: now).speaker?.name, 'Alex');
      final q = parser().parse('what did Jordan say', now: now);
      expect(q.speaker, isNull);
      expect(q.keywords, ['jordan']);
    });
  });

  group('Retriever', () {
    test('returns hits with surrounding lines, chronologically, without duplicates', () {
      final t = DateTime(2026, 6, 10, 9);
      final ids = [
        say('are we all set for today', t, speaker: alex.id),
        say('the plumber is coming at four', t.add(const Duration(minutes: 1)), speaker: sam.id),
        say('great thanks for arranging that', t.add(const Duration(minutes: 2)), speaker: alex.id),
        say('anyway what is for dinner', t.add(const Duration(minutes: 3)), speaker: alex.id),
      ];
      final q = QueryParser(people: speakers.profiles()).parse('when is the plumber coming', now: now);
      final out = Retriever(transcripts, contextLines: 1).retrieve(q);
      expect(out.map((s) => s.id), [ids[0].id, ids[1].id, ids[2].id]);
    });

    test('speaker and time filters narrow results', () {
      final t = DateTime(2026, 6, 9, 10);
      say('plumber tomorrow says alex', t, speaker: alex.id);
      say('plumber tomorrow says sam', t.add(const Duration(hours: 1)), speaker: sam.id);
      final q = QueryParser(people: speakers.profiles()).parse('what did Sam say yesterday about the plumber', now: now);
      final out = Retriever(transcripts, contextLines: 0).retrieve(q);
      expect(out.single.text, contains('says sam'));
    });

    test('falls back to the time window when keywords are paraphrased', () {
      say('we should book the boiler service', DateTime(2026, 6, 9, 10), speaker: sam.id);
      final q = QueryParser(people: speakers.profiles()).parse('what did Sam say yesterday about heating', now: now);
      expect(Retriever(transcripts, contextLines: 0).retrieve(q).length, 1);
    });

    test('returns nothing when nothing matches', () {
      say('hello', DateTime(2026, 6, 9, 10));
      final q = QueryParser(people: speakers.profiles()).parse('tell me about submarines', now: now);
      expect(Retriever(transcripts).retrieve(q), isEmpty);
    });

    test('respects the character budget but keeps direct hits', () {
      final t = DateTime(2026, 6, 10, 9);
      for (var i = 0; i < 30; i++) {
        say('filler sentence number $i ${'blah ' * 20}', t.add(Duration(seconds: i * 5)));
      }
      say('the secret code word is pineapple', t.add(const Duration(seconds: 80)));
      for (var i = 0; i < 30; i++) {
        say('more filler $i ${'blah ' * 20}', t.add(Duration(seconds: 90 + i * 5)));
      }
      final q = QueryParser(people: speakers.profiles()).parse('what is the code word', now: now);
      final out = Retriever(transcripts, contextLines: 25, maxChars: 800).retrieve(q);
      expect(out.fold<int>(0, (a, s) => a + s.text.length), lessThan(1200));
      expect(out.any((s) => s.text.contains('pineapple')), isTrue);
    });
  });

  group('PromptBuilder', () {
    test('includes speakers, times and the question; forbids inventing facts', () {
      final seg = say('the plumber is coming at four', DateTime(2026, 6, 10, 9, 5), speaker: sam.id);
      const b = PromptBuilder();
      final prompt = b.question(question: 'When is the plumber coming?', excerpts: [seg], now: now);
      expect(prompt, contains('[09:05] Sam: the plumber is coming at four'));
      expect(prompt, contains('Question: When is the plumber coming?'));
      final system = b.system(now: now, people: ['Alex', 'Sam']);
      expect(system, contains('never invent'));
      expect(system, contains('Alex, Sam'));
      expect(system, contains('Wednesday 2026-06-10 15:30'));
    });

    test('says so when there are no excerpts', () {
      expect(const PromptBuilder().question(question: 'x?', excerpts: const [], now: now), contains('(none found)'));
    });
  });

  group('AssistantService', () {
    late FakeLlm llm;
    late AssistantService service;

    setUp(() {
      llm = FakeLlm();
      service = AssistantService(llm: llm, transcripts: transcripts, speakers: speakers, clock: () => now);
    });

    test('retrieves excerpts, prompts the model with them, and returns its answer with sources', () async {
      say('the plumber is coming at four', DateTime(2026, 6, 10, 9, 5), speaker: sam.id);
      final answer = await service.answer('When is the plumber coming?');
      expect(answer.text, 'Sam said the plumber comes at four.');
      expect(answer.sources.single.speakerName, 'Sam');
      expect(llm.lastPrompt, contains('Sam: the plumber is coming at four'));
      expect(llm.loaded, isTrue);
    });

    test('streams sources first, then tokens', () async {
      say('the plumber is coming at four', DateTime(2026, 6, 10, 9, 5));
      final events = await service.ask('plumber?').toList();
      expect(events.first.sources, isNotNull);
      expect(events.skip(1).every((e) => e.token != null), isTrue);
    });

    test('blank questions do nothing', () async {
      expect(await service.ask('   ').toList(), isEmpty);
    });

    test('a missing model surfaces a clear error', () async {
      llm.unavailable = true;
      expect(service.answer('anything'), throwsA(isA<LlmUnavailable>()));
    });

    test('overview questions read across the whole period', () async {
      for (var d = 1; d <= 5; d++) {
        say('day $d we talked about topic$d', DateTime(2026, 6, d, 10), speaker: alex.id);
      }
      await service.answer('What did we talk about last week?');
      for (var d = 1; d <= 5; d++) {
        expect(llm.lastPrompt, contains('topic$d'));
      }
      expect(llm.lastPrompt, contains('last week (Mon 1 Jun – Sun 7 Jun)'));
    });
  });
}
