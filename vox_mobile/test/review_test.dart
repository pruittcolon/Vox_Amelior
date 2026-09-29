import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/context_probe.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/assistant/review_worker.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late TranscriptRepository transcripts;
  late SpeakerRepository speakers;
  late ReviewRepository reviews;
  late FakeLlm llm;
  late ReviewEngine engine;
  final monday = DateTime(2026, 9, 21);
  late String alex;
  late String sam;

  setUp(() {
    db = AppDatabase.inMemory();
    transcripts = TranscriptRepository(db);
    speakers = SpeakerRepository(db);
    reviews = ReviewRepository(db);
    llm = FakeLlm();
    engine = ReviewEngine(reviews: reviews, transcripts: transcripts, llm: llm);
    alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]).id;
    sam = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]).id;
    // A week of conversation: 7 days × 30 lines, alternating speakers.
    for (var d = 0; d < 7; d++) {
      for (var i = 0; i < 30; i++) {
        transcripts.addSegment(
          text: 'On day $d line $i we talked about the garden, the budget and who takes the bins out tonight',
          startedAt: monday.add(Duration(days: d, hours: 18, minutes: i)),
          duration: const Duration(seconds: 4),
          speakerId: i.isEven ? alex : sam,
        );
      }
    }
  });
  tearDown(() => db.close());

  Future<void> runToEnd(int id) async {
    for (var guard = 0; guard < 500; guard++) {
      if (!await engine.step(id)) return;
    }
    fail('review did not finish');
  }

  int start(ReviewTemplate t, {int chunk = 600, List<String> focus = const []}) => engine.start(
        title: t.name,
        prompt: t.prompt,
        format: t.format,
        kind: t.kind,
        periodLabel: 'last week',
        from: monday,
        to: monday.add(const Duration(days: 7)),
        budget: ContextBudget(4096, chunkTokens: chunk),
        focus: focus,
      )!;

  group('parsing findings', () {
    test('reads the fixed format and its usual variations', () {
      final found = ReviewEngine.parseFindings('''
Here is what I found:
- [3] "You always forget - every single time" — Hasty generalisation — one case made into always
1. [#7] you never listen — ad hominem. — attacks the person
- [Line 12] "fine" — Strawman
* "no number here" — Appeal to emotion — guilt
NONE
''');
      expect(found.length, 4);
      expect(found[0].line, 3);
      expect(found[0].quote, 'You always forget - every single time');
      expect(found[0].category, 'Hasty generalisation');
      expect(found[0].note, 'one case made into always');
      expect(found[1].line, 7);
      expect(found[1].category, 'Ad hominem');
      expect(found[2].line, 12);
      expect(found[2].note, '');
      expect(found[3].line, isNull);
      expect(ReviewEngine.parseFindings('NONE'), isEmpty);
      expect(ReviewEngine.parseFindings('none.'), isEmpty);
    });

    test('plans parts of about the requested size', () {
      final segs = transcripts.between(monday, monday.add(const Duration(days: 7)), limit: 10000);
      final parts = ReviewEngine.plan(segs, 600);
      expect(parts.expand((x) => x).length, segs.length);
      expect(parts.length, greaterThan(3));
      for (final part in parts) {
        final tokens = part.fold<int>(0, (a, s) => a + ContextBudget.estimateTokens('[99] 18:00 Alex: ${s.text}') + 1);
        expect(tokens, lessThanOrEqualTo(600));
      }
    });
  });

  test('a list review reads every part, counts findings exactly and links them to lines', () async {
    llm.responder = (prompt) {
      if (prompt.contains('Write a short overview')) return 'Mostly hasty generalisations from both.';
      return '- [1] "first line" — Hasty generalisation — too broad\n- [2] "second line" — Strawman — misrepresents';
    };
    final id = start(ReviewTemplate.builtIns.first);
    final parts = reviews.run(id)!.totalChunks;
    await runToEnd(id);

    final run = reviews.run(id)!;
    expect(run.status, ReviewStatus.done);
    expect(run.doneChunks, run.totalChunks + 1);
    final items = reviews.items(id);
    expect(items.length, parts * 2);
    expect(ReviewEngine.countByCategory(items), {'Hasty generalisation': parts, 'Strawman': parts});
    expect(run.finalAnswer, contains('${parts * 2} found'));
    expect(run.finalAnswer, contains('Mostly hasty generalisations'));
    // Line [1] of the first part is the week's first line.
    final first = transcripts.segment(items.first.segmentId!)!;
    expect(first.text, startsWith('On day 0 line 0'));
    expect(items.first.speaker, 'Alex');
    // Later parts were told what was already found, and saw the format.
    expect(llm.prompts[1], contains('Found in earlier parts'));
    expect(llm.prompts[1], contains('Hasty generalisation ×1'));
    expect(llm.prompts[2], contains('Hasty generalisation ×2'));
    expect(llm.prompts.first, contains('Answer format:'));
    expect(llm.prompts.first, contains('part 1 of $parts'));
  });

  test('focusing on one person drops findings from others', () async {
    llm.responder = (prompt) => prompt.contains('overview') ? 'ok' : '- [1] "a" — X — y\n- [2] "b" — X — y';
    final id = start(ReviewTemplate.builtIns.first, focus: [alex]);
    await runToEnd(id);
    final items = reviews.items(id);
    expect(items, isNotEmpty);
    expect(items.every((i) => i.speaker == 'Alex'), isTrue);
    expect(llm.prompts.first, contains('Only report things said by: Alex'));
  });

  test('a summary review merges the parts in rounds into one answer', () async {
    llm.responder = (prompt) => prompt.contains('Combine them') ? 'MERGED' : 'Talked about the garden.';
    final summary = ReviewTemplate.builtIns.firstWhere((t) => t.kind == ReviewKind.summary);
    final id = start(summary, chunk: 512);
    await runToEnd(id);
    final run = reviews.run(id)!;
    expect(run.status, ReviewStatus.done);
    expect(run.finalAnswer, 'MERGED');
    expect(llm.prompts.where((p) => p.contains('Notes from earlier parts')).length, greaterThan(0));
  });

  test('a failing part is recorded and skipped; retry reads it again', () async {
    llm
      ..canRecover = false
      ..failSends = 1
      ..responder = (prompt) => prompt.contains('overview') ? 'ok' : 'NONE';
    final id = start(ReviewTemplate.builtIns.first);
    await runToEnd(id);
    final run = reviews.run(id)!;
    expect(run.status, ReviewStatus.done);
    expect(reviews.chunks(id).where((c) => c.status == 'failed').length, 1);
    reviews.retryFailed(id);
    expect(reviews.run(id)!.status, ReviewStatus.queued);
    await runToEnd(id);
    expect(reviews.chunks(id).where((c) => c.status == 'failed'), isEmpty);
  });

  test('a missing model pauses the review with a clear reason', () async {
    llm.unavailable = true;
    final id = start(ReviewTemplate.builtIns.first);
    expect(await engine.step(id), isFalse);
    final run = reviews.run(id)!;
    expect(run.status, ReviewStatus.paused);
    expect(run.error, contains('not downloaded'));
  });

  test('nothing said in the period: no review', () {
    final none = engine.start(
      title: 'x',
      prompt: 'x',
      format: 'x',
      kind: ReviewKind.list,
      periodLabel: 'x',
      from: DateTime(2020),
      to: DateTime(2020, 2),
      budget: const ContextBudget(4096),
    );
    expect(none, isNull);
  });

  group('workers', () {
    test('only one worker holds a review; an expired lease is taken over', () async {
      var now = DateTime(2026, 9, 30, 12);
      final repo = ReviewRepository(db, clock: () => now);
      final id = start(ReviewTemplate.builtIns.first);
      expect(repo.claim(id, 'service'), isTrue);
      expect(repo.claim(id, 'app'), isFalse);
      expect(repo.nextRunnable('app'), isNull);
      now = now.add(const Duration(minutes: 5)); // the service died
      expect(repo.nextRunnable('app')!.id, id);
      expect(repo.claim(id, 'app'), isTrue);
    });

    test('a worker runs a review to the end, and stops at pause', () async {
      llm.responder = (prompt) => prompt.contains('overview') ? 'ok' : 'NONE';
      final id = start(ReviewTemplate.builtIns.first);
      var steps = 0;
      final worker = ReviewWorker(
        engine: engine,
        reviews: reviews,
        owner: 'app',
        schedule: (step) {
          steps++;
          if (steps == 2) reviews.setStatus(id, ReviewStatus.paused);
          return step();
        },
      );
      worker.kick();
      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(reviews.run(id)!.status, ReviewStatus.paused);
      reviews.setStatus(id, ReviewStatus.queued);
      worker.kick();
      for (var i = 0; i < 100 && reviews.run(id)!.status != ReviewStatus.done; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
      expect(reviews.run(id)!.status, ReviewStatus.done);
    });
  });

  group('phone context test', () {
    late Directory tmp;
    setUp(() => tmp = Directory.systemTemp.createTempSync('vox_probe_'));
    tearDown(() => tmp.deleteSync(recursive: true));

    String echoCode(String prompt) => RegExp(r'code word: (\S+)').firstMatch(prompt)?.group(1) ?? '?';

    test('finds the largest working size and recommends half of it', () async {
      llm
        ..contextLimit = 8192
        ..responder = echoCode;
      final marker = File(p.join(tmp.path, 'marker.json'));
      final steps = <String>[];
      final r = await ContextProbe(llm: llm, markerFile: marker).run(onStep: (s) => steps.add('${s.size}:${s.status.name}'));
      expect(r.best, 8192);
      expect(r.failedAt, 16384);
      expect(r.recommendedChunk, 4096);
      expect(r.note, contains('8,192'));
      expect(steps, containsAllInOrder(['2048:passed', '4096:passed', '8192:passed', '16384:failed']));
      expect(marker.existsSync(), isFalse);
      // The filler really filled the window (the prompt is close to the size).
      final big = llm.prompts.firstWhere((x) => x.length > 8192 * 3);
      expect(big.length, lessThan(8192 * 4));
    });

    test('a model that drops text (wrong code word) fails that size', () async {
      llm.responder = (prompt) => prompt.length > 10000 ? 'I do not know' : echoCode(prompt);
      final r = await ContextProbe(llm: llm, markerFile: File(p.join(tmp.path, 'm.json'))).run();
      expect(r.best, 2048);
      expect(r.failedAt, 4096);
    });

    test('a crash during the test is remembered at the next start', () {
      final marker = File(p.join(tmp.path, 'marker.json'))
        ..writeAsStringSync(jsonEncode({'pid': -1, 'testing': 16384, 'passed': [2048, 4096, 8192]}));
      final r = ContextProbe.resultAfterCrash(marker, processId: 42)!;
      expect(r.best, 8192);
      expect(r.failedAt, 16384);
      expect(r.note, contains('closed'));
      expect(marker.existsSync(), isFalse);
      // A test still running in this same app is left alone.
      marker.writeAsStringSync(jsonEncode({'pid': 42, 'testing': 4096, 'passed': [2048]}));
      expect(ContextProbe.resultAfterCrash(marker, processId: 42), isNull);
    });
  });
}
