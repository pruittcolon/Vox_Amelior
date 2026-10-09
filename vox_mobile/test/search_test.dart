import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sqlite3/sqlite3.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/retriever.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/search/hybrid_search.dart';
import 'package:vox_amelior_mobile/search/vector_store.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

import 'support/fakes.dart';

const String model = 'test-embedder';

void main() {
  group('storing vectors compactly', () {
    test('shortened to 256 dimensions at unit length', () {
      final full = Float32List.fromList(List.generate(768, (i) => math.sin(i * 0.37)));
      final short = EmbeddingCodec.shorten(full);
      expect(short, hasLength(256));
      expect(short.fold<double>(0, (a, x) => a + x * x), closeTo(1, 1e-5));
      expect(EmbeddingCodec.shorten(Float32List(768)).every((x) => x == 0), isTrue);
    });

    test('8-bit storage keeps similarity within 1%', () {
      final rnd = math.Random(4);
      for (var trial = 0; trial < 50; trial++) {
        final a = EmbeddingCodec.shorten(Float32List.fromList(List.generate(768, (_) => rnd.nextDouble() * 2 - 1)));
        final b = EmbeddingCodec.shorten(Float32List.fromList([for (var i = 0; i < 768; i++) (i < 256 ? a[i] : 0) + (rnd.nextDouble() - 0.5) * 0.08]));
        final q = EmbeddingCodec.quantize(b);
        expect(EmbeddingCodec.score(a, q.bytes, 0, q.scale), closeTo(cosine(a, b), 0.01));
      }
    });
  });

  group('with a database', () {
    late AppDatabase db;
    late TranscriptRepository transcripts;
    late SpeakerRepository speakers;
    late VectorStore store;
    late FakeTextEmbedder fake;
    late InlineEmbedder embedder;
    late SearchIndexer indexer;
    late HybridSearch search;
    late String pruitt;
    late String ericah;
    final t0 = DateTime(2026, 10, 1, 19);

    setUp(() {
      db = AppDatabase.inMemory();
      transcripts = TranscriptRepository(db);
      speakers = SpeakerRepository(db);
      store = VectorStore(db);
      fake = FakeTextEmbedder();
      embedder = InlineEmbedder(fake);
      indexer = SearchIndexer(store: store, embedder: () => embedder, modelId: model, batch: 3, pause: Duration.zero);
      search = HybridSearch(db: db, transcripts: transcripts, vectors: store, embedder: () => embedder, modelId: model);
      pruitt = speakers.create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]).id;
      ericah = speakers.create(name: 'Ericah', embeddingModel: 'm', samples: [voiceprint(2)]).id;
    });
    tearDown(() => db.close());

    SegmentView say(String text, int minute, {String? who, String? emotion, Float32List? embedding}) {
      final s = transcripts.addSegment(
        text: text,
        startedAt: t0.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 3),
        speakerId: who,
        embedding: embedding,
      );
      if (emotion != null) transcripts.setTone(s.id, emotion: emotion);
      return transcripts.segment(s.id)!;
    }

    /// A small household archive.
    void archive() {
      say('We cannot pay the electric bill this month', 0, who: ericah, emotion: 'sad');
      say('Rent is due', 1, who: pruitt);
      say('The puppy chewed my shoe again', 30, who: pruitt, emotion: 'angry');
      say('Can you take him for a walk later', 31, who: ericah);
      say('Yes', 32, who: pruitt);
      say('The dentist appointment is on Tuesday at nine', 90, who: ericah);
    }

    group('what gets embedded', () {
      test('the line itself; short lines with the line before; one-word lines not at all', () {
        expect(VectorStore.documentText('The dentist appointment is on Tuesday at nine', 'x'), 'The dentist appointment is on Tuesday at nine');
        expect(VectorStore.documentText('yes, Tuesday works', 'When is the dentist?'), 'When is the dentist?\nyes, Tuesday works');
        expect(VectorStore.documentText('Yes', 'Are you coming?'), isNull);
        expect(VectorStore.documentText('ok ok', null), 'ok ok');
      });

      test('newest first, never TV voices, and lines already done are skipped', () {
        archive();
        final tv = say('Breaking news about the economy tonight', 120, who: pruitt, embedding: voiceprint(77));
        speakers.markBackground(tv.id);
        final first = store.pending(model, limit: 3);
        expect(first.map((l) => l.text), [
          'The dentist appointment is on Tuesday at nine',
          null, // "Yes" alone
          'Can you take him for a walk later',
        ]);
        store.put(model, [for (final l in first) (l.id, l.text == null ? null : FakeTextEmbedder.vector(l.text!))]);
        expect(store.pending(model, limit: 10), hasLength(3));
        expect(store.progress(model), (done: 3, total: 6));
      });
    });

    group('indexing', () {
      test('embeds every line in batches and reports progress', () async {
        archive();
        final seen = <({int done, int total})>[];
        final sub = indexer.progress.listen(seen.add);
        await indexer.run();
        await sub.cancel();
        expect(store.pending(model), isEmpty);
        expect(seen.last, (done: 6, total: 6));
        expect(seen, hasLength(2), reason: '6 lines in batches of 3');
        expect(fake.seen, hasLength(5), reason: '"Yes" is not embedded');
        expect(fake.seen, everyElement(isNot(startsWith(EmbeddingPrompts.document))), reason: 'the prompt is added by the real model wrapper');
      });

      test('a line whose text changes is embedded again; a deleted one leaves no vector', () async {
        archive();
        await indexer.run();
        final line = transcripts.search(const SegmentQuery(keywords: ['puppy'])).single;
        transcripts.updateSegment(line.id, text: 'The puppy ate my homework', duration: const Duration(seconds: 3), speakerId: pruitt);
        expect(store.pending(model).single.id, line.id);
        await indexer.run();
        expect(store.pending(model), isEmpty);
        transcripts.deleteConversation(line.conversationId);
        expect(db.raw.select('SELECT COUNT(*) AS n FROM segment_vectors WHERE segment_id = ?', [line.id]).first['n'], 0);
      });

      test('without an embedder it does nothing; an embedder failure stops it with the error kept', () async {
        archive();
        final off = SearchIndexer(store: store, embedder: () => null, modelId: model, pause: Duration.zero);
        await off.run();
        expect(store.progress(model).done, 0);
        fake.fail = true;
        await indexer.run();
        expect(indexer.error, isNotNull);
        expect(indexer.isRunning, isFalse);
      });

      test('after a failure it waits before trying again by itself; asked to retry, it goes at once', () async {
        archive();
        var now = DateTime(2026, 10, 1, 20);
        final idx = SearchIndexer(
            store: store, embedder: () => embedder, modelId: model, batch: 3, pause: Duration.zero, clock: () => now);
        final seen = <({int done, int total})>[];
        final sub = idx.progress.listen(seen.add);
        fake.fail = true;
        await idx.run();
        expect(idx.error, isNotNull);
        await Future<void>.delayed(Duration.zero);
        expect(seen, isNotEmpty, reason: 'progress shown on screen learns it stopped');
        fake.fail = false;
        await idx.run(); // a new line was heard
        expect(store.progress(model).done, 0, reason: 'a model that failed is not reloaded with every new line');
        now = now.add(const Duration(minutes: 6));
        await idx.run();
        expect(idx.error, isNull);
        expect(store.pending(model), isEmpty);

        say('The car needs new tyres', 200);
        fake.fail = true;
        await idx.run();
        fake.fail = false;
        await idx.run(retry: true);
        expect(idx.error, isNull);
        expect(store.pending(model), isEmpty);
        await sub.cancel();
        idx.dispose();
      });

      test('switched off and on again in the middle of a batch, it carries on with the new model', () async {
        archive();
        final gate = Completer<void>();
        AsyncEmbedder current = _Held(embedder, gate.future);
        final idx = SearchIndexer(store: store, embedder: () => current, modelId: model, batch: 3, pause: Duration.zero);
        final first = idx.run(); // waiting on the first batch
        idx.stop(); // switched off: the old model is closed…
        current = embedder; // …and on again, with a new one
        await idx.run(); // already running: it carries on
        gate.completeError(StateError('Embedder closed'));
        await first;
        expect(idx.error, isNull, reason: 'a batch cut short by switching off is not a failure');
        expect(store.pending(model), isEmpty);
      });

      test('stopped for good, it does not start again', () async {
        archive();
        indexer.dispose();
        await indexer.run();
        expect(store.progress(model).done, 0);
      });

      test('a new model means everything is embedded again', () async {
        archive();
        await indexer.run();
        expect(store.pending('another-model'), isNotEmpty);
      });
    });

    group('searching', () {
      setUp(() async {
        archive();
        await indexer.run();
      });

      test('words: full-text matches only', () {
        expect(search.words('electric', const SearchFilters()).map((s) => s.text), ['We cannot pay the electric bill this month']);
        expect(search.words('   ', const SearchFilters()), isEmpty);
      });

      test('meaning finds a paraphrase that shares no words', () async {
        final hits = await search.meaning('money', const SearchFilters());
        final texts = transcripts.segmentsByIds([for (final (id, _) in hits) id]).map((s) => s.text).toList();
        expect(texts, contains('We cannot pay the electric bill this month'));
        expect(texts, contains('Rent is due'));
        expect(texts, isNot(contains('The puppy chewed my shoe again')));
        expect(hits.first.$2, greaterThanOrEqualTo(HybridSearch.minSimilarity));
        expect(search.words('money', const SearchFilters()), isEmpty, reason: 'no line contains the word');
      });

      test('filters apply to meaning too: person, tone, period, and never TV', () async {
        Future<List<String>> texts(String q, SearchFilters f) async =>
            transcripts.segmentsByIds([for (final (id, _) in await search.meaning(q, f)) id]).map((s) => s.text).toList();
        expect(await texts('money', SearchFilters(speakerIds: {pruitt})), ['Rent is due']);
        expect(await texts('money', const SearchFilters(emotions: {'sad'})), ['We cannot pay the electric bill this month']);
        expect(await texts('money', SearchFilters(from: t0.add(const Duration(hours: 1)))), isEmpty);
        final tv = say('Money money money, the price is right', 200, who: ericah, embedding: voiceprint(77));
        speakers.markBackground(tv.id);
        await indexer.run();
        expect(await texts('money', const SearchFilters()), isNot(contains('Money money money, the price is right')));
      });

      test('smart search fuses both and says how each line was found', () async {
        final hits = await search.search('dog walk', const SearchFilters());
        final byText = {for (final h in hits) h.segment.text: h};
        final walk = byText['Can you take him for a walk later']!;
        expect(walk.byWords, isTrue);
        expect(walk.byMeaning, isTrue);
        final puppy = byText['The puppy chewed my shoe again']!;
        expect(puppy.byWords, isFalse, reason: '"puppy" is not "dog"');
        expect(puppy.byMeaning, isTrue);
        expect(puppy.similarity, isNotNull);
        expect(hits.first.segment.text, 'Can you take him for a walk later', reason: 'found both ways ranks first');

        final wordsOnly = await search.search('dog walk', const SearchFilters(), mode: SearchMode.words);
        expect(wordsOnly.map((h) => h.segment.text), ['Can you take him for a walk later']);
        final meaningOnly = await search.search('dog walk', const SearchFilters(), mode: SearchMode.meaning);
        expect(meaningOnly.every((h) => !h.byWords), isTrue);
      });

      test('the query runs through the model once, however often its results are refreshed', () async {
        fake.seen.clear();
        await search.search('money', const SearchFilters());
        await search.search('money', const SearchFilters(), mode: SearchMode.meaning);
        say('We pay the rent bill in cash today', 120, who: pruitt);
        await indexer.run();
        final hits = await search.search('money', const SearchFilters());
        expect(fake.seen.where((t) => t == 'money'), hasLength(1));
        expect(hits.map((h) => h.segment.text), contains('We pay the rent bill in cash today'), reason: 'lines embedded since are found');
      });

      test('if the model fails, smart search still answers with words', () async {
        fake.fail = true;
        final hits = await search.search('electric', const SearchFilters());
        expect(hits.single.byWords, isTrue);
        await expectLater(search.search('electric', const SearchFilters(), mode: SearchMode.meaning), throwsStateError);
      });

      test('hints for Gemma are the best lines by meaning, or none when unavailable', () async {
        final hints = await search.hintsFor('are we short of money?');
        expect(transcripts.segmentsByIds(hints).map((s) => s.text), contains('We cannot pay the electric bill this month'));
        fake.fail = true;
        expect(await search.hintsFor('who walks the dog?'), isEmpty);
        final off = HybridSearch(db: db, transcripts: transcripts, vectors: store, embedder: () => null, modelId: model);
        expect(await off.hintsFor('money'), isEmpty);
        expect(off.meaningReady, isFalse);
        expect(search.meaningReady, isTrue);
      });
    });

    test('rank fusion: in both lists beats top of one', () {
      final fused = HybridSearch.fuse([
        [1, 2, 3],
        [4, 2, 5],
      ]);
      expect(fused.first.$1, 2);
      expect(fused.first.$2, closeTo(1 / 62 + 1 / 62, 1e-12));
      expect(fused.map((e) => e.$1).toSet(), {1, 2, 3, 4, 5});
      expect(HybridSearch.fuse([[], []]), isEmpty);
    });

    group('Gemma reads what meaning search found (RAG)', () {
      test('hint lines join the keyword matches, held to the question\'s person and period', () {
        archive();
        final bill = transcripts.search(const SegmentQuery(keywords: ['electric'])).single;
        final rent = transcripts.search(const SegmentQuery(keywords: ['rent'])).single;
        final dentist = transcripts.search(const SegmentQuery(keywords: ['dentist'])).single;
        final r = Retriever(transcripts, contextLines: 0);
        final parser = QueryParser(people: speakers.profiles());
        // No keyword in the question matches; the hints carry it.
        final out = r.retrieve(parser.parse('are we struggling financially', now: t0.add(const Duration(days: 1))), hints: [bill.id, rent.id]);
        expect(out.map((s) => s.id), containsAll([bill.id, rent.id]));
        // Held to the person asked about.
        final hers = r.retrieve(parser.parse('what did Ericah say about finances', now: t0.add(const Duration(days: 1))), hints: [bill.id, rent.id, dentist.id]);
        expect(hers.map((s) => s.id), isNot(contains(rent.id)));
        expect(hers.map((s) => s.id), contains(bill.id));
        // Without hints, as before.
        expect(r.retrieve(parser.parse('are we struggling financially', now: t0.add(const Duration(days: 1)))), isEmpty);
      });

      test('the app hands its hints to the listening service through the request', () {
        final requests = AssistantRequestRepository(db);
        final id = requests.add('are we short of money?', source: RequestSource.app, hints: [3, 1]);
        expect(requests.get(id)!.hintIds, [3, 1]);
        expect(requests.get(requests.add('hello'))!.hintIds, isEmpty);
      });
    });
  });

  test('an older database gains the vector table and keeps every line', () {
    final dir = Directory.systemTemp.createTempSync('vox_v6_');
    addTearDown(() => dir.deleteSync(recursive: true));
    final path = '${dir.path}/vox.db';
    final first = AppDatabase.open(path);
    TranscriptRepository(first).addSegment(text: 'kept line here', startedAt: DateTime(2026, 10, 1), duration: const Duration(seconds: 2));
    first.close();
    final old = sqlite3.open(path)
      ..execute('DROP TRIGGER segment_vectors_text')
      ..execute('DROP TRIGGER segment_vectors_gone')
      ..execute('DROP TABLE segment_vectors')
      ..execute('ALTER TABLE assistant_requests DROP COLUMN hint_ids')
      ..userVersion = 6;
    old.close();

    final upgraded = AppDatabase.open(path);
    addTearDown(upgraded.close);
    expect(upgraded.raw.userVersion, AppDatabase.schemaVersion);
    expect(TranscriptRepository(upgraded).search(const SegmentQuery()).single.text, 'kept line here');
    expect(VectorStore(upgraded).pending(model).single.text, 'kept line here');
  });
}

/// An embedder whose next batch waits for [gate].
class _Held implements AsyncEmbedder {
  _Held(this._inner, this._gate);

  final AsyncEmbedder _inner;
  final Future<void> _gate;

  @override
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task}) async {
    await _gate;
    return _inner.embed(texts, task: task);
  }

  @override
  Future<void> close() async {}
}
