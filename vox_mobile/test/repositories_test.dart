import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late TranscriptRepository repo;
  late SpeakerRepository speakers;
  final base = DateTime(2026, 5, 4, 9);

  SegmentView add(String text, int minute, {String? speakerId, Float32List? emb}) => repo.addSegment(
        text: text,
        startedAt: base.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 4),
        speakerId: speakerId,
        embedding: emb,
      );

  setUp(() {
    db = AppDatabase.inMemory();
    repo = TranscriptRepository(db);
    speakers = SpeakerRepository(db);
  });
  tearDown(() => db.close());

  group('transcripts', () {
    test('groups segments into conversations by silence gap', () {
      final a = add('breakfast is ready', 0);
      final b = add('coming down now', 2);
      final c = add('did you call the plumber', 30);
      expect(a.conversationId, b.conversationId);
      expect(c.conversationId, isNot(a.conversationId));
      expect(repo.conversations().length, 2);
      expect(repo.conversations().first.segmentCount, 1);
    });

    test('full-text search is stemmed, ranked and case-insensitive', () {
      add('The plumbers are coming Tuesday', 0);
      add('we need milk and eggs', 1);
      add('call the plumber back', 2);
      final hits = repo.search(const SegmentQuery(keywords: ['plumber']));
      expect(hits.length, 2);
      expect(repo.search(const SegmentQuery(keywords: ['MILK'])).single.text, contains('milk'));
      expect(repo.search(const SegmentQuery(keywords: ['nothing'])).isEmpty, isTrue);
    });

    test('search survives punctuation and FTS operators in keywords', () {
      add('milk and eggs', 0);
      expect(() => repo.search(const SegmentQuery(keywords: ['"milk"', 'AND', 'NEAR(', '*', ''])), returnsNormally);
      expect(repo.search(const SegmentQuery(keywords: ['"milk"'])).length, 1);
    });

    test('filters by speaker and time window', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      add('groceries list', 0, speakerId: alex.id);
      add('groceries again', 90);
      final byAlex = repo.search(SegmentQuery(keywords: const ['groceries'], speakerId: alex.id));
      expect(byAlex.single.speakerName, 'Alex');
      final early = repo.search(SegmentQuery(keywords: const ['groceries'], to: base.add(const Duration(minutes: 60))));
      expect(early.length, 1);
      expect(repo.between(base, base.add(const Duration(hours: 3))).length, 2);
    });

    test('around returns neighbours within the same conversation only', () {
      final ids = [for (var i = 0; i < 5; i++) add('line $i', i)];
      add('separate later chat', 120);
      final around = repo.around(ids[2], before: 1, after: 1);
      expect(around.map((s) => s.text), ['line 1', 'line 2', 'line 3']);
      expect(repo.around(ids[4], after: 3).map((s) => s.text), ['line 2', 'line 3', 'line 4']);
    });

    test('recent pages backwards', () {
      for (var i = 0; i < 5; i++) {
        add('m$i', i);
      }
      final page1 = repo.recent(limit: 2);
      final page2 = repo.recent(limit: 2, beforeId: page1.last.id);
      expect(page1.map((s) => s.text), ['m4', 'm3']);
      expect(page2.map((s) => s.text), ['m2', 'm1']);
    });

    test('retention deletes old segments, empty conversations and their index entries', () {
      add('old talk about taxes', 0);
      add('recent talk about taxes', 60 * 24 * 40);
      final removed = repo.deleteOlderThan(base.add(const Duration(days: 10)));
      expect(removed, 1);
      expect(repo.count(), 1);
      expect(repo.conversations().length, 1);
      expect(repo.search(const SegmentQuery(keywords: ['taxes'])).single.text, startsWith('recent'));
    });

    test('deleteAllTranscripts keeps enrolled people but removes everything else', () {
      speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      add('secret plans', 0);
      repo.deleteAllTranscripts();
      expect(repo.count(), 0);
      expect(repo.search(const SegmentQuery(keywords: ['secret'])), isEmpty);
      expect(speakers.profiles().length, 1);
    });

    test('exportText lists speakers per conversation', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      add('hello', 0, speakerId: alex.id);
      add('hi', 1);
      final text = repo.exportText();
      expect(text, contains('Alex: hello'));
      expect(text, contains('Unknown: hi'));
    });
  });

  group('timeline', () {
    test('days, conversations per day with participants, and deletion', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final d1 = DateTime(2026, 5, 4, 9);
      final d2 = DateTime(2026, 5, 5, 20);
      repo.addSegment(text: 'morning', startedAt: d1, duration: const Duration(seconds: 3), speakerId: alex.id);
      repo.addSegment(text: 'reply', startedAt: d1.add(const Duration(minutes: 1)), duration: const Duration(seconds: 3));
      repo.addSegment(text: 'later that day', startedAt: d1.add(const Duration(hours: 5)), duration: const Duration(seconds: 3), speakerId: alex.id);
      final other = repo.addSegment(text: 'next day', startedAt: d2, duration: const Duration(seconds: 3));

      final days = repo.days();
      expect(days.map((d) => d.day), [DateTime(2026, 5, 5), DateTime(2026, 5, 4)]);
      expect(days.last.conversations, 2);
      expect(days.last.segments, 3);

      final convs = repo.conversationsBetween(DateTime(2026, 5, 4), DateTime(2026, 5, 5));
      expect(convs.length, 2);
      expect(convs.last.participants, ['Alex', 'Unknown']);
      expect(convs.last.segmentCount, 2);

      expect(repo.segmentsByIds([other.id]).single.text, 'next day');
      expect(repo.segmentsByIds(const []), isEmpty);

      repo.deleteConversation(other.conversationId);
      expect(repo.days().length, 1);
      expect(repo.search(const SegmentQuery(keywords: ['next'])), isEmpty);
    });
  });

  group('speakers', () {
    test('create validates input and rejects duplicate names case-insensitively', () {
      expect(() => speakers.create(name: ' ', embeddingModel: 'f', samples: [voiceprint(1)]), throwsArgumentError);
      expect(() => speakers.create(name: 'A', embeddingModel: 'f', samples: []), throwsArgumentError);
      speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      expect(() => speakers.create(name: 'ALEX', embeddingModel: 'f', samples: [voiceprint(2)]), throwsStateError);
    });

    test('rename enforces uniqueness; delete leaves history unattributed', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final s = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
      expect(() => speakers.rename(s.id, 'alex'), throwsStateError);
      speakers.rename(s.id, 'Samantha');
      expect(speakers.profile(s.id)!.name, 'Samantha');
      add('hi there', 0, speakerId: a.id);
      speakers.delete(a.id);
      expect(repo.recent().single.speakerId, isNull);
    });

    test('guest labels increment and clusters persist', () {
      expect(speakers.nextGuestLabel(), 'Guest 1');
      speakers.saveCluster(UnknownCluster(
        id: 'c1', label: 'Guest 1', centroid: voiceprint(5), count: 3, updatedAt: base,
      ));
      expect(speakers.nextGuestLabel(), 'Guest 2');
      final c = speakers.clusters().single;
      c.count = 9;
      speakers.saveCluster(c);
      expect(speakers.clusters().single.count, 9);
    });

    test('correcting one segment teaches the profile', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final seg = add('that was Alex', 0, emb: voiceprint(1, variant: 3));
      speakers.assignSegmentToSpeaker(seg.id, a.id);
      expect(repo.segment(seg.id)!.speakerName, 'Alex');
      expect(speakers.profile(a.id)!.sampleCount, 2);
    });

    test('promoting a guest creates a person from their past speech and relabels it', () {
      final seg = repo.addSegment(
        text: 'hi from a guest',
        startedAt: base,
        duration: const Duration(seconds: 3),
        clusterId: null,
        embedding: voiceprint(5, variant: 1),
      );
      speakers.saveCluster(UnknownCluster(id: 'g1', label: 'Guest 1', centroid: voiceprint(5), count: 1, updatedAt: base));
      db.raw.execute('UPDATE segments SET cluster_id = ? WHERE id = ?', ['g1', seg.id]);
      final jo = speakers.promoteCluster('g1', 'Jo', embeddingModel: 'f');
      expect(jo.sampleCount, 1);
      expect(repo.segment(seg.id)!.speakerName, 'Jo');
      expect(speakers.clusters(), isEmpty);
      expect(() => speakers.promoteCluster('missing', 'X', embeddingModel: 'f'), throwsStateError);
    });

    test('sample history is capped', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      speakers.addSamples(a.id, [for (var i = 0; i < SpeakerRepository.maxSamplesPerSpeaker + 20; i++) voiceprint(1, variant: i)]);
      expect(speakers.sampleCount(a.id), SpeakerRepository.maxSamplesPerSpeaker);
    });
  });
}
