import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sqlite3/sqlite3.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late SpeakerProfile me;
  late SpeakerProfile wife;
  final t0 = DateTime(2026, 10, 3, 20);

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    me = speakers.create(name: 'Me', embeddingModel: 'm', samples: [voiceprint(1)]);
    wife = speakers.create(name: 'Wife', embeddingModel: 'm', samples: [voiceprint(2)]);
  });
  tearDown(() => db.close());

  SegmentView say(String text, int minute, {String? speakerId, String? clusterId, Float32List? embedding}) => transcripts.addSegment(
        text: text,
        startedAt: t0.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 3),
        speakerId: speakerId,
        clusterId: clusterId,
        embedding: embedding,
      );

  /// A movie voice: a guest cluster made from one line.
  SegmentView movieLine(String text, int minute) {
    final seg = say(text, minute, speakerId: me.id, embedding: voiceprint(77));
    speakers.markNotSpeaker(seg.id);
    return transcripts.segment(seg.id)!;
  }

  group('filtering by people', () {
    test('conversations where all picked people talked, newest first', () {
      say('Dinner?', 0, speakerId: me.id);
      say('Sure', 1, speakerId: wife.id); // conversation 1: both
      say('Note to self', 60, speakerId: me.id); // conversation 2: only me
      say('Movie time', 120, speakerId: wife.id);
      say('Popcorn', 121, speakerId: me.id); // conversation 3: both

      final both = transcripts.conversationsWith({me.id, wife.id});
      expect(both.map((c) => c.preview), ['Movie time', 'Dinner?']);
      expect(transcripts.conversationsWith({me.id}), hasLength(3));
      expect(transcripts.conversationsWith({}), isEmpty);
      expect(transcripts.conversationsWith({me.id, wife.id}, before: both.first.startedAt).single.preview, 'Dinner?');
    });

    test('search can be narrowed to some people', () {
      say('pizza tonight', 0, speakerId: me.id);
      say('pizza again?', 1, speakerId: wife.id);
      final hers = transcripts.search(SegmentQuery(keywords: const ['pizza'], speakerIds: {wife.id}));
      expect(hers.map((s) => s.text), ['pizza again?']);
      expect(transcripts.search(const SegmentQuery(keywords: ['pizza'])), hasLength(2));
    });
  });

  group('TV and background voices', () {
    test('marking a voice as background hides it from previews, participants and search', () {
      final tv = movieLine('Previously on the show', 0);
      say('Turn it down', 1, speakerId: me.id);
      say('Okay', 2, speakerId: wife.id);
      expect(tv.clusterId, isNotNull);

      expect(speakers.markBackground(tv.id), isNotNull);
      final lines = transcripts.conversation(tv.conversationId);
      expect(lines.first.background, isTrue);
      expect(lines.skip(1).every((l) => !l.background), isTrue);

      final summary = transcripts.conversationsWith({me.id, wife.id}).single;
      expect(summary.preview, 'Turn it down', reason: 'the movie line is not the preview');
      expect(summary.participants, unorderedEquals(['Me', 'Wife']));
      expect(transcripts.search(const SegmentQuery(keywords: ['previously'])), isEmpty);
      expect(transcripts.search(const SegmentQuery(keywords: ['previously'], includeBackground: true)), hasLength(1));
      expect(speakers.clusters().single.background, isTrue);
    });

    test('a movie line wrongly given to a person is taken off them and hidden', () {
      final seg = say('I am your father', 0, speakerId: wife.id, embedding: voiceprint(88));
      expect(speakers.markBackground(seg.id), isNotNull);
      final after = transcripts.segment(seg.id)!;
      expect(after.speakerId, isNull);
      expect(after.background, isTrue);
    });

    test('a background voice can be shown again', () {
      final tv = movieLine('Breaking news', 0);
      speakers.markBackground(tv.id);
      speakers.setClusterBackground(tv.clusterId!, false);
      expect(transcripts.segment(tv.id)!.background, isFalse);
    });

    test('lines without a voice cannot be marked', () {
      final seg = say('hm', 0);
      expect(speakers.markBackground(seg.id), isNull);
    });

    test('voice keys tell people, guests, imported names and unknown apart', () {
      expect(say('a', 0, speakerId: me.id).voiceKey, 'person:${me.id}');
      expect(movieLine('b', 1).voiceKey, startsWith('guest:'));
      expect(say('c', 2).voiceKey, 'unknown');
    });
  });

  group('import', () {
    test('an export comes back after a reinstall, linked to the same people', () {
      say('Morning', 0, speakerId: me.id);
      say('Coffee?', 1, speakerId: wife.id);
      say('Hello there', 1, clusterId: null);
      say('Later', 30, speakerId: me.id);
      final exported = transcripts.exportText();

      // A fresh install: same people re-enrolled, no transcripts.
      final fresh = AppDatabase.inMemory();
      addTearDown(fresh.close);
      final freshSpeakers = SpeakerRepository(fresh);
      final freshTranscripts = TranscriptRepository(fresh);
      final me2 = freshSpeakers.create(name: 'Me', embeddingModel: 'm', samples: [voiceprint(1)]);
      freshSpeakers.create(name: 'Guest 9', embeddingModel: 'm', samples: [voiceprint(9)]);

      final r = freshTranscripts.importText(exported);
      expect(r.conversations, 2);
      expect(r.lines, 4);
      expect(r.skipped, 0);
      expect(freshTranscripts.exportText(), isNotEmpty);
      final first = freshTranscripts.conversationsWith({me2.id});
      expect(first, hasLength(2));
      final lines = freshTranscripts.conversation(first.last.id);
      expect(lines.map((l) => '${l.speakerLabel}: ${l.text}'), ['Me: Morning', 'Wife: Coffee?', 'Unknown: Hello there']);
      expect(lines[0].speakerId, me2.id, reason: 'linked to the re-enrolled person');
      expect(lines[1].speakerId, isNull);
      expect(lines[1].importedLabel, 'Wife', reason: 'not enrolled yet: name kept as text');
      expect(lines[2].importedLabel, isNull);
      expect(first.last.startedAt, t0.copyWith(microsecond: 0));

      // Importing the same text again adds nothing.
      final again = freshTranscripts.importText(exported);
      expect(again.conversations, 0);
      expect(again.skipped, 2);
      expect(freshTranscripts.count(), 4);
    });

    test('text that is not an export imports nothing', () {
      final r = transcripts.importText('hello\nworld');
      expect(r.conversations, 0);
      expect(transcripts.count(), 0);
    });

    test('imported lines are searchable and live listening starts a new conversation after them', () {
      final r = transcripts.importText('--- 2026-09-01T10:00:00.000 ---\nMe: the plumber comes Tuesday\n');
      expect(r.lines, 1);
      expect(transcripts.search(const SegmentQuery(keywords: ['plumber'])).single.speakerId, me.id);
      final live = say('back to now', 0);
      expect(live.conversationId, isNot(transcripts.search(const SegmentQuery(keywords: ['plumber'])).single.conversationId));
    });
  });

  test('an older database keeps every line when upgraded', () {
    final dir = Directory.systemTemp.createTempSync('vox_v4_');
    addTearDown(() => dir.deleteSync(recursive: true));
    final path = '${dir.path}/vox.db';
    final first = AppDatabase.open(path);
    TranscriptRepository(first).addSegment(text: 'kept', startedAt: t0, duration: const Duration(seconds: 2));
    first.close();
    final old = sqlite3.open(path)
      ..execute('ALTER TABLE segments DROP COLUMN emotion')
      ..execute('ALTER TABLE segments DROP COLUMN sound')
      ..execute('ALTER TABLE segments DROP COLUMN speaker_label')
      ..execute('ALTER TABLE unknown_clusters DROP COLUMN background')
      ..userVersion = 4;
    old.close();

    final upgraded = AppDatabase.open(path);
    addTearDown(upgraded.close);
    expect(upgraded.raw.userVersion, AppDatabase.schemaVersion);
    expect(TranscriptRepository(upgraded).search(const SegmentQuery()).single.text, 'kept');
  });
}
