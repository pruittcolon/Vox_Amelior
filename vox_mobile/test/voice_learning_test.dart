import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';
import 'package:vox_amelior_mobile/speakers/voice_patterns.dart';

import 'support/fakes.dart';

/// A voice heard in a certain situation: the person's voice plus a shared
/// "room" colouring (0 = close to the phone, 1 = across the kitchen...).
Float32List heard(int speaker, int situation, int variant, {double room = 0.7}) {
  final v = voiceprint(speaker, variant: variant);
  if (situation == 0) return v;
  final r = voiceprint(500 + situation, noise: 0);
  return Float32List.fromList([for (var i = 0; i < v.length; i++) v[i] + room * r[i]]);
}

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  final t0 = DateTime(2026, 9, 28, 9);

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
  });
  tearDown(() => db.close());

  SegmentView line(String text, Float32List e, {String? speakerId, int minute = 0}) => transcripts.addSegment(
        text: text,
        startedAt: t0.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 3),
        speakerId: speakerId,
        embedding: e,
      );

  SpeakerIdentifier identifier({bool patterns = true}) => SpeakerIdentifier(
        profiles: speakers.profiles(),
        config: IdentifierConfig(usePatterns: patterns),
        newClusterId: SpeakerRepository.newId,
        nextGuestLabel: speakers.nextGuestLabel,
      );

  group('voice patterns', () {
    test('need at least 10 similar samples per group; few samples give none', () {
      expect(voicePatterns([for (var i = 0; i < 19; i++) heard(1, 0, i)]), isEmpty);
    });

    test('separate situations become separate patterns, deterministically', () {
      final samples = [
        for (var i = 0; i < 15; i++) heard(1, 0, i),
        for (var i = 0; i < 15; i++) heard(1, 1, 100 + i),
      ];
      final a = voicePatterns(samples);
      final b = voicePatterns(samples);
      expect(a.length, 2);
      for (var i = 0; i < a.length; i++) {
        expect(cosine(a[i], b[i]), closeTo(1, 1e-6));
      }
      // Each situation is best described by its own pattern.
      final close = heard(1, 0, 999);
      final far = heard(1, 1, 998);
      final closeBest = cosine(close, a[0]) > cosine(close, a[1]) ? 0 : 1;
      final farBest = cosine(far, a[0]) > cosine(far, a[1]) ? 0 : 1;
      expect(closeBest, isNot(farBest));
    });

    test('are stored with the person and reloaded', () {
      final p = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [
        for (var i = 0; i < 12; i++) heard(1, 0, i),
        for (var i = 0; i < 12; i++) heard(1, 1, 50 + i),
      ]);
      expect(p.patterns.length, 2);
      expect(speakers.profiles().single.patterns.length, 2);
      expect(patternsFromBlob(patternsToBlob(p.patterns), p.centroid.length).length, 2);
    });

    test('help recognise a person in an unusual situation', () {
      speakers.create(name: 'Alex', embeddingModel: 'f', samples: [
        for (var i = 0; i < 30; i++) heard(1, 0, i),
        for (var i = 0; i < 12; i++) heard(1, 2, 200 + i, room: 1.2),
      ]);
      final alexId = speakers.profiles().single.id;
      final withPatterns = identifier();
      final far = heard(1, 2, 777, room: 1.2);
      final score = withPatterns.scoreFor(speakers.profiles().single, far);
      expect(score, greaterThan(cosine(far, speakers.profiles().single.centroid)));
      expect(withPatterns.identify(far, seconds: 3).speakerId, alexId);
    });
  });

  group('corrections', () {
    test('fixing the same line again moves its sample instead of copying it', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final b = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
      final seg = line('hello', voiceprint(2, variant: 5));
      speakers.assignSegmentToSpeaker(seg.id, a.id); // wrong tap
      expect(speakers.sampleCount(a.id), 2);
      speakers.assignSegmentToSpeaker(seg.id, b.id); // fixed
      expect(speakers.sampleCount(a.id), 1);
      expect(speakers.sampleCount(b.id), 2);
      expect(transcripts.segment(seg.id)!.speakerName, 'Sam');
    });

    test('"Not Alex" keeps Alex\'s voiceprint, makes the line a guest and blocks similar voices only', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [for (var i = 0; i < 5; i++) voiceprint(1, variant: i)]);
      final before = speakers.profile(a.id)!.centroid;
      // A visitor who sounds a lot like Alex got Alex's name.
      final visitor = Float32List.fromList([
        for (var i = 0; i < kDim; i++) voiceprint(1, variant: 40)[i] * 0.8 + voiceprint(9, variant: 1)[i] * 0.45,
      ]);
      final seg = line('that was the visitor', visitor, speakerId: a.id);
      final guest = speakers.markNotSpeaker(seg.id);

      expect(guest, 'Guest 1');
      expect(transcripts.segment(seg.id)!.speakerId, isNull);
      expect(transcripts.segment(seg.id)!.clusterLabel, 'Guest 1');
      expect(speakers.negativeCount(a.id), 1);
      expect(cosine(speakers.profile(a.id)!.centroid, before), closeTo(1, 1e-6)); // voiceprint untouched

      final id = identifier();
      final visitorAgain = Float32List.fromList([
        for (var i = 0; i < kDim; i++) voiceprint(1, variant: 41)[i] * 0.8 + voiceprint(9, variant: 2)[i] * 0.45,
      ]);
      expect(id.identify(visitorAgain, seconds: 3).speakerId, isNull, reason: 'the visitor is no longer called Alex');
      for (var v = 60; v < 70; v++) {
        expect(id.identify(voiceprint(1, variant: v), seconds: 3).speakerId, a.id, reason: 'Alex is still Alex');
      }
    });

    test('works for any person, not just one', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final b = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
      speakers.markNotSpeaker(line('x', voiceprint(7), speakerId: a.id).id);
      speakers.markNotSpeaker(line('y', voiceprint(8), speakerId: b.id).id);
      expect(speakers.negativeCount(a.id), 1);
      expect(speakers.negativeCount(b.id), 1);
      speakers.clearNegatives(b.id);
      expect(speakers.negativeCount(b.id), 0);
    });

    test('confirming a line that sounds like a "not" example removes that example', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final wrongNot = line('actually me', voiceprint(1, variant: 3), speakerId: a.id);
      speakers.markNotSpeaker(wrongNot.id);
      expect(speakers.negativeCount(a.id), 1);
      final mine = line('also me', voiceprint(1, variant: 4));
      speakers.assignSegmentToSpeaker(mine.id, a.id);
      expect(speakers.negativeCount(a.id), 0);
    });

    test('re-fixing a "not" line removes its example', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final b = speakers.create(name: 'Sam', embeddingModel: 'f', samples: [voiceprint(2)]);
      final seg = line('who?', voiceprint(2, variant: 2), speakerId: a.id);
      speakers.markNotSpeaker(seg.id);
      speakers.assignSegmentToSpeaker(seg.id, b.id);
      expect(speakers.negativeCount(a.id), 0);
      expect(speakers.sampleCount(b.id), 2);
    });

    test('"New person…" creates someone from one line', () {
      final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final seg = line('hi, I am Jo', voiceprint(3), speakerId: a.id);
      final jo = speakers.createFromSegment(seg.id, 'Jo', embeddingModel: 'f');
      expect(jo.sampleCount, 1);
      expect(transcripts.segment(seg.id)!.speakerName, 'Jo');
      expect(() => speakers.createFromSegment(seg.id, 'alex', embeddingModel: 'f'), throwsStateError);
      final short = transcripts.addSegment(text: 'ok', startedAt: t0, duration: const Duration(milliseconds: 400));
      expect(() => speakers.createFromSegment(short.id, 'Kim', embeddingModel: 'f'), throwsStateError);
    });
  });

  test('replay: patterns never move a line from one person to another', () {
    // Two people, each heard close up and across the room.
    speakers.create(name: 'Alex', embeddingModel: 'f', samples: [
      for (var i = 0; i < 25; i++) heard(1, 0, i),
      for (var i = 0; i < 12; i++) heard(1, 1, 300 + i),
    ]);
    speakers.create(name: 'Sam', embeddingModel: 'f', samples: [
      for (var i = 0; i < 25; i++) heard(2, 0, i),
      for (var i = 0; i < 12; i++) heard(2, 1, 300 + i),
    ]);
    final names = {for (final p in speakers.profiles()) p.id: p.name};
    final before = identifier(patterns: false);
    final after = identifier();
    var recognisedBefore = 0;
    var recognisedAfter = 0;
    for (final speaker in [1, 2, 7]) {
      for (final situation in [0, 1]) {
        for (var v = 1000; v < 1040; v++) {
          final e = heard(speaker, situation, v);
          final old = before.identify(e, seconds: 3).speakerId;
          final now = after.identify(e, seconds: 3).speakerId;
          if (old != null && now != null) {
            expect(now, old, reason: 'switched ${names[old]} → ${names[now]}');
          }
          if (old != null) recognisedBefore++;
          if (now != null) recognisedAfter++;
        }
      }
    }
    expect(recognisedAfter, greaterThanOrEqualTo(recognisedBefore));
  });

  test('people enrolled before patterns existed get them on the next start', () {
    final a = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [
      for (var i = 0; i < 12; i++) heard(1, 0, i),
      for (var i = 0; i < 12; i++) heard(1, 1, 50 + i),
    ]);
    db.raw.execute('UPDATE speakers SET patterns = NULL WHERE id = ?', [a.id]); // as after the upgrade
    expect(speakers.profile(a.id)!.patterns, isEmpty);
    expect(speakers.ensurePatterns(), 1);
    expect(speakers.profile(a.id)!.patterns.length, 2);
    expect(speakers.ensurePatterns(), 0);
  });
}
