import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/insights_repository.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late InsightsRepository insights;
  late SpeakerProfile pruitt;
  late SpeakerProfile ericah;
  late SpeakerProfile sam;
  // Wednesday 30 September 2026.
  final wed = DateTime(2026, 9, 30);

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    insights = InsightsRepository(db, transcripts);
    pruitt = speakers.create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]);
    ericah = speakers.create(name: 'Ericah', embeddingModel: 'm', samples: [voiceprint(2)]);
    sam = speakers.create(name: 'Sam', embeddingModel: 'm', samples: [voiceprint(3)]);
  });
  tearDown(() => db.close());

  SegmentView say(
    String text,
    DateTime at, {
    String? who,
    int seconds = 4,
    String? emotion,
    String? sound,
    Float32List? embedding,
  }) {
    final s = transcripts.addSegment(text: text, startedAt: at, duration: Duration(seconds: seconds), speakerId: who, embedding: embedding);
    if (emotion != null || sound != null) transcripts.setTone(s.id, emotion: emotion, sound: sound);
    return s;
  }

  /// Wednesday evening: Pruitt and Ericah argue, then laugh.
  void evening() {
    final t = wed.add(const Duration(hours: 20));
    say('Did you pay the electric bill', t, who: pruitt.id, emotion: 'neutral');
    say('You never pay the bill on time', t.add(const Duration(seconds: 10)), who: ericah.id, emotion: 'angry', seconds: 6);
    say('That is not fair at all', t.add(const Duration(seconds: 20)), who: pruitt.id, emotion: 'angry');
    say('Okay sorry about the bill', t.add(const Duration(seconds: 30)), who: ericah.id, emotion: 'sad');
    say('Ha ha the dog is on the couch', t.add(const Duration(seconds: 40)), who: pruitt.id, emotion: 'happy', sound: 'laughter');
  }

  group('scope', () {
    test('the previous period has the same length, just before', () {
      final q = InsightsScope(from: DateTime(2026, 10, 1), to: DateTime(2026, 10, 8), speakerId: 'x');
      final p = q.previous!;
      expect(p.from, DateTime(2026, 9, 24));
      expect(p.to, DateTime(2026, 10, 1));
      expect(p.speakerId, 'x');
      expect(const InsightsScope().previous, isNull, reason: 'all time has nothing before it');
    });

    test('chart stretches: days up to a month, weeks up to half a year, then months', () {
      expect(InsightsRepository.granularityFor(const Duration(days: 7)), Granularity.day);
      expect(InsightsRepository.granularityFor(const Duration(days: 31)), Granularity.day);
      expect(InsightsRepository.granularityFor(const Duration(days: 90)), Granularity.week);
      expect(InsightsRepository.granularityFor(const Duration(days: 365)), Granularity.month);
      expect(InsightsRepository.bucketStart(DateTime(2026, 10, 4, 15), Granularity.week), DateTime(2026, 9, 28), reason: 'weeks start Monday');
      expect(InsightsRepository.bucketStart(DateTime(2026, 10, 4, 15), Granularity.month), DateTime(2026, 10));
      expect(InsightsRepository.bucketStart(DateTime(2026, 10, 4, 15), Granularity.day), DateTime(2026, 10, 4));
    });
  });

  group('overview', () {
    test('counts conversations, lines, talk time, words, laughs and tones', () {
      evening();
      say('Morning', wed.add(const Duration(days: 1, hours: 8)), who: sam.id);
      final o = insights.overview(const InsightsScope());
      expect(o.conversations, 2);
      expect(o.lines, 6);
      expect(o.talk, const Duration(seconds: 4 * 5 + 6));
      expect(o.words, 6 + 7 + 6 + 5 + 8 + 1);
      expect(o.laughs, 1);
      expect(o.moods, {'neutral': 1, 'angry': 2, 'sad': 1, 'happy': 1});
      expect(o.feelingLines, 4);
    });

    test('a period and a person narrow it', () {
      evening();
      say('Later that week', wed.add(const Duration(days: 3)), who: pruitt.id);
      final day = insights.overview(InsightsScope(from: wed, to: wed.add(const Duration(days: 1))));
      expect(day.lines, 5);
      final his = insights.overview(InsightsScope(speakerId: pruitt.id));
      expect(his.lines, 4);
      expect(his.moods, {'neutral': 1, 'angry': 1, 'happy': 1});
      expect(insights.overview(InsightsScope(from: wed.add(const Duration(days: 30)))).isEmpty, isTrue);
    });

    test('TV and background voices are never counted', () {
      evening();
      final tv = say('Breaking news tonight', wed.add(const Duration(hours: 20, minutes: 1)), who: pruitt.id, emotion: 'angry', embedding: voiceprint(77));
      speakers.markBackground(tv.id);
      final o = insights.overview(const InsightsScope());
      expect(o.lines, 5);
      expect(o.moods['angry'], 2);
      expect(insights.topWords(const InsightsScope()).map((w) => w.$1), isNot(contains('breaking')));
    });
  });

  group('mood over time', () {
    test('every day of the period gets a column, empty ones included', () {
      evening();
      say('Grumpy morning', wed.add(const Duration(days: 2, hours: 9)), who: sam.id, emotion: 'angry');
      final b = insights.moodOverTime(InsightsScope(from: wed, to: wed.add(const Duration(days: 7))));
      expect(b, hasLength(7));
      expect(b.first.start, wed);
      expect(b[0].counts, {'neutral': 1, 'angry': 2, 'sad': 1, 'happy': 1});
      expect(b[0].lines, 5);
      expect(b[0].feeling, 4);
      expect(b[1].counts, isEmpty);
      expect(b[1].lines, 0);
      expect(b[2].counts, {'angry': 1});
    });

    test('longer periods are grouped by week or month', () {
      say('a', DateTime(2026, 8, 3, 10), who: pruitt.id, emotion: 'happy'); // Monday
      say('b', DateTime(2026, 8, 9, 10), who: pruitt.id, emotion: 'happy'); // Sunday, same week
      say('c', DateTime(2026, 9, 15, 10), who: pruitt.id, emotion: 'sad');
      final weeks = insights.moodOverTime(InsightsScope(from: DateTime(2026, 8, 3), to: DateTime(2026, 10, 3)));
      expect(weeks.first.start, DateTime(2026, 8, 3));
      expect(weeks.first.counts, {'happy': 2});
      expect(weeks[1].start, DateTime(2026, 8, 10));
      // August to March: over half a year, so by month.
      final months = insights.moodOverTime(const InsightsScope(), now: DateTime(2027, 3, 1));
      expect(months.map((m) => m.start), [for (var m = 8; m <= 15; m++) DateTime(2026, m)]);
      expect(months.first.counts, {'happy': 2});
      expect(months[1].counts, {'sad': 1});
    });

    test('nothing said gives no columns', () {
      expect(insights.moodOverTime(const InsightsScope()), isEmpty);
    });
  });

  group('people', () {
    test('ranked by talk time, with their tones', () {
      evening();
      say('Long story from Sam', wed.add(const Duration(days: 1)), who: sam.id, seconds: 60);
      final p = insights.people(const InsightsScope());
      expect(p.map((x) => x.name), ['Sam', 'Pruitt', 'Ericah']);
      final e = p.last;
      expect(e.lines, 2);
      expect(e.talk, const Duration(seconds: 10));
      expect(e.conversations, 1);
      expect(e.moods, {'angry': 1, 'sad': 1});
    });

    test('who talks with whom, and with one person in the scope only their pairs', () {
      evening(); // Pruitt + Ericah
      final t = wed.add(const Duration(days: 1));
      say('hi', t, who: pruitt.id);
      say('hello', t.add(const Duration(seconds: 5)), who: sam.id); // Pruitt + Sam
      final t2 = wed.add(const Duration(days: 2));
      say('dinner', t2, who: pruitt.id);
      say('yes', t2.add(const Duration(seconds: 5)), who: ericah.id); // Pruitt + Ericah again
      final all = insights.pairs(const InsightsScope());
      expect(all.first.conversations, 2);
      expect({all.first.aName, all.first.bName}, {'Pruitt', 'Ericah'});
      expect(all, hasLength(2));
      final ericahs = insights.pairs(InsightsScope(speakerId: ericah.id));
      expect(ericahs.single.conversations, 2);
      expect(insights.pairs(InsightsScope(speakerId: sam.id)).single.conversations, 1);
    });

    test('when each person was last heard', () {
      evening();
      final heard = insights.lastHeard();
      expect(heard[pruitt.id], wed.add(const Duration(hours: 20, seconds: 40)));
      expect(heard[ericah.id], wed.add(const Duration(hours: 20, seconds: 30)));
      expect(heard.containsKey(sam.id), isFalse);
    });
  });

  test('talk time lands on the right weekday and hour', () {
    evening(); // Wednesday 20:00
    say('Sunday brunch', DateTime(2026, 10, 4, 11, 30), who: sam.id, seconds: 90);
    final h = insights.weekHours(const InsightsScope());
    expect(h, hasLength(7 * 24));
    expect(h[2 * 24 + 20], 22, reason: 'Wednesday 20:00: 4 + 6 + 4 + 4 + 4 s');
    expect(h[6 * 24 + 11], 90, reason: 'Sunday 11:00');
    expect(h.fold<int>(0, (a, b) => a + b), 112);
  });

  group('standout conversations', () {
    setUp(() {
      evening(); // 2 heated lines, 1 laugh, under a minute
      final calm = wed.add(const Duration(days: 1, hours: 9));
      say('Lovely morning', calm, who: sam.id, sound: 'laughter');
      say('It really is', calm.add(const Duration(minutes: 5)), who: pruitt.id, sound: 'laughter', seconds: 30);
    });

    test('most heated, most laughter and longest', () {
      final heated = insights.highlights(const InsightsScope(), HighlightKind.heated);
      expect(heated.single.value, 2);
      expect(heated.single.conversation.preview, 'Did you pay the electric bill');
      final laughs = insights.highlights(const InsightsScope(), HighlightKind.laughter);
      expect(laughs.map((h) => h.value), [2, 1]);
      final longest = insights.highlights(const InsightsScope(), HighlightKind.longest);
      expect(longest.single.value, 5, reason: 'only conversations of a minute or more; 5 min 30 s');
    });

    test('summaries keep the order asked for', () {
      final ids = transcripts.conversations().map((c) => c.id).toList();
      expect(transcripts.summariesFor(ids.reversed.toList()).map((c) => c.id), ids.reversed);
      expect(transcripts.summariesFor([...ids, 999]), hasLength(ids.length));
      expect(transcripts.summariesFor(const []), isEmpty);
    });
  });

  test('most said words leave out little words and one-offs', () {
    evening();
    final words = insights.topWords(const InsightsScope());
    expect(words.first, ('bill', 3));
    final all = words.map((w) => w.$1).toList();
    expect(all, isNot(contains('the')));
    expect(all, isNot(contains('you')));
    expect(all, isNot(contains('couch')), reason: 'said once');
    final his = insights.topWords(InsightsScope(speakerId: ericah.id));
    expect(his.map((w) => w.$1), ['bill']);
  });
}
