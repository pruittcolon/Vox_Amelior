import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/time_window.dart';
import 'package:vox_amelior_mobile/data/models.dart';

import 'support/fakes.dart';

void main() {
  // Wednesday 30 Sep 2026, 15:30. This week started Monday 28 Sep.
  final now = DateTime(2026, 9, 30, 15, 30);
  const parser = TimeWindowParser();
  TimeWindow at(String q) => parser.parse(q, now)!.$1;

  test('last week is the previous Monday–Sunday', () {
    final w = at('what did we talk about last week?');
    expect(w.from, DateTime(2026, 9, 21));
    expect(w.to, DateTime(2026, 9, 28));
    expect(w.describe(), 'last week (Mon 21 Sep – Sun 27 Sep)');
  });

  test('this week runs from Monday until now', () {
    final w = at('anything important this week');
    expect(w.from, DateTime(2026, 9, 28));
    expect(w.to.isAfter(now), isTrue);
  });

  test('rolling periods count back from now', () {
    expect(at('in the past week').from, DateTime(2026, 9, 24));
    expect(at('over the last week').from, DateTime(2026, 9, 24));
    expect(at('the last 3 days').from, DateTime(2026, 9, 28));
    expect(at('last few days').from, DateTime(2026, 9, 28));
    expect(at('past 2 hours').from, DateTime(2026, 9, 30, 13, 30));
    expect(at('in the last month').from, DateTime(2026, 9, 1));
  });

  test('days, parts of days and "ago"', () {
    expect(at('yesterday').from, DateTime(2026, 9, 29));
    expect(at('yesterday evening').from, DateTime(2026, 9, 29, 17));
    expect(at('last night').to, DateTime(2026, 9, 30, 5));
    expect(at('this morning').to, DateTime(2026, 9, 30, 12));
    expect(at('day before yesterday').from, DateTime(2026, 9, 28));
    expect(at('3 days ago').from, DateTime(2026, 9, 27));
    expect(at('two weeks ago').from, DateTime(2026, 9, 14));
    expect(at('the week before last').from, DateTime(2026, 9, 14));
  });

  test('weekdays refer to the most recent past one', () {
    expect(at('on Monday').from, DateTime(2026, 9, 28));
    expect(at('last Wednesday').from, DateTime(2026, 9, 23));
    expect(at('on Friday').from, DateTime(2026, 9, 25));
    expect(at('this Tuesday').from, DateTime(2026, 9, 29));
  });

  test('weekends and months', () {
    expect(at('last weekend').from, DateTime(2026, 9, 26));
    expect(at('last weekend').to, DateTime(2026, 9, 28));
    expect(at('this month').from, DateTime(2026, 9, 1));
    expect(at('last month').from, DateTime(2026, 8, 1));
    expect(at('last month').to, DateTime(2026, 9, 1));
  });

  test('calendar dates, never in the future', () {
    expect(at('on September 12').from, DateTime(2026, 9, 12));
    expect(at('the 3rd of March').from, DateTime(2026, 3, 3));
    expect(at('on December 25').from, DateTime(2025, 12, 25));
    expect(at('on the 5th').from, DateTime(2026, 9, 5));
    expect(at('in august').from, DateTime(2026, 8, 1));
  });

  test('no time phrase means no window', () {
    expect(parser.parse('when is the plumber coming', now), isNull);
  });

  group('QueryParser intent', () {
    final people = [
      SpeakerProfile(id: 's', name: 'Sam', embeddingModel: 'f', centroid: voiceprint(2), sampleCount: 1, createdAt: DateTime(2026)),
    ];
    final qp = QueryParser(people: people);

    test('broad questions about a period are overviews', () {
      final q = qp.parse('What did we talk about last week?', now: now);
      expect(q.intent, QueryIntent.overview);
      expect(q.window!.label, 'last week');
      expect(q.keywords, isEmpty);
      expect(qp.parse('summarise this week', now: now).intent, QueryIntent.overview);
      expect(qp.parse('recap of yesterday about the car', now: now).intent, QueryIntent.overview);
      expect(qp.parse('what happened with the car yesterday', now: now).intent, QueryIntent.lookup);
    });

    test('specific questions stay lookups with the window as a filter', () {
      final q = qp.parse('What did Sam say about the plumber last week?', now: now);
      expect(q.intent, QueryIntent.lookup);
      expect(q.speaker?.name, 'Sam');
      expect(q.keywords, ['plumber']);
      expect(q.window!.from, DateTime(2026, 9, 21));
    });

    test('time words never leak into keywords', () {
      expect(qp.parse('dentist appointment on March 3', now: now).keywords, ['dentist', 'appointment']);
      expect(qp.parse('groceries in the last 3 days', now: now).keywords, ['groceries']);
    });
  });
}
