import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

/// Which lines the statistics are about: a period (open-ended when null) and,
/// optionally, one person. TV / background voices are always left out.
class InsightsScope {
  const InsightsScope({this.from, this.to, this.speakerId});

  final DateTime? from;
  final DateTime? to;
  final String? speakerId;

  /// The same length of time just before this one, for "vs previous" deltas
  /// (null for all time).
  InsightsScope? get previous {
    final a = from, b = to;
    if (a == null || b == null) return null;
    return InsightsScope(from: a.subtract(b.difference(a)), to: a, speakerId: speakerId);
  }
}

/// Headline numbers for a scope.
class Overview {
  const Overview({
    this.conversations = 0,
    this.lines = 0,
    this.talk = Duration.zero,
    this.words = 0,
    this.laughs = 0,
    this.moods = const {},
  });

  final int conversations;
  final int lines;

  /// Total length of the lines (time someone was talking).
  final Duration talk;
  final int words;

  /// Lines with laughter in them.
  final int laughs;

  /// Lines per tone ([Tone.names]), only tones that occur.
  final Map<String, int> moods;

  /// Lines that had a non-neutral tone.
  int get feelingLines => moods.entries.where((e) => e.key != 'neutral').fold(0, (a, e) => a + e.value);

  bool get isEmpty => lines == 0;
}

/// How long one stretch of the mood chart covers.
enum Granularity { day, week, month }

/// Lines per tone in one stretch of time (a day, week or month).
class MoodBucket {
  const MoodBucket(this.start, this.counts, this.lines);

  final DateTime start;

  /// Lines per tone in this stretch (tones that occur).
  final Map<String, int> counts;

  /// All lines in this stretch, with or without a tone.
  final int lines;

  int get feeling => counts.entries.where((e) => e.key != 'neutral').fold(0, (a, e) => a + e.value);
}

/// One person's share of the talking.
class PersonStats {
  const PersonStats({
    required this.id,
    required this.name,
    required this.lines,
    required this.talk,
    required this.words,
    required this.conversations,
    this.moods = const {},
  });

  final String id;
  final String name;
  final int lines;
  final Duration talk;
  final int words;
  final int conversations;
  final Map<String, int> moods;
}

/// Two people and how many conversations they were both in.
class PairStats {
  const PairStats(this.a, this.aName, this.b, this.bName, this.conversations);

  final String a;
  final String aName;
  final String b;
  final String bName;
  final int conversations;
}

/// Ways to pick standout conversations.
enum HighlightKind { heated, laughter, longest }

/// A standout conversation and its score (heated lines, laughs, or minutes).
class Highlight {
  const Highlight(this.conversation, this.value);

  final ConversationSummary conversation;
  final int value;
}

/// Statistics over everything that was said: totals, mood over time, who
/// talks, when, with whom, standout conversations and common words.
class InsightsRepository {
  InsightsRepository(this._db, this._transcripts);

  final AppDatabase _db;
  final TranscriptRepository _transcripts;

  /// Lines with a tone that counts as a heated moment.
  static const Set<String> heated = {'angry', 'disgusted'};

  ({String sql, List<Object?> args}) _from(InsightsScope q) {
    final where = <String>['COALESCE(uc.background, 0) = 0'];
    final args = <Object?>[];
    if (q.from != null) {
      where.add('s.started_at >= ?');
      args.add(q.from!.millisecondsSinceEpoch);
    }
    if (q.to != null) {
      where.add('s.started_at < ?');
      args.add(q.to!.millisecondsSinceEpoch);
    }
    if (q.speakerId != null) {
      where.add('s.speaker_id = ?');
      args.add(q.speakerId);
    }
    return (
      sql: 'FROM segments s LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id WHERE ${where.join(' AND ')}',
      args: args,
    );
  }

  /// Words in a line, counted in SQL (spaces + 1).
  static const String _wordsSql = "LENGTH(TRIM(s.text)) - LENGTH(REPLACE(TRIM(s.text), ' ', '')) + 1";

  Overview overview(InsightsScope q) {
    final f = _from(q);
    final r = _db.raw.select(
      'SELECT COUNT(*) AS n, COUNT(DISTINCT s.conversation_id) AS c, COALESCE(SUM(s.duration_ms), 0) AS ms, '
      'COALESCE(SUM($_wordsSql), 0) AS w, COALESCE(SUM(CASE WHEN s.sound = \'laughter\' THEN 1 ELSE 0 END), 0) AS l ${f.sql}',
      f.args,
    ).first;
    final moods = _db.raw.select('SELECT s.emotion AS e, COUNT(*) AS n ${f.sql} AND s.emotion IS NOT NULL GROUP BY s.emotion', f.args);
    return Overview(
      lines: r['n']! as int,
      conversations: r['c']! as int,
      talk: Duration(milliseconds: r['ms']! as int),
      words: r['w']! as int,
      laughs: r['l']! as int,
      moods: {for (final m in moods) m['e']! as String: m['n']! as int},
    );
  }

  /// The chart's stretch for a span of time: days up to a month, weeks up to
  /// half a year, months beyond.
  static Granularity granularityFor(Duration span) => span.inDays <= 31
      ? Granularity.day
      : span.inDays <= 183
          ? Granularity.week
          : Granularity.month;

  /// Start of the stretch [d] falls in (weeks start on Monday).
  static DateTime bucketStart(DateTime d, Granularity g) => switch (g) {
        Granularity.day => DateTime(d.year, d.month, d.day),
        Granularity.week => DateTime(d.year, d.month, d.day - (d.weekday - 1)),
        Granularity.month => DateTime(d.year, d.month),
      };

  static DateTime _next(DateTime d, Granularity g) => switch (g) {
        Granularity.day => DateTime(d.year, d.month, d.day + 1),
        Granularity.week => DateTime(d.year, d.month, d.day + 7),
        Granularity.month => DateTime(d.year, d.month + 1),
      };

  /// Lines per tone for every stretch from the scope's start (or the first
  /// line) to its end (or [now]), empty stretches included so the time axis
  /// is even. Empty when nothing was said.
  List<MoodBucket> moodOverTime(InsightsScope q, {DateTime? now, Granularity? granularity}) {
    final f = _from(q);
    final rows = _db.raw.select(
      "SELECT date(s.started_at / 1000, 'unixepoch', 'localtime') AS d, s.emotion AS e, COUNT(*) AS n "
      '${f.sql} GROUP BY d, e ORDER BY d',
      f.args,
    );
    if (rows.isEmpty) return const [];
    DateTime day(String iso) {
      final p = iso.split('-').map(int.parse).toList();
      return DateTime(p[0], p[1], p[2]);
    }

    final first = q.from ?? day(rows.first['d']! as String);
    final last = q.to?.subtract(const Duration(milliseconds: 1)) ?? now ?? DateTime.now();
    final g = granularity ?? granularityFor(last.difference(first));
    final counts = <DateTime, Map<String, int>>{};
    final lines = <DateTime, int>{};
    for (final r in rows) {
      final b = bucketStart(day(r['d']! as String), g);
      final n = r['n']! as int;
      lines[b] = (lines[b] ?? 0) + n;
      final e = r['e'] as String?;
      if (e != null) (counts[b] ??= {})[e] = (counts[b]?[e] ?? 0) + n;
    }
    final out = <MoodBucket>[];
    final end = bucketStart(last, g);
    for (var b = bucketStart(first, g); !b.isAfter(end); b = _next(b, g)) {
      out.add(MoodBucket(b, counts[b] ?? const {}, lines[b] ?? 0));
      if (out.length > 400) break; // never an unreadable chart
    }
    return out;
  }

  /// Enrolled people by talk time, most first, with their tones. The scope's
  /// person is ignored (everyone is compared).
  List<PersonStats> people(InsightsScope q) {
    final f = _from(InsightsScope(from: q.from, to: q.to));
    final rows = _db.raw.select(
      'SELECT s.speaker_id AS id, sp.name AS name, COUNT(*) AS n, COALESCE(SUM(s.duration_ms), 0) AS ms, '
      'COALESCE(SUM($_wordsSql), 0) AS w, COUNT(DISTINCT s.conversation_id) AS c '
      '${f.sql.replaceFirst('FROM segments s', 'FROM segments s JOIN speakers sp ON sp.id = s.speaker_id')} '
      'GROUP BY s.speaker_id ORDER BY ms DESC, n DESC',
      f.args,
    );
    final moods = <String, Map<String, int>>{};
    for (final m in _db.raw.select(
      'SELECT s.speaker_id AS id, s.emotion AS e, COUNT(*) AS n ${f.sql} AND s.speaker_id IS NOT NULL AND s.emotion IS NOT NULL '
      'GROUP BY s.speaker_id, s.emotion',
      f.args,
    )) {
      (moods[m['id']! as String] ??= {})[m['e']! as String] = m['n']! as int;
    }
    return [
      for (final r in rows)
        PersonStats(
          id: r['id']! as String,
          name: r['name']! as String,
          lines: r['n']! as int,
          talk: Duration(milliseconds: r['ms']! as int),
          words: r['w']! as int,
          conversations: r['c']! as int,
          moods: moods[r['id']] ?? const {},
        ),
    ];
  }

  /// When each enrolled person was last heard (not as TV / background).
  Map<String, DateTime> lastHeard() => {
        for (final r in _db.raw.select(
          'SELECT s.speaker_id AS id, MAX(s.started_at) AS t FROM segments s '
          'LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id '
          'WHERE s.speaker_id IS NOT NULL AND COALESCE(uc.background, 0) = 0 GROUP BY s.speaker_id',
        ))
          r['id']! as String: DateTime.fromMillisecondsSinceEpoch(r['t']! as int),
      };

  /// Talk time per weekday (Monday first) and hour: 7 × 24 values in seconds.
  List<int> weekHours(InsightsScope q) {
    final f = _from(q);
    final out = List<int>.filled(7 * 24, 0);
    for (final r in _db.raw.select(
      "SELECT CAST(strftime('%w', s.started_at / 1000, 'unixepoch', 'localtime') AS INTEGER) AS wd, "
      "CAST(strftime('%H', s.started_at / 1000, 'unixepoch', 'localtime') AS INTEGER) AS h, "
      'COALESCE(SUM(s.duration_ms), 0) AS ms ${f.sql} GROUP BY wd, h',
      f.args,
    )) {
      final monday0 = ((r['wd']! as int) + 6) % 7;
      out[monday0 * 24 + (r['h']! as int)] = ((r['ms']! as int) / 1000).round();
    }
    return out;
  }

  /// Pairs of enrolled people by how many conversations they were both in,
  /// most first. With a person in the scope, only pairs that include them.
  List<PairStats> pairs(InsightsScope q, {int limit = 10}) {
    final f = _from(InsightsScope(from: q.from, to: q.to));
    final rows = _db.raw.select(
      'SELECT DISTINCT s.conversation_id AS c, s.speaker_id AS id, sp.name AS name '
      '${f.sql.replaceFirst('FROM segments s', 'FROM segments s JOIN speakers sp ON sp.id = s.speaker_id')}',
      f.args,
    );
    final byConversation = <int, List<(String, String)>>{};
    for (final r in rows) {
      (byConversation[r['c']! as int] ??= []).add((r['id']! as String, r['name']! as String));
    }
    final counts = <(String, String), int>{};
    final names = <String, String>{};
    for (final people in byConversation.values) {
      people.sort((x, y) => x.$1.compareTo(y.$1));
      for (var i = 0; i < people.length; i++) {
        names[people[i].$1] = people[i].$2;
        for (var j = i + 1; j < people.length; j++) {
          final key = (people[i].$1, people[j].$1);
          counts[key] = (counts[key] ?? 0) + 1;
        }
      }
    }
    final who = q.speakerId;
    final list = [
      for (final e in counts.entries)
        if (who == null || e.key.$1 == who || e.key.$2 == who)
          PairStats(e.key.$1, names[e.key.$1]!, e.key.$2, names[e.key.$2]!, e.value),
    ]..sort((x, y) => y.conversations.compareTo(x.conversations));
    return list.take(limit).toList();
  }

  /// Standout conversations: the most heated (angry or disgusted lines), the
  /// most laughter, or the longest (minutes), best first.
  List<Highlight> highlights(InsightsScope q, HighlightKind kind, {int limit = 5}) {
    final f = _from(q);
    final value = switch (kind) {
      HighlightKind.heated => "SUM(CASE WHEN s.emotion IN ('angry', 'disgusted') THEN 1 ELSE 0 END)",
      HighlightKind.laughter => "SUM(CASE WHEN s.sound = 'laughter' THEN 1 ELSE 0 END)",
      HighlightKind.longest => '(MAX(s.started_at + s.duration_ms) - MIN(s.started_at)) / 60000',
    };
    final rows = _db.raw.select(
      'SELECT s.conversation_id AS id, $value AS v ${f.sql} GROUP BY s.conversation_id HAVING v > 0 '
      'ORDER BY v DESC, s.conversation_id DESC LIMIT ?',
      [...f.args, limit],
    );
    final values = {for (final r in rows) r['id']! as int: (r['v']! as num).toInt()};
    return [for (final c in _transcripts.summariesFor(values.keys.toList())) Highlight(c, values[c.id]!)];
  }

  /// The most used words (common little words left out), most first. Reads
  /// up to [maxLines] of the newest lines in the scope.
  List<(String, int)> topWords(InsightsScope q, {int limit = 24, int maxLines = 20000}) {
    final f = _from(q);
    final counts = <String, int>{};
    final word = RegExp(r"[\p{L}][\p{L}']*", unicode: true);
    for (final r in _db.raw.select('SELECT s.text AS t ${f.sql} ORDER BY s.id DESC LIMIT ?', [...f.args, maxLines])) {
      for (final m in word.allMatches((r['t']! as String).toLowerCase())) {
        var w = m.group(0)!;
        if (w.endsWith("'s")) w = w.substring(0, w.length - 2);
        if (w.length < 3 || stopWords.contains(w) || w.contains("'")) continue;
        counts[w] = (counts[w] ?? 0) + 1;
      }
    }
    final list = counts.entries.where((e) => e.value > 1).toList()
      ..sort((a, b) => b.value != a.value ? b.value.compareTo(a.value) : a.key.compareTo(b.key));
    return [for (final e in list.take(limit)) (e.key, e.value)];
  }

  /// Common English words that say nothing about the topic.
  static const Set<String> stopWords = {
    'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this', 'have', 'from', 'one', 'had',
    'word', 'but', 'not', 'what', 'all', 'were', 'when', 'your', 'can', 'said', 'there', 'use', 'each', 'which',
    'she', 'how', 'their', 'will', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'some', 'her', 'would',
    'make', 'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more', 'write', 'see', 'number', 'way', 'could',
    'people', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part', 'yeah', 'yes', 'okay', 'just', 'know', 'think', 'right', 'really', 'going',
    'gonna', 'well', 'got', 'oh', 'um', 'uh', 'mean', 'something', 'thing', 'things', 'because', 'here', 'where',
    'why', 'too', 'very', 'much', 'any', 'also', 'our', 'over', 'only', 'even', 'back', 'want', 'let', 'say', 'tell',
    'need', 'still', 'though', 'should', 'does', 'doing', 'done', 'being', 'little', 'lot', 'kind', 'sure', 'good',
    'take', 'thank', 'thanks', 'maybe', 'actually', 'stuff', 'guess', 'feel', 'alright', 'hey', 'hmm', 'huh', 'wow',
    'nah', 'yep', 'nope', 'off', 'put', 'went', 'after', 'before', 'again', 'never', 'always', 'every', 'everything',
    'nothing', 'anything', 'someone', 'anyone', 'everyone', 'around', 'through', 'while', 'same', 'those', 'such',
    'own', 'most', 'both', 'few', 'yet', 'ever', 'mine', 'yours', 'theirs', 'ours', 'myself', 'yourself', 'himself',
    'herself', 'itself', 'cause', 'gotta', 'wanna', 'kinda', 'sorta', 'pretty', 'probably', 'today', 'tomorrow',
    'yesterday', 'tonight',
  };
}
