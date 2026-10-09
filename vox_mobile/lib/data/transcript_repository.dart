import 'dart:convert';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

/// Stores what was said, grouped into conversations, and searches it.
class TranscriptRepository {
  TranscriptRepository(this._db, {this.conversationGap = const Duration(minutes: 5)});

  final AppDatabase _db;

  /// Silence longer than this starts a new conversation.
  final Duration conversationGap;

  static const String _selectSegments = '''
SELECT s.id, s.conversation_id, s.started_at, s.duration_ms, s.text,
       s.speaker_id, sp.name AS speaker_name,
       s.cluster_id, c.label AS cluster_label, s.score, s.overlap,
       COALESCE(c.background, 0) AS background, s.speaker_label, s.emotion, s.sound
FROM segments s
LEFT JOIN speakers sp ON sp.id = s.speaker_id
LEFT JOIN unknown_clusters c ON c.id = s.cluster_id''';

  /// First line of a conversation that is not TV / background.
  static const String _previewColumn = '''
(SELECT s.text FROM segments s LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
 WHERE s.conversation_id = c.id AND COALESCE(uc.background, 0) = 0 ORDER BY s.id LIMIT 1) AS preview''';

  /// Saves an utterance, attaching it to the current conversation or opening
  /// a new one when the gap since the last utterance is long enough.
  SegmentView addSegment({
    required String text,
    required DateTime startedAt,
    required Duration duration,
    String? speakerId,
    String? clusterId,
    double? score,
    Float32List? embedding,
    bool overlap = false,
  }) {
    final startMs = startedAt.millisecondsSinceEpoch;
    final endMs = startMs + duration.inMilliseconds;
    late int id;
    _db.transaction(() {
      final last = _db.raw.select(
        'SELECT id, ended_at FROM conversations ORDER BY id DESC LIMIT 1',
      );
      int conversationId;
      if (last.isNotEmpty && startMs - (last.first['ended_at'] as int) <= conversationGap.inMilliseconds) {
        conversationId = last.first['id'] as int;
        _db.raw.execute(
          'UPDATE conversations SET ended_at = MAX(ended_at, ?) WHERE id = ?',
          [endMs, conversationId],
        );
      } else {
        _db.raw.execute(
          'INSERT INTO conversations(started_at, ended_at) VALUES (?, ?)',
          [startMs, endMs],
        );
        conversationId = _db.raw.lastInsertRowId;
      }
      _db.raw.execute(
        'INSERT INTO segments(conversation_id, started_at, duration_ms, text, speaker_id, '
        'cluster_id, score, embedding, overlap) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
          conversationId,
          startMs,
          duration.inMilliseconds,
          text,
          speakerId,
          clusterId,
          score,
          embedding == null ? null : floatsToBlob(embedding),
          overlap ? 1 : 0,
        ],
      );
      id = _db.raw.lastInsertRowId;
    });
    return segment(id)!;
  }

  SegmentView? segment(int id) {
    final rows = _db.raw.select('$_selectSegments WHERE s.id = ?', [id]);
    return rows.isEmpty ? null : _view(rows.first);
  }

  /// Newest-first page of segments. Pass the smallest id already loaded as
  /// [beforeId] to fetch the next (older) page.
  List<SegmentView> recent({int limit = 100, int? beforeId}) {
    final rows = beforeId == null
        ? _db.raw.select('$_selectSegments ORDER BY s.id DESC LIMIT ?', [limit])
        : _db.raw.select('$_selectSegments WHERE s.id < ? ORDER BY s.id DESC LIMIT ?', [beforeId, limit]);
    return rows.map(_view).toList();
  }

  List<SegmentView> conversation(int conversationId) {
    final rows = _db.raw.select(
      '$_selectSegments WHERE s.conversation_id = ? ORDER BY s.id',
      [conversationId],
    );
    return rows.map(_view).toList();
  }

  List<SegmentView> segmentsByIds(List<int> ids) {
    if (ids.isEmpty) return const [];
    final marks = List.filled(ids.length, '?').join(',');
    final rows = _db.raw.select('$_selectSegments WHERE s.id IN ($marks) ORDER BY s.id', ids);
    return rows.map(_view).toList();
  }

  /// Days that have recordings, newest first.
  List<DaySummary> days({int limit = 120}) {
    final rows = _db.raw.select(
      '''
SELECT date(s.started_at / 1000, 'unixepoch', 'localtime') AS d,
       COUNT(DISTINCT s.conversation_id) AS c, COUNT(*) AS n
FROM segments s GROUP BY d ORDER BY d DESC LIMIT ?''',
      [limit],
    );
    return rows.map((r) {
      final parts = (r['d']! as String).split('-').map(int.parse).toList();
      return DaySummary(day: DateTime(parts[0], parts[1], parts[2]), conversations: r['c']! as int, segments: r['n']! as int);
    }).toList();
  }

  /// Conversations that started within [from, to), newest first, with who spoke.
  List<ConversationSummary> conversationsBetween(DateTime from, DateTime to) {
    final rows = _db.raw.select(
      '''
SELECT c.id, c.started_at, c.ended_at,
       (SELECT COUNT(*) FROM segments s WHERE s.conversation_id = c.id) AS n,
       $_previewColumn
FROM conversations c
WHERE c.started_at >= ? AND c.started_at < ?
ORDER BY c.started_at DESC''',
      [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch],
    );
    return rows.map(_summary).where((c) => c.segmentCount > 0).toList();
  }

  /// Who spoke in a conversation, most lines first. TV / background voices
  /// are left out.
  List<String> participants(int conversationId) {
    final rows = _db.raw.select(
      '''
SELECT COALESCE(sp.name, uc.label, s.speaker_label, 'Unknown') AS who, COUNT(*) AS n
FROM segments s
LEFT JOIN speakers sp ON sp.id = s.speaker_id
LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
WHERE s.conversation_id = ? AND COALESCE(uc.background, 0) = 0
GROUP BY who ORDER BY n DESC, MIN(s.id)''',
      [conversationId],
    );
    return [for (final r in rows) r['who']! as String];
  }

  /// Conversations in which every one of [speakerIds] said something, newest
  /// first ("me and my wife"). Pass the oldest start already shown as
  /// [before] for the next page.
  List<ConversationSummary> conversationsWith(Set<String> speakerIds, {int limit = 100, DateTime? before}) {
    if (speakerIds.isEmpty) return const [];
    final marks = List.filled(speakerIds.length, '?').join(',');
    final rows = _db.raw.select(
      '''
SELECT c.id, c.started_at, c.ended_at,
       (SELECT COUNT(*) FROM segments s WHERE s.conversation_id = c.id) AS n,
       $_previewColumn
FROM conversations c
WHERE c.id IN (
  SELECT conversation_id FROM segments WHERE speaker_id IN ($marks)
  GROUP BY conversation_id HAVING COUNT(DISTINCT speaker_id) = ?)
${before == null ? '' : 'AND c.started_at < ?'}
ORDER BY c.started_at DESC LIMIT ?''',
      [...speakerIds, speakerIds.length, ?before?.millisecondsSinceEpoch, limit],
    );
    return rows.map(_summary).toList();
  }

  /// Summaries of conversations [ids], in the order given (missing ones left out).
  List<ConversationSummary> summariesFor(List<int> ids) {
    if (ids.isEmpty) return const [];
    final rows = _db.raw.select(
      '''
SELECT c.id, c.started_at, c.ended_at,
       (SELECT COUNT(*) FROM segments s WHERE s.conversation_id = c.id) AS n,
       $_previewColumn
FROM conversations c
WHERE c.id IN (${List.filled(ids.length, '?').join(',')})''',
      ids,
    );
    final byId = {for (final r in rows) r['id']! as int: _summary(r)};
    return [for (final id in ids) ?byId[id]];
  }

  /// Rewrites a saved line in place (stage 2 cutting it at a speaker change).
  SegmentView updateSegment(
    int id, {
    required String text,
    required Duration duration,
    String? speakerId,
    String? clusterId,
    double? score,
    Float32List? embedding,
    bool overlap = false,
  }) {
    _db.raw.execute(
      'UPDATE segments SET text = ?, duration_ms = ?, speaker_id = ?, cluster_id = ?, score = ?, embedding = ?, overlap = ? '
      'WHERE id = ?',
      [text, duration.inMilliseconds, speakerId, clusterId, score, embedding == null ? null : floatsToBlob(embedding), overlap ? 1 : 0, id],
    );
    return segment(id)!;
  }

  void setOverlap(int id) => _db.raw.execute('UPDATE segments SET overlap = 1 WHERE id = ?', [id]);

  /// Saves the tone of voice and sound heard in line [id] (see [Tone]).
  void setTone(int id, {String? emotion, String? sound}) =>
      _db.raw.execute('UPDATE segments SET emotion = ?, sound = ? WHERE id = ?', [emotion, sound, id]);

  /// How many lines of each tone conversation [conversationId] has (TV /
  /// background voices left out). Empty when the tone model never ran.
  MoodCount mood(int conversationId) {
    final rows = _db.raw.select(
      '''
SELECT s.emotion AS e, COUNT(*) AS n FROM segments s
LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
WHERE s.conversation_id = ? AND s.emotion IS NOT NULL AND COALESCE(uc.background, 0) = 0
GROUP BY s.emotion''',
      [conversationId],
    );
    return MoodCount({for (final r in rows) r['e']! as String: r['n']! as int});
  }

  /// Whether any line has a tone yet (the tone model has run at least once).
  bool get hasTones =>
      _db.raw.select('SELECT 1 FROM segments WHERE emotion IS NOT NULL OR sound IS NOT NULL LIMIT 1').isNotEmpty;

  /// Matches lines whose emotion or sound is one of [n] values (bind them twice).
  static String _toneClause(int n) {
    final marks = List.filled(n, '?').join(',');
    return '(s.emotion IN ($marks) OR s.sound IN ($marks))';
  }

  /// Conversations with at least one line (not TV / background) in any of
  /// [emotions] or sounds, newest first. With [speakerIds], only conversations
  /// where all of them talked, and only their lines count for the tone ("me
  /// and my wife, when someone sounded angry").
  List<ConversationSummary> conversationsWithTone(
    Set<String> emotions, {
    Set<String> speakerIds = const {},
    int limit = 100,
    DateTime? before,
  }) {
    if (emotions.isEmpty) return const [];
    final people = List.filled(speakerIds.length, '?').join(',');
    final rows = _db.raw.select(
      '''
SELECT c.id, c.started_at, c.ended_at,
       (SELECT COUNT(*) FROM segments s WHERE s.conversation_id = c.id) AS n,
       $_previewColumn
FROM conversations c
WHERE c.id IN (
  SELECT s.conversation_id FROM segments s LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
  WHERE ${_toneClause(emotions.length)} AND COALESCE(uc.background, 0) = 0
  ${speakerIds.isEmpty ? '' : 'AND s.speaker_id IN ($people)'})
${speakerIds.length < 2 ? '' : 'AND c.id IN (SELECT conversation_id FROM segments WHERE speaker_id IN ($people) '
        'GROUP BY conversation_id HAVING COUNT(DISTINCT speaker_id) = ?)'}
${before == null ? '' : 'AND c.started_at < ?'}
ORDER BY c.started_at DESC LIMIT ?''',
      [
        ...emotions,
        ...emotions,
        ...speakerIds,
        if (speakerIds.length >= 2) ...[...speakerIds, speakerIds.length],
        ?before?.millisecondsSinceEpoch,
        limit,
      ],
    );
    return rows.map(_summary).toList();
  }

  /// The newest [count] lines said by any of [speakerIds] (everyone when
  /// empty), optionally only in [emotions], returned oldest first. TV /
  /// background voices are always left out. Used to pick lines for a review
  /// ("the last 100 things Pruitt and Ericah said").
  List<SegmentView> lastLines({
    required int count,
    Set<String> speakerIds = const {},
    Set<String> emotions = const {},
    DateTime? from,
    DateTime? to,
  }) {
    final where = <String>['COALESCE(c.background, 0) = 0'];
    final args = <Object?>[];
    if (speakerIds.isNotEmpty) {
      where.add('s.speaker_id IN (${List.filled(speakerIds.length, '?').join(',')})');
      args.addAll(speakerIds);
    }
    if (emotions.isNotEmpty) {
      where.add(_toneClause(emotions.length));
      args.addAll([...emotions, ...emotions]);
    }
    if (from != null) {
      where.add('s.started_at >= ?');
      args.add(from.millisecondsSinceEpoch);
    }
    if (to != null) {
      where.add('s.started_at < ?');
      args.add(to.millisecondsSinceEpoch);
    }
    final rows = _db.raw.select(
      '$_selectSegments WHERE ${where.join(' AND ')} ORDER BY s.started_at DESC, s.id DESC LIMIT ?',
      [...args, count],
    );
    return rows.map(_view).toList().reversed.toList();
  }

  void deleteConversation(int conversationId) {
    _db.transaction(() {
      _db.raw
        ..execute('UPDATE voice_clips SET segment_id = NULL WHERE segment_id IN (SELECT id FROM segments WHERE conversation_id = ?)', [conversationId])
        ..execute('DELETE FROM segments WHERE conversation_id = ?', [conversationId])
        ..execute('DELETE FROM conversations WHERE id = ?', [conversationId]);
    });
  }

  ConversationSummary _summary(Map<String, Object?> r) => ConversationSummary(
        id: r['id']! as int,
        startedAt: DateTime.fromMillisecondsSinceEpoch(r['started_at']! as int),
        endedAt: DateTime.fromMillisecondsSinceEpoch(r['ended_at']! as int),
        segmentCount: r['n']! as int,
        preview: (r['preview'] as String?) ?? '',
        participants: participants(r['id']! as int),
      );

  List<ConversationSummary> conversations({int limit = 50, int? beforeId}) {
    final rows = _db.raw.select(
      '''
SELECT c.id, c.started_at, c.ended_at,
       (SELECT COUNT(*) FROM segments s WHERE s.conversation_id = c.id) AS n,
       $_previewColumn
FROM conversations c
${beforeId == null ? '' : 'WHERE c.id < ?'}
ORDER BY c.id DESC LIMIT ?''',
      [?beforeId, limit],
    );
    return rows.map(_summary).toList();
  }

  /// Full-text search (stemmed, BM25-ranked) with optional speaker/time
  /// filters. With no keywords, returns the newest matching segments.
  List<SegmentView> search(SegmentQuery q) {
    final tokens = q.keywords.map(_sanitizeToken).where((t) => t.isNotEmpty).toSet().toList();
    final where = <String>[];
    final args = <Object?>[];
    final people = {?q.speakerId, ...q.speakerIds};
    if (people.isNotEmpty) {
      where.add('s.speaker_id IN (${List.filled(people.length, '?').join(',')})');
      args.addAll(people);
    }
    if (!q.includeBackground) where.add('COALESCE(c.background, 0) = 0');
    if (q.emotions.isNotEmpty) {
      where.add(_toneClause(q.emotions.length));
      args.addAll([...q.emotions, ...q.emotions]);
    }
    if (q.from != null) {
      where.add('s.started_at >= ?');
      args.add(q.from!.millisecondsSinceEpoch);
    }
    if (q.to != null) {
      where.add('s.started_at < ?');
      args.add(q.to!.millisecondsSinceEpoch);
    }
    final filter = where.isEmpty ? '' : ' AND ${where.join(' AND ')}';

    if (tokens.isEmpty) {
      final clause = where.isEmpty ? '' : 'WHERE ${where.join(' AND ')}';
      final rows = _db.raw.select(
        '$_selectSegments $clause ORDER BY s.id DESC LIMIT ?',
        [...args, q.limit],
      );
      return rows.map(_view).toList();
    }
    final match = tokens.map((t) => '"$t"').join(' OR ');
    final rows = _db.raw.select(
      '$_selectSegments JOIN segments_fts ON segments_fts.rowid = s.id '
      'WHERE segments_fts MATCH ?$filter ORDER BY bm25(segments_fts), s.id DESC LIMIT ?',
      [match, ...args, q.limit],
    );
    return rows.map(_view).toList();
  }

  /// Up to [before]/[after] neighbouring segments of [seg] in its conversation.
  List<SegmentView> around(SegmentView seg, {int before = 2, int after = 2}) {
    final prev = _db.raw.select(
      '$_selectSegments WHERE s.conversation_id = ? AND s.id < ? ORDER BY s.id DESC LIMIT ?',
      [seg.conversationId, seg.id, before],
    );
    final next = _db.raw.select(
      '$_selectSegments WHERE s.conversation_id = ? AND s.id > ? ORDER BY s.id LIMIT ?',
      [seg.conversationId, seg.id, after],
    );
    return [...prev.map(_view).toList().reversed, seg, ...next.map(_view)];
  }

  /// Segments between [from] (inclusive) and [to] (exclusive), oldest first.
  List<SegmentView> between(DateTime from, DateTime to, {int limit = 2000}) {
    final rows = _db.raw.select(
      '$_selectSegments WHERE s.started_at >= ? AND s.started_at < ? ORDER BY s.id LIMIT ?',
      [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch, limit],
    );
    return rows.map(_view).toList();
  }

  int count() => _db.raw.select('SELECT COUNT(*) AS c FROM segments').first['c'] as int;

  /// How many lines, and how many characters of text, were said in a period.
  ({int lines, int chars}) sizeBetween(DateTime from, DateTime to) {
    final r = _db.raw.select(
      'SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(text)), 0) AS c FROM segments WHERE started_at >= ? AND started_at < ?',
      [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch],
    ).first;
    return (lines: r['n']! as int, chars: r['c']! as int);
  }

  /// Deletes everything older than [cutoff]; returns the number of segments removed.
  int deleteOlderThan(DateTime cutoff) {
    late int removed;
    _db.transaction(() {
      _db.raw.execute('DELETE FROM segments WHERE started_at < ?', [cutoff.millisecondsSinceEpoch]);
      removed = _db.raw.updatedRows;
      _pruneEmpty();
      // Saved clips outlive their transcript line (they keep their own text).
      _db.raw.execute('UPDATE voice_clips SET segment_id = NULL WHERE segment_id NOT IN (SELECT id FROM segments)');
    });
    return removed;
  }

  /// Wipes all transcripts and unnamed voices. Enrolled people are kept.
  void deleteAllTranscripts() {
    _db.transaction(() {
      _db.raw
        ..execute('DELETE FROM segments')
        ..execute('DELETE FROM conversations')
        ..execute('DELETE FROM unknown_clusters')
        ..execute('UPDATE voice_clips SET segment_id = NULL');
    });
    // Rebuild removes the FTS shadow content too.
    _db.raw.execute("INSERT INTO segments_fts(segments_fts) VALUES ('rebuild')");
  }

  /// Plain-text export, oldest first.
  String exportText({DateTime? from, DateTime? to}) {
    final rows = between(from ?? DateTime.fromMillisecondsSinceEpoch(0), to ?? DateTime(9999));
    final b = StringBuffer();
    int? lastConversation;
    for (final s in rows) {
      if (s.conversationId != lastConversation) {
        if (lastConversation != null) b.writeln();
        b.writeln('--- ${s.startedAt.toIso8601String()} ---');
        lastConversation = s.conversationId;
      }
      final tone = [?s.emotion, ?s.sound].join(', ');
      b.writeln('${s.speakerLabel}${tone.isEmpty ? '' : ' [$tone]'}: ${s.text}');
    }
    return b.toString();
  }

  /// Reads back text made by [exportText] (or the "Copy all transcripts"
  /// button of any earlier version), so transcripts survive a reinstall.
  ///
  /// Each `--- <time> ---` block becomes a conversation. Names that match a
  /// person on this phone are linked to them; other names are kept as text.
  /// Exact times of lines were not exported, so they are spread out from the
  /// conversation's start by their length. Conversations already here (same
  /// start time) are skipped, so importing twice adds nothing.
  ImportResult importText(String text) {
    final blocks = <({DateTime start, List<({String who, String text, String? emotion, String? sound})> lines})>[];
    final header = RegExp(r'^---\s*(.+?)\s*---$');
    for (final raw in const LineSplitter().convert(text)) {
      final line = raw.trimRight();
      if (line.trim().isEmpty) continue;
      final h = header.firstMatch(line.trim());
      if (h != null) {
        final start = DateTime.tryParse(h.group(1)!);
        if (start != null) blocks.add((start: start, lines: []));
        continue;
      }
      if (blocks.isEmpty) continue;
      final colon = line.indexOf(': ');
      final lines = blocks.last.lines;
      if (colon > 0 && colon <= 80) {
        final (who, emotion, sound) = _splitTone(line.substring(0, colon).trim());
        lines.add((who: who, text: line.substring(colon + 2).trim(), emotion: emotion, sound: sound));
      } else if (lines.isNotEmpty) {
        final l = lines.last;
        lines[lines.length - 1] = (who: l.who, text: '${l.text} ${line.trim()}', emotion: l.emotion, sound: l.sound);
      }
    }

    final people = {
      for (final r in _db.raw.select('SELECT id, name FROM speakers')) (r['name']! as String).toLowerCase(): r['id']! as String,
    };
    var conversations = 0, lines = 0, skipped = 0;
    _db.transaction(() {
      for (final b in blocks) {
        if (b.lines.isEmpty) continue;
        final startMs = b.start.millisecondsSinceEpoch;
        if (_db.raw.select('SELECT 1 FROM conversations WHERE started_at = ? LIMIT 1', [startMs]).isNotEmpty) {
          skipped++;
          continue;
        }
        _db.raw.execute('INSERT INTO conversations(started_at, ended_at) VALUES (?, ?)', [startMs, startMs]);
        final conversationId = _db.raw.lastInsertRowId;
        var at = startMs;
        for (final l in b.lines) {
          final words = l.text.split(RegExp(r'\s+')).length;
          final ms = (words * 400).clamp(1000, 30000);
          final speakerId = people[l.who.toLowerCase()];
          final label = speakerId != null || l.who == 'Unknown' ? null : l.who;
          _db.raw.execute(
            'INSERT INTO segments(conversation_id, started_at, duration_ms, text, speaker_id, speaker_label, emotion, sound) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            [conversationId, at, ms, l.text, speakerId, label, l.emotion, l.sound],
          );
          at += ms;
          lines++;
        }
        _db.raw.execute('UPDATE conversations SET ended_at = ? WHERE id = ?', [at, conversationId]);
        conversations++;
      }
    });
    return ImportResult(conversations: conversations, lines: lines, skipped: skipped);
  }

  /// "Ericah [angry, laughter]" → ('Ericah', 'angry', 'laughter'). Only known
  /// tone words are taken; anything else stays part of the name.
  static (String, String?, String?) _splitTone(String who) {
    final m = RegExp(r'^(.*?)\s*\[([a-z ,]+)\]$').firstMatch(who);
    if (m == null) return (who, null, null);
    final words = m.group(2)!.split(',').map((w) => w.trim()).toList();
    if (words.isEmpty || !words.every((w) => Tone.names.contains(w) || Tone.sounds.contains(w))) return (who, null, null);
    return (
      m.group(1)!.trim(),
      words.where(Tone.names.contains).firstOrNull,
      words.where(Tone.sounds.contains).firstOrNull,
    );
  }

  void _pruneEmpty() {
    _db.raw.execute(
      'DELETE FROM conversations WHERE id NOT IN (SELECT DISTINCT conversation_id FROM segments)',
    );
  }

  static String _sanitizeToken(String raw) => raw.replaceAll(RegExp(r'[^\p{L}\p{N}]', unicode: true), '');

  SegmentView _view(Map<String, Object?> r) => SegmentView(
        id: r['id']! as int,
        conversationId: r['conversation_id']! as int,
        startedAt: DateTime.fromMillisecondsSinceEpoch(r['started_at']! as int),
        duration: Duration(milliseconds: r['duration_ms']! as int),
        text: r['text']! as String,
        speakerId: r['speaker_id'] as String?,
        speakerName: r['speaker_name'] as String?,
        clusterId: r['cluster_id'] as String?,
        clusterLabel: r['cluster_label'] as String?,
        score: (r['score'] as num?)?.toDouble(),
        overlap: (r['overlap'] as int? ?? 0) != 0,
        background: (r['background'] as int? ?? 0) != 0,
        importedLabel: r['speaker_label'] as String?,
        emotion: r['emotion'] as String?,
        sound: r['sound'] as String?,
      );
}

/// What [TranscriptRepository.importText] added.
class ImportResult {
  const ImportResult({required this.conversations, required this.lines, required this.skipped});

  final int conversations;
  final int lines;

  /// Conversations that were already on the phone.
  final int skipped;
}
