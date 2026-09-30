import 'dart:convert';

import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';

enum ReviewStatus { queued, running, paused, done, failed, cancelled }

/// A "go through everything" job over a period of transcripts.
class ReviewRun {
  const ReviewRun({
    required this.id,
    required this.title,
    required this.prompt,
    required this.format,
    required this.kind,
    required this.periodLabel,
    required this.from,
    required this.to,
    required this.focus,
    required this.chunkTokens,
    required this.contextTokens,
    required this.status,
    required this.totalChunks,
    required this.doneChunks,
    required this.createdAt,
    required this.updatedAt,
    this.mergeJson,
    this.finalAnswer,
    this.error,
    this.finishedAt,
  });

  final int id;
  final String title;
  final String prompt;
  final String format;
  final ReviewKind kind;
  final String periodLabel;
  final DateTime from;
  final DateTime to;

  /// Only report things said by these people (empty = everyone).
  final List<String> focus;
  final int chunkTokens;
  final int contextTokens;
  final ReviewStatus status;
  final int totalChunks;
  final int doneChunks;
  final DateTime createdAt;
  final DateTime updatedAt;
  final String? mergeJson;
  final String? finalAnswer;
  final String? error;
  final DateTime? finishedAt;

  bool get isActive => status == ReviewStatus.queued || status == ReviewStatus.running;
  bool get isFinished => status == ReviewStatus.done || status == ReviewStatus.failed || status == ReviewStatus.cancelled;

  /// 0–1, counting the final combining step as one more part.
  double get progress => totalChunks == 0 ? 1 : (doneChunks / (totalChunks + 1)).clamp(0, 1);
}

/// One part of a review: a slice of the transcript and what Gemma found in it.
class ReviewChunk {
  const ReviewChunk({
    required this.runId,
    required this.idx,
    required this.segmentIds,
    required this.lineCount,
    required this.firstAt,
    required this.lastAt,
    required this.status,
    this.answer,
    this.error,
  });

  final int runId;
  final int idx;
  final List<int> segmentIds;
  final int lineCount;
  final DateTime firstAt;
  final DateTime lastAt;

  /// 'pending' | 'done' | 'failed'
  final String status;
  final String? answer;
  final String? error;

  bool get isPending => status == 'pending';
}

/// One finding ("- [12] "quote" — Strawman — why").
class ReviewItem {
  const ReviewItem({
    required this.chunkIdx,
    required this.category,
    required this.quote,
    required this.note,
    this.segmentId,
    this.speaker,
    this.saidAt,
  });

  final int chunkIdx;
  final int? segmentId;
  final String category;
  final String quote;
  final String note;
  final String? speaker;
  final DateTime? saidAt;
}

class ReviewRepository {
  ReviewRepository(this._db, {this.clock = systemClock});

  final AppDatabase _db;
  final Clock clock;

  /// Creates a run with its parts already planned.
  int create({
    required String title,
    required String prompt,
    required String format,
    required ReviewKind kind,
    required String periodLabel,
    required DateTime from,
    required DateTime to,
    required List<String> focus,
    required int chunkTokens,
    required int contextTokens,
    required List<List<SegmentView>> parts,
  }) {
    final now = clock().millisecondsSinceEpoch;
    return _db.transaction(() {
      _db.raw.execute(
        'INSERT INTO review_runs(title, prompt, format, kind, period_label, from_ms, to_ms, focus_json, chunk_tokens, '
        'context_tokens, status, total_chunks, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
          title,
          prompt,
          format,
          kind.name,
          periodLabel,
          from.millisecondsSinceEpoch,
          to.millisecondsSinceEpoch,
          jsonEncode(focus),
          chunkTokens,
          contextTokens,
          ReviewStatus.queued.name,
          parts.length,
          now,
          now,
        ],
      );
      final id = _db.raw.lastInsertRowId;
      for (var i = 0; i < parts.length; i++) {
        final part = parts[i];
        _db.raw.execute(
          'INSERT INTO review_chunks(run_id, idx, segment_ids, line_count, first_at, last_at, status) '
          "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
          [
            id,
            i,
            jsonEncode([for (final s in part) s.id]),
            part.length,
            part.first.startedAt.millisecondsSinceEpoch,
            part.last.startedAt.millisecondsSinceEpoch,
          ],
        );
      }
      return id;
    });
  }

  ReviewRun? run(int id) {
    final rows = _db.raw.select('SELECT * FROM review_runs WHERE id = ?', [id]);
    return rows.isEmpty ? null : _run(rows.first);
  }

  List<ReviewRun> runs({int limit = 100}) =>
      _db.raw.select('SELECT * FROM review_runs ORDER BY id DESC LIMIT ?', [limit]).map(_run).toList();

  bool get hasActive =>
      _db.raw.select("SELECT 1 FROM review_runs WHERE status IN ('queued', 'running') LIMIT 1").isNotEmpty;

  List<ReviewChunk> chunks(int runId) =>
      _db.raw.select('SELECT * FROM review_chunks WHERE run_id = ? ORDER BY idx', [runId]).map(_chunk).toList();

  ReviewChunk? nextPending(int runId) {
    final rows = _db.raw.select(
      "SELECT * FROM review_chunks WHERE run_id = ? AND status = 'pending' ORDER BY idx LIMIT 1",
      [runId],
    );
    return rows.isEmpty ? null : _chunk(rows.first);
  }

  List<ReviewItem> items(int runId) => _db.raw
      .select('SELECT * FROM review_items WHERE run_id = ? ORDER BY chunk_idx, id', [runId])
      .map((r) => ReviewItem(
            chunkIdx: r['chunk_idx']! as int,
            segmentId: r['segment_id'] as int?,
            category: r['category']! as String,
            quote: r['quote']! as String,
            note: r['note']! as String,
            speaker: r['speaker'] as String?,
            saidAt: r['said_at'] == null ? null : DateTime.fromMillisecondsSinceEpoch(r['said_at']! as int),
          ))
      .toList();

  /// Records a finished part and its findings.
  void completeChunk(int runId, int idx, String answer, List<ReviewItem> found) {
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      _db.raw
        ..execute('DELETE FROM review_items WHERE run_id = ? AND chunk_idx = ?', [runId, idx])
        ..execute(
          "UPDATE review_chunks SET status = 'done', answer = ?, error = NULL, finished_at = ? WHERE run_id = ? AND idx = ?",
          [answer, now, runId, idx],
        );
      for (final it in found) {
        _db.raw.execute(
          'INSERT INTO review_items(run_id, chunk_idx, segment_id, category, quote, note, speaker, said_at) '
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
          [runId, idx, it.segmentId, it.category, it.quote, it.note, it.speaker, it.saidAt?.millisecondsSinceEpoch],
        );
      }
      _bumpDone(runId, now);
    });
  }

  void failChunk(int runId, int idx, String error) {
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      _db.raw.execute(
        "UPDATE review_chunks SET status = 'failed', error = ?, finished_at = ? WHERE run_id = ? AND idx = ?",
        [error, now, runId, idx],
      );
      _bumpDone(runId, now);
    });
  }

  void _bumpDone(int runId, int now) => _db.raw.execute(
        "UPDATE review_runs SET done_chunks = (SELECT COUNT(*) FROM review_chunks WHERE run_id = ? AND status != 'pending'), "
        'updated_at = ? WHERE id = ?',
        [runId, now, runId],
      );

  void setMerge(int runId, String json) => _db.raw.execute(
        'UPDATE review_runs SET merge_json = ?, updated_at = ? WHERE id = ?',
        [json, clock().millisecondsSinceEpoch, runId],
      );

  void finish(int runId, String answer) {
    final now = clock().millisecondsSinceEpoch;
    _db.raw.execute(
      "UPDATE review_runs SET status = 'done', final_answer = ?, done_chunks = total_chunks + 1, lease_owner = NULL, "
      'lease_until = 0, updated_at = ?, finished_at = ? WHERE id = ?',
      [answer, now, now, runId],
    );
  }

  void setStatus(int runId, ReviewStatus status, {String? error}) {
    final now = clock().millisecondsSinceEpoch;
    final finished = status == ReviewStatus.done || status == ReviewStatus.failed || status == ReviewStatus.cancelled;
    _db.raw.execute(
      'UPDATE review_runs SET status = ?, error = ?, updated_at = ?, finished_at = CASE WHEN ? THEN ? ELSE finished_at END, '
      'lease_owner = CASE WHEN ? THEN lease_owner ELSE NULL END, lease_until = CASE WHEN ? THEN lease_until ELSE 0 END '
      'WHERE id = ?',
      [status.name, error, now, finished ? 1 : 0, now, status == ReviewStatus.running ? 1 : 0, status == ReviewStatus.running ? 1 : 0, runId],
    );
  }

  /// Retries the parts that failed.
  void retryFailed(int runId) {
    _db.transaction(() {
      _db.raw.execute("UPDATE review_chunks SET status = 'pending', error = NULL WHERE run_id = ? AND status = 'failed'", [runId]);
      _bumpDone(runId, clock().millisecondsSinceEpoch);
      _db.raw.execute('UPDATE review_runs SET merge_json = NULL, final_answer = NULL WHERE id = ?', [runId]);
    });
    setStatus(runId, ReviewStatus.queued);
  }

  void delete(int runId) => _db.raw.execute('DELETE FROM review_runs WHERE id = ?', [runId]);

  /// The next run [owner] may work on: queued or running, and not leased by
  /// another worker (leases expire, so a crashed worker's run is picked up).
  ReviewRun? nextRunnable(String owner) {
    final now = clock().millisecondsSinceEpoch;
    final rows = _db.raw.select(
      "SELECT * FROM review_runs WHERE status IN ('queued', 'running') "
      'AND (lease_owner IS NULL OR lease_owner = ? OR lease_until < ?) ORDER BY id LIMIT 1',
      [owner, now],
    );
    return rows.isEmpty ? null : _run(rows.first);
  }

  /// Takes (or renews) the run for [owner]. False if another worker holds it.
  bool claim(int runId, String owner, {Duration lease = const Duration(minutes: 3)}) {
    final now = clock().millisecondsSinceEpoch;
    _db.raw.execute(
      "UPDATE review_runs SET lease_owner = ?, lease_until = ?, status = 'running' "
      "WHERE id = ? AND status IN ('queued', 'running') AND (lease_owner IS NULL OR lease_owner = ? OR lease_until < ?)",
      [owner, now + lease.inMilliseconds, runId, owner, now],
    );
    return _db.raw.updatedRows > 0;
  }

  /// Lets another worker take the run straight away.
  void release(int runId, String owner) => _db.raw.execute(
        'UPDATE review_runs SET lease_owner = NULL, lease_until = 0 WHERE id = ? AND lease_owner = ?',
        [runId, owner],
      );

  void releaseAll(String owner) =>
      _db.raw.execute('UPDATE review_runs SET lease_owner = NULL, lease_until = 0 WHERE lease_owner = ?', [owner]);

  ReviewRun _run(Map<String, Object?> r) => ReviewRun(
        id: r['id']! as int,
        title: r['title']! as String,
        prompt: r['prompt']! as String,
        format: r['format']! as String,
        kind: ReviewKind.values.asNameMap()[r['kind']] ?? ReviewKind.list,
        periodLabel: r['period_label']! as String,
        from: DateTime.fromMillisecondsSinceEpoch(r['from_ms']! as int),
        to: DateTime.fromMillisecondsSinceEpoch(r['to_ms']! as int),
        focus: (jsonDecode(r['focus_json']! as String) as List<Object?>).whereType<String>().toList(),
        chunkTokens: r['chunk_tokens']! as int,
        contextTokens: r['context_tokens']! as int,
        status: ReviewStatus.values.asNameMap()[r['status']] ?? ReviewStatus.failed,
        totalChunks: r['total_chunks']! as int,
        doneChunks: r['done_chunks']! as int,
        createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
        updatedAt: DateTime.fromMillisecondsSinceEpoch(r['updated_at']! as int),
        mergeJson: r['merge_json'] as String?,
        finalAnswer: r['final_answer'] as String?,
        error: r['error'] as String?,
        finishedAt: r['finished_at'] == null ? null : DateTime.fromMillisecondsSinceEpoch(r['finished_at']! as int),
      );

  ReviewChunk _chunk(Map<String, Object?> r) => ReviewChunk(
        runId: r['run_id']! as int,
        idx: r['idx']! as int,
        segmentIds: (jsonDecode(r['segment_ids']! as String) as List<Object?>).whereType<int>().toList(),
        lineCount: r['line_count']! as int,
        firstAt: DateTime.fromMillisecondsSinceEpoch(r['first_at']! as int),
        lastAt: DateTime.fromMillisecondsSinceEpoch(r['last_at']! as int),
        status: r['status']! as String,
        answer: r['answer'] as String?,
        error: r['error'] as String?,
      );
}
