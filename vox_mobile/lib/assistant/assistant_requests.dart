import 'dart:convert';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';

enum RequestStatus { pending, answered, failed }

/// Where a question came from.
enum RequestSource { voice, app }

class AssistantRequest {
  const AssistantRequest({
    required this.id,
    required this.text,
    required this.createdAt,
    required this.status,
    required this.source,
    this.answer,
    this.sourceSegmentIds = const [],
    this.hintIds = const [],
  });

  final int id;
  final String text;
  final DateTime createdAt;
  final RequestStatus status;
  final RequestSource source;
  final String? answer;

  /// Transcript segments the answer was based on.
  final List<int> sourceSegmentIds;

  /// Lines the app found by meaning for this question (EmbeddingGemma), for
  /// the service to read too; it has no embedder of its own.
  final List<int> hintIds;
}

/// Questions for the assistant, spoken ("Hey Vox, ...") or typed. The
/// always-on service answers them in order; the app shows them.
class AssistantRequestRepository {
  AssistantRequestRepository(this._db, {this._clock = systemClock});

  final AppDatabase _db;
  final Clock _clock;

  int add(String text, {RequestSource source = RequestSource.voice, List<int> hints = const []}) {
    _db.raw.execute(
      'INSERT INTO assistant_requests(text, created_at, status, source, hint_ids) VALUES (?, ?, ?, ?, ?)',
      [text, _clock().millisecondsSinceEpoch, RequestStatus.pending.name, source.name, hints.isEmpty ? null : jsonEncode(hints)],
    );
    return _db.raw.lastInsertRowId;
  }

  AssistantRequest? get(int id) {
    final rows = _db.raw.select('SELECT * FROM assistant_requests WHERE id = ?', [id]);
    return rows.isEmpty ? null : _fromRow(rows.first);
  }

  /// Unanswered requests, oldest first. Requests older than [maxAge] are
  /// considered stale (nobody is waiting any more) and are marked failed.
  List<AssistantRequest> takePending({Duration maxAge = const Duration(minutes: 30)}) {
    final cutoff = _clock().subtract(maxAge).millisecondsSinceEpoch;
    _db.raw.execute(
      'UPDATE assistant_requests SET status = ?, answer = ? WHERE status = ? AND created_at < ?',
      [RequestStatus.failed.name, 'Expired before it could be answered', RequestStatus.pending.name, cutoff],
    );
    final rows = _db.raw.select(
      'SELECT * FROM assistant_requests WHERE status = ? ORDER BY id',
      [RequestStatus.pending.name],
    );
    return rows.map(_fromRow).toList();
  }

  void answer(int id, String answer, {List<int> sources = const []}) {
    _db.raw.execute(
      'UPDATE assistant_requests SET status = ?, answer = ?, answered_at = ?, sources_json = ? WHERE id = ?',
      [RequestStatus.answered.name, answer, _clock().millisecondsSinceEpoch, jsonEncode(sources), id],
    );
  }

  void fail(int id, String reason) {
    _db.raw.execute(
      'UPDATE assistant_requests SET status = ?, answer = ? WHERE id = ?',
      [RequestStatus.failed.name, reason, id],
    );
  }

  List<AssistantRequest> recent({int limit = 30, RequestSource? source}) {
    final rows = source == null
        ? _db.raw.select('SELECT * FROM assistant_requests ORDER BY id DESC LIMIT ?', [limit])
        : _db.raw.select('SELECT * FROM assistant_requests WHERE source = ? ORDER BY id DESC LIMIT ?', [source.name, limit]);
    return rows.map(_fromRow).toList();
  }

  AssistantRequest _fromRow(Map<String, Object?> r) {
    final src = r['sources_json'] as String?;
    final hints = r['hint_ids'] as String?;
    return AssistantRequest(
      id: r['id']! as int,
      text: r['text']! as String,
      createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
      status: RequestStatus.values.byName(r['status']! as String),
      source: RequestSource.values.asNameMap()[r['source']] ?? RequestSource.voice,
      answer: r['answer'] as String?,
      sourceSegmentIds: src == null ? const [] : (jsonDecode(src) as List<Object?>).whereType<int>().toList(),
      hintIds: hints == null ? const [] : (jsonDecode(hints) as List<Object?>).whereType<int>().toList(),
    );
  }
}

class Reminder {
  const Reminder(this.id, this.text, this.dueAt, {this.firedAt});
  final int id;
  final String text;
  final DateTime dueAt;
  final DateTime? firedAt;
}

/// Reminders the assistant sets ("remind me in 20 minutes to...").
class ReminderRepository {
  ReminderRepository(this._db, {this._clock = systemClock});

  final AppDatabase _db;
  final Clock _clock;

  int add(String text, DateTime dueAt) {
    _db.raw.execute(
      'INSERT INTO reminders(text, due_at, created_at) VALUES (?, ?, ?)',
      [text, dueAt.millisecondsSinceEpoch, _clock().millisecondsSinceEpoch],
    );
    return _db.raw.lastInsertRowId;
  }

  /// Reminders that are due and not yet shown; marks them as shown.
  List<Reminder> takeDue() {
    final now = _clock().millisecondsSinceEpoch;
    late List<Reminder> due;
    _db.transaction(() {
      final rows = _db.raw.select(
        'SELECT * FROM reminders WHERE fired_at IS NULL AND due_at <= ? ORDER BY due_at',
        [now],
      );
      due = rows.map(_fromRow).toList();
      _db.raw.execute('UPDATE reminders SET fired_at = ? WHERE fired_at IS NULL AND due_at <= ?', [now, now]);
    });
    return due;
  }

  List<Reminder> upcoming({int limit = 50}) {
    final rows = _db.raw.select('SELECT * FROM reminders WHERE fired_at IS NULL ORDER BY due_at LIMIT ?', [limit]);
    return rows.map(_fromRow).toList();
  }

  void delete(int id) => _db.raw.execute('DELETE FROM reminders WHERE id = ?', [id]);

  Reminder _fromRow(Map<String, Object?> r) => Reminder(
        r['id']! as int,
        r['text']! as String,
        DateTime.fromMillisecondsSinceEpoch(r['due_at']! as int),
        firedAt: r['fired_at'] == null ? null : DateTime.fromMillisecondsSinceEpoch(r['fired_at']! as int),
      );
}
