import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';

enum RequestStatus { pending, answered, failed }

class AssistantRequest {
  const AssistantRequest({
    required this.id,
    required this.text,
    required this.createdAt,
    required this.status,
    this.answer,
  });

  final int id;
  final String text;
  final DateTime createdAt;
  final RequestStatus status;
  final String? answer;
}

/// Questions spoken after the wake phrase. The always-on listener queues
/// them here; the app (which owns the language model) answers them.
class AssistantRequestRepository {
  AssistantRequestRepository(this._db, {this._clock = systemClock});

  final AppDatabase _db;
  final Clock _clock;

  int add(String text) {
    _db.raw.execute(
      'INSERT INTO assistant_requests(text, created_at, status) VALUES (?, ?, ?)',
      [text, _clock().millisecondsSinceEpoch, RequestStatus.pending.name],
    );
    return _db.raw.lastInsertRowId;
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

  void answer(int id, String answer) {
    _db.raw.execute(
      'UPDATE assistant_requests SET status = ?, answer = ?, answered_at = ? WHERE id = ?',
      [RequestStatus.answered.name, answer, _clock().millisecondsSinceEpoch, id],
    );
  }

  void fail(int id, String reason) {
    _db.raw.execute(
      'UPDATE assistant_requests SET status = ?, answer = ? WHERE id = ?',
      [RequestStatus.failed.name, reason, id],
    );
  }

  List<AssistantRequest> recent({int limit = 30}) {
    final rows = _db.raw.select('SELECT * FROM assistant_requests ORDER BY id DESC LIMIT ?', [limit]);
    return rows.map(_fromRow).toList();
  }

  AssistantRequest _fromRow(Map<String, Object?> r) => AssistantRequest(
        id: r['id']! as int,
        text: r['text']! as String,
        createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
        status: RequestStatus.values.byName(r['status']! as String),
        answer: r['answer'] as String?,
      );
}
