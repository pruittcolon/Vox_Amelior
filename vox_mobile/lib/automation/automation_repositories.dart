import 'dart:convert';

import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/core/log.dart';

class RuleRepository {
  RuleRepository(this._db);

  final AppDatabase _db;
  List<AutomationRule>? _cache;

  /// Rules are read on every utterance, so they are cached. Another isolate's
  /// edits become visible after [invalidate].
  List<AutomationRule> all() => _cache ??= _load();

  void invalidate() => _cache = null;

  List<AutomationRule> _load() {
    final rows = _db.raw.select('SELECT * FROM rules ORDER BY name COLLATE NOCASE');
    final rules = <AutomationRule>[];
    for (final r in rows) {
      try {
        rules.add(_fromRow(r));
      } on Object catch (e) {
        // A corrupt rule must not take the whole automation system down.
        Log.w('rules', 'skipping unreadable rule ${r['id']}', e);
      }
    }
    return rules;
  }

  void save(AutomationRule rule) {
    _cache = null;
    _db.raw.execute(
      'INSERT INTO rules(id, name, enabled, trigger_json, actions_json, cooldown_s, last_fired_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, '
      'enabled = excluded.enabled, trigger_json = excluded.trigger_json, '
      'actions_json = excluded.actions_json, cooldown_s = excluded.cooldown_s',
      [
        rule.id,
        rule.name.trim(),
        rule.enabled ? 1 : 0,
        rule.triggerJson(),
        rule.actionsJson(),
        rule.cooldownSeconds,
        rule.lastFiredAt?.millisecondsSinceEpoch,
      ],
    );
  }

  void setEnabled(String id, {required bool enabled}) {
    _cache = null;
    _db.raw.execute('UPDATE rules SET enabled = ? WHERE id = ?', [enabled ? 1 : 0, id]);
  }

  void markFired(String id, DateTime at) {
    _cache = null;
    _db.raw.execute('UPDATE rules SET last_fired_at = ? WHERE id = ?', [at.millisecondsSinceEpoch, id]);
  }

  void delete(String id) {
    _cache = null;
    _db.raw.execute('DELETE FROM rules WHERE id = ?', [id]);
  }

  AutomationRule _fromRow(Map<String, Object?> r) => AutomationRule(
        id: r['id']! as String,
        name: r['name']! as String,
        enabled: (r['enabled']! as int) == 1,
        trigger: RuleTrigger.fromJson((jsonDecode(r['trigger_json']! as String) as Map<String, Object?>)),
        actions: (jsonDecode(r['actions_json']! as String) as List<Object?>)
            .map((a) => RuleAction.fromJson(a! as Map<String, Object?>))
            .toList(),
        cooldownSeconds: r['cooldown_s']! as int,
        lastFiredAt: r['last_fired_at'] == null
            ? null
            : DateTime.fromMillisecondsSinceEpoch(r['last_fired_at']! as int),
      );
}

enum OutboxStatus { pending, delivered, failed }

class OutboxItem {
  const OutboxItem({
    required this.id,
    required this.ruleId,
    required this.url,
    required this.method,
    required this.headers,
    required this.body,
    required this.secret,
    required this.allowInsecure,
    required this.attempts,
    required this.nextAttemptAt,
    required this.status,
    required this.lastError,
    required this.createdAt,
  });

  final int id;
  final String? ruleId;
  final String url;
  final String method;
  final Map<String, String> headers;
  final String body;
  final String? secret;
  final bool allowInsecure;
  final int attempts;
  final DateTime nextAttemptAt;
  final OutboxStatus status;
  final String? lastError;
  final DateTime createdAt;
}

/// Durable queue of webhook deliveries, so nothing is lost if the phone is
/// offline or the receiving server is down.
class OutboxRepository {
  OutboxRepository(this._db, {this._clock = systemClock});

  final AppDatabase _db;
  final Clock _clock;

  int enqueue({
    String? ruleId,
    required String url,
    required String method,
    required Map<String, String> headers,
    required String body,
    String? secret,
    bool allowInsecure = false,
  }) {
    final now = _clock().millisecondsSinceEpoch;
    _db.raw.execute(
      'INSERT INTO outbox(rule_id, url, method, headers_json, body, secret, allow_insecure, '
      'attempts, next_attempt_at, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)',
      [ruleId, url, method, jsonEncode(headers), body, secret, allowInsecure ? 1 : 0, now, OutboxStatus.pending.name, now],
    );
    return _db.raw.lastInsertRowId;
  }

  List<OutboxItem> due({int limit = 20}) {
    final rows = _db.raw.select(
      'SELECT * FROM outbox WHERE status = ? AND next_attempt_at <= ? ORDER BY id LIMIT ?',
      [OutboxStatus.pending.name, _clock().millisecondsSinceEpoch, limit],
    );
    return rows.map(_fromRow).toList();
  }

  List<OutboxItem> recent({int limit = 50}) {
    final rows = _db.raw.select('SELECT * FROM outbox ORDER BY id DESC LIMIT ?', [limit]);
    return rows.map(_fromRow).toList();
  }

  void markDelivered(int id) {
    _db.raw.execute(
      'UPDATE outbox SET status = ?, delivered_at = ?, attempts = attempts + 1, last_error = NULL WHERE id = ?',
      [OutboxStatus.delivered.name, _clock().millisecondsSinceEpoch, id],
    );
  }

  /// Records a failed attempt; retries at [retryAt], or gives up if null.
  void markAttemptFailed(int id, String error, {DateTime? retryAt}) {
    _db.raw.execute(
      'UPDATE outbox SET attempts = attempts + 1, last_error = ?, status = ?, next_attempt_at = ? WHERE id = ?',
      [
        error,
        retryAt == null ? OutboxStatus.failed.name : OutboxStatus.pending.name,
        (retryAt ?? _clock()).millisecondsSinceEpoch,
        id,
      ],
    );
  }

  /// Puts a failed delivery back in the queue to try again now.
  void retryNow(int id) {
    _db.raw.execute(
      'UPDATE outbox SET status = ?, next_attempt_at = ?, attempts = 0 WHERE id = ?',
      [OutboxStatus.pending.name, _clock().millisecondsSinceEpoch, id],
    );
  }

  /// Housekeeping: drops finished deliveries older than [cutoff].
  int purgeFinishedBefore(DateTime cutoff) {
    _db.raw.execute(
      'DELETE FROM outbox WHERE status != ? AND created_at < ?',
      [OutboxStatus.pending.name, cutoff.millisecondsSinceEpoch],
    );
    return _db.raw.updatedRows;
  }

  OutboxItem _fromRow(Map<String, Object?> r) => OutboxItem(
        id: r['id']! as int,
        ruleId: r['rule_id'] as String?,
        url: r['url']! as String,
        method: r['method']! as String,
        headers: (jsonDecode(r['headers_json']! as String) as Map<String, Object?>).map((k, v) => MapEntry(k, '$v')),
        body: r['body']! as String,
        secret: r['secret'] as String?,
        allowInsecure: (r['allow_insecure']! as int) == 1,
        attempts: r['attempts']! as int,
        nextAttemptAt: DateTime.fromMillisecondsSinceEpoch(r['next_attempt_at']! as int),
        status: OutboxStatus.values.byName(r['status']! as String),
        lastError: r['last_error'] as String?,
        createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
      );
}

class Note {
  const Note(this.id, this.text, this.createdAt, this.source);
  final int id;
  final String text;
  final DateTime createdAt;
  final String source;
}

class NoteRepository {
  NoteRepository(this._db, {this._clock = systemClock});

  final AppDatabase _db;
  final Clock _clock;

  int add(String text, {String source = 'manual'}) {
    _db.raw.execute(
      'INSERT INTO notes(text, created_at, source) VALUES (?, ?, ?)',
      [text, _clock().millisecondsSinceEpoch, source],
    );
    return _db.raw.lastInsertRowId;
  }

  List<Note> all({int limit = 200}) {
    final rows = _db.raw.select('SELECT * FROM notes ORDER BY id DESC LIMIT ?', [limit]);
    return rows
        .map((r) => Note(
              r['id']! as int,
              r['text']! as String,
              DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
              r['source']! as String,
            ))
        .toList();
  }

  void delete(int id) => _db.raw.execute('DELETE FROM notes WHERE id = ?', [id]);
}
