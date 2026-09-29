import 'dart:io';

import 'package:sqlite3/sqlite3.dart';

/// Thin owner of the SQLite connection and schema migrations.
///
/// Two isolates open the same file (the always-on listener writes, the UI
/// reads and edits), so file databases use WAL and a busy timeout.
class AppDatabase {
  AppDatabase._(this.raw, this.path);

  final Database raw;

  /// File location, or ':memory:' for test databases.
  final String path;

  static const int schemaVersion = 1;

  /// Opens (creating and migrating if needed) the database at [path].
  static AppDatabase open(String path) {
    Directory(File(path).parent.path).createSync(recursive: true);
    final db = sqlite3.open(path);
    db
      ..execute('PRAGMA journal_mode = WAL')
      ..execute('PRAGMA synchronous = NORMAL')
      ..execute('PRAGMA busy_timeout = 8000')
      ..execute('PRAGMA foreign_keys = ON');
    final app = AppDatabase._(db, path).._migrate();
    return app;
  }

  /// In-memory database for tests.
  static AppDatabase inMemory() {
    final db = sqlite3.openInMemory()..execute('PRAGMA foreign_keys = ON');
    return AppDatabase._(db, ':memory:').._migrate();
  }

  void close() => raw.close();

  /// Runs [body] in a transaction; rolls back if it throws.
  T transaction<T>(T Function() body) {
    raw.execute('BEGIN IMMEDIATE');
    try {
      final result = body();
      raw.execute('COMMIT');
      return result;
    } catch (_) {
      raw.execute('ROLLBACK');
      rethrow;
    }
  }

  void _migrate() {
    final current = raw.userVersion;
    if (current >= schemaVersion) return;
    transaction(() {
      if (current < 1) _v1();
      raw.userVersion = schemaVersion;
    });
  }

  void _v1() {
    raw.execute('''
CREATE TABLE speakers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  embedding_model TEXT NOT NULL,
  centroid BLOB NOT NULL,
  sample_count INTEGER NOT NULL DEFAULT 0
)''');
    raw.execute('CREATE UNIQUE INDEX speakers_name ON speakers(name COLLATE NOCASE)');
    raw.execute('''
CREATE TABLE speaker_samples (
  id INTEGER PRIMARY KEY,
  speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  source TEXT NOT NULL,
  created_at INTEGER NOT NULL
)''');
    raw.execute('CREATE INDEX speaker_samples_speaker ON speaker_samples(speaker_id)');
    raw.execute('''
CREATE TABLE unknown_clusters (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  centroid BLOB NOT NULL,
  count INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
)''');
    raw.execute('''
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY,
  started_at INTEGER NOT NULL,
  ended_at INTEGER NOT NULL
)''');
    raw.execute('''
CREATE TABLE segments (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  started_at INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  text TEXT NOT NULL,
  speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
  cluster_id TEXT REFERENCES unknown_clusters(id) ON DELETE SET NULL,
  score REAL,
  embedding BLOB
)''');
    raw.execute('CREATE INDEX segments_time ON segments(started_at)');
    raw.execute('CREATE INDEX segments_conversation ON segments(conversation_id)');
    raw.execute('CREATE INDEX segments_speaker ON segments(speaker_id)');
    raw.execute('CREATE INDEX segments_cluster ON segments(cluster_id)');
    raw.execute('''
CREATE VIRTUAL TABLE segments_fts USING fts5(
  text, content='segments', content_rowid='id', tokenize='porter unicode61'
)''');
    raw.execute('''
CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
  INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END''');
    raw.execute('''
CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
  INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END''');
    raw.execute('''
CREATE TRIGGER segments_au AFTER UPDATE OF text ON segments BEGIN
  INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END''');
    raw.execute('''
CREATE TABLE rules (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  trigger_json TEXT NOT NULL,
  actions_json TEXT NOT NULL,
  cooldown_s INTEGER NOT NULL DEFAULT 30,
  last_fired_at INTEGER
)''');
    raw.execute('''
CREATE TABLE outbox (
  id INTEGER PRIMARY KEY,
  rule_id TEXT,
  url TEXT NOT NULL,
  method TEXT NOT NULL,
  headers_json TEXT NOT NULL,
  body TEXT NOT NULL,
  secret TEXT,
  allow_insecure INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at INTEGER NOT NULL,
  status TEXT NOT NULL,
  last_error TEXT,
  created_at INTEGER NOT NULL,
  delivered_at INTEGER
)''');
    raw.execute('CREATE INDEX outbox_due ON outbox(status, next_attempt_at)');
    raw.execute('''
CREATE TABLE assistant_requests (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  status TEXT NOT NULL,
  answer TEXT,
  answered_at INTEGER
)''');
    raw.execute('''
CREATE TABLE notes (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  source TEXT NOT NULL
)''');
  }
}
