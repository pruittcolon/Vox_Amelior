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

  static const int schemaVersion = 6;

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
    if (raw.userVersion >= schemaVersion) return;
    transaction(() {
      // Re-read under the write lock: the app and the listening service can
      // open the database at the same moment, and only one may migrate.
      final current = raw.userVersion;
      if (current >= schemaVersion) return;
      if (current < 1) _v1();
      if (current < 2) _v2();
      if (current < 3) _v3();
      if (current < 4) _v4();
      if (current < 5) _v5();
      if (current < 6) _v6();
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

  /// Assistant request origin and sources; reminders set by the assistant.
  void _v2() {
    raw
      ..execute("ALTER TABLE assistant_requests ADD COLUMN source TEXT NOT NULL DEFAULT 'voice'")
      ..execute('ALTER TABLE assistant_requests ADD COLUMN sources_json TEXT')
      ..execute('''
CREATE TABLE reminders (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  due_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  fired_at INTEGER
)''')
      ..execute('CREATE INDEX reminders_due ON reminders(fired_at, due_at)');
  }

  /// Voice patterns and "not this person" examples, saved voice clips and
  /// long-running reviews.
  void _v3() {
    raw
      ..execute('ALTER TABLE speakers ADD COLUMN patterns BLOB')
      ..execute('ALTER TABLE speaker_samples ADD COLUMN segment_id INTEGER')
      ..execute('CREATE INDEX speaker_samples_segment ON speaker_samples(segment_id)')
      ..execute('''
CREATE TABLE speaker_negatives (
  id INTEGER PRIMARY KEY,
  speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  segment_id INTEGER,
  created_at INTEGER NOT NULL
)''')
      ..execute('CREATE INDEX speaker_negatives_speaker ON speaker_negatives(speaker_id)')
      ..execute('''
CREATE TABLE voice_clips (
  id INTEGER PRIMARY KEY,
  segment_id INTEGER UNIQUE,
  speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
  text TEXT NOT NULL,
  path TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  started_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL
)''')
      ..execute('CREATE INDEX voice_clips_speaker ON voice_clips(speaker_id)')
      ..execute('''
CREATE TABLE review_runs (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  prompt TEXT NOT NULL,
  format TEXT NOT NULL,
  kind TEXT NOT NULL,
  period_label TEXT NOT NULL,
  from_ms INTEGER NOT NULL,
  to_ms INTEGER NOT NULL,
  focus_json TEXT NOT NULL,
  chunk_tokens INTEGER NOT NULL,
  context_tokens INTEGER NOT NULL,
  status TEXT NOT NULL,
  total_chunks INTEGER NOT NULL,
  done_chunks INTEGER NOT NULL DEFAULT 0,
  merge_json TEXT,
  final_answer TEXT,
  error TEXT,
  lease_owner TEXT,
  lease_until INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  finished_at INTEGER
)''')
      ..execute('''
CREATE TABLE review_chunks (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES review_runs(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  segment_ids TEXT NOT NULL,
  line_count INTEGER NOT NULL,
  first_at INTEGER NOT NULL,
  last_at INTEGER NOT NULL,
  status TEXT NOT NULL,
  answer TEXT,
  error TEXT,
  finished_at INTEGER
)''')
      ..execute('CREATE UNIQUE INDEX review_chunks_run ON review_chunks(run_id, idx)')
      ..execute('''
CREATE TABLE review_items (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES review_runs(id) ON DELETE CASCADE,
  chunk_idx INTEGER NOT NULL,
  segment_id INTEGER,
  category TEXT NOT NULL,
  quote TEXT NOT NULL,
  note TEXT NOT NULL,
  speaker TEXT,
  said_at INTEGER
)''')
      ..execute('CREATE INDEX review_items_run ON review_items(run_id, chunk_idx)');
  }

  /// Lines where two people talked at the same time.
  void _v4() {
    raw.execute('ALTER TABLE segments ADD COLUMN overlap INTEGER NOT NULL DEFAULT 0');
  }

  /// Voices marked as TV or background (hidden when reading back), and the
  /// speaker name of lines imported from a text export.
  void _v5() {
    raw
      ..execute('ALTER TABLE unknown_clusters ADD COLUMN background INTEGER NOT NULL DEFAULT 0')
      ..execute('ALTER TABLE segments ADD COLUMN speaker_label TEXT');
  }

  /// Tone of voice per line (happy, sad, angry, ...) and sounds heard in it
  /// (laughter, music, ...), when the tone model is installed.
  void _v6() {
    raw
      ..execute('ALTER TABLE segments ADD COLUMN emotion TEXT')
      ..execute('ALTER TABLE segments ADD COLUMN sound TEXT');
  }
}
