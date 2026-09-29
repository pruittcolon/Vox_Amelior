import 'dart:math';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

/// Persistence for enrolled speakers, their voice samples and the
/// automatically discovered "Guest N" clusters.
class SpeakerRepository {
  SpeakerRepository(this._db, {this.clock = systemClock});

  final AppDatabase _db;
  final Clock clock;

  /// Cap on stored samples per person; the newest are kept when exceeded.
  static const int maxSamplesPerSpeaker = 200;

  List<SpeakerProfile> profiles() {
    final rows = _db.raw.select('SELECT * FROM speakers ORDER BY name COLLATE NOCASE');
    return rows.map(_profileFromRow).toList();
  }

  SpeakerProfile? profile(String id) {
    final rows = _db.raw.select('SELECT * FROM speakers WHERE id = ?', [id]);
    return rows.isEmpty ? null : _profileFromRow(rows.first);
  }

  /// Creates a person from voice [samples] (embeddings).
  ///
  /// Throws [ArgumentError] for an empty name/sample list and
  /// [StateError] if the name is already used.
  SpeakerProfile create({
    required String name,
    required String embeddingModel,
    required List<Float32List> samples,
    String source = 'enroll',
  }) {
    late SpeakerProfile created;
    _db.transaction(() => created = _createInTransaction(name, embeddingModel, samples, source));
    return created;
  }

  SpeakerProfile _createInTransaction(String name, String embeddingModel, List<Float32List> samples, String source) {
    final cleaned = name.trim();
    if (cleaned.isEmpty) throw ArgumentError('Name is required');
    if (samples.isEmpty) throw ArgumentError('At least one sample is required');
    final exists = _db.raw.select('SELECT 1 FROM speakers WHERE name = ? COLLATE NOCASE', [cleaned]);
    if (exists.isNotEmpty) throw StateError('A person named "$cleaned" already exists');

    final id = newId();
    final now = clock().millisecondsSinceEpoch;
    _db.raw.execute(
      'INSERT INTO speakers(id, name, created_at, embedding_model, centroid, sample_count) '
      'VALUES (?, ?, ?, ?, ?, ?)',
      [id, cleaned, now, embeddingModel, floatsToBlob(meanEmbedding(samples)), samples.length],
    );
    for (final s in samples) {
      _insertSample(id, s, source, now);
    }
    return profile(id)!;
  }

  /// Adds more voice [samples] to an existing person and refreshes their profile.
  void addSamples(String speakerId, List<Float32List> samples, {String source = 'enroll'}) {
    if (samples.isEmpty) return;
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      for (final s in samples) {
        _insertSample(speakerId, s, source, now);
      }
      _rebuildCentroid(speakerId);
    });
  }

  void rename(String speakerId, String newName) {
    final cleaned = newName.trim();
    if (cleaned.isEmpty) throw ArgumentError('Name is required');
    final clash = _db.raw.select(
      'SELECT 1 FROM speakers WHERE name = ? COLLATE NOCASE AND id != ?',
      [cleaned, speakerId],
    );
    if (clash.isNotEmpty) throw StateError('A person named "$cleaned" already exists');
    _db.raw.execute('UPDATE speakers SET name = ? WHERE id = ?', [cleaned, speakerId]);
  }

  /// Removes a person and their samples. Their past segments stay in the
  /// archive but become unattributed.
  void delete(String speakerId) {
    _db.raw.execute('DELETE FROM speakers WHERE id = ?', [speakerId]);
  }

  int sampleCount(String speakerId) {
    final r = _db.raw.select(
      'SELECT COUNT(*) AS c FROM speaker_samples WHERE speaker_id = ?',
      [speakerId],
    );
    return r.first['c'] as int;
  }

  // ---- unknown clusters -------------------------------------------------

  List<UnknownCluster> clusters() {
    final rows = _db.raw.select('SELECT * FROM unknown_clusters ORDER BY updated_at DESC');
    return rows.map(_clusterFromRow).toList();
  }

  /// Label for the next newly discovered voice: "Guest 1", "Guest 2", ...
  String nextGuestLabel() {
    final rows = _db.raw.select(
      "SELECT label FROM unknown_clusters WHERE label LIKE 'Guest %'",
    );
    var highest = 0;
    for (final r in rows) {
      final n = int.tryParse((r['label'] as String).substring(6));
      if (n != null && n > highest) highest = n;
    }
    return 'Guest ${highest + 1}';
  }

  void saveCluster(UnknownCluster c) {
    _db.raw.execute(
      'INSERT INTO unknown_clusters(id, label, centroid, count, updated_at) VALUES (?, ?, ?, ?, ?) '
      'ON CONFLICT(id) DO UPDATE SET centroid = excluded.centroid, '
      'count = excluded.count, updated_at = excluded.updated_at',
      [c.id, c.label, floatsToBlob(c.centroid), c.count, c.updatedAt.millisecondsSinceEpoch],
    );
  }

  /// Names an unknown voice: its segments move to [speakerId], and their
  /// embeddings become training samples for that person.
  void assignClusterToSpeaker(String clusterId, String speakerId) {
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      final rows = _db.raw.select(
        'SELECT embedding FROM segments WHERE cluster_id = ? AND embedding IS NOT NULL',
        [clusterId],
      );
      for (final r in rows) {
        _insertSample(speakerId, blobToFloats(r['embedding'] as Uint8List), 'label', now);
      }
      _db.raw
        ..execute(
          'UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE cluster_id = ?',
          [speakerId, clusterId],
        )
        ..execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
      _rebuildCentroid(speakerId);
    });
  }

  /// Corrects a single segment's speaker and learns from it.
  void assignSegmentToSpeaker(int segmentId, String speakerId) {
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      final rows = _db.raw.select('SELECT embedding FROM segments WHERE id = ?', [segmentId]);
      if (rows.isEmpty) return;
      _db.raw.execute(
        'UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE id = ?',
        [speakerId, segmentId],
      );
      final blob = rows.first['embedding'] as Uint8List?;
      if (blob != null) {
        _insertSample(speakerId, blobToFloats(blob), 'label', now);
        _rebuildCentroid(speakerId);
      }
    });
  }

  /// Turns an unnamed voice into a new person, learning from everything
  /// they have already said.
  SpeakerProfile promoteCluster(String clusterId, String name, {required String embeddingModel}) {
    final rows = _db.raw.select(
      'SELECT embedding FROM segments WHERE cluster_id = ? AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?',
      [clusterId, maxSamplesPerSpeaker],
    );
    var samples = rows.map((r) => blobToFloats(r['embedding']! as Uint8List)).toList();
    if (samples.isEmpty) {
      final c = _db.raw.select('SELECT centroid FROM unknown_clusters WHERE id = ?', [clusterId]);
      if (c.isEmpty) throw StateError('That voice no longer exists');
      samples = [blobToFloats(c.first['centroid']! as Uint8List)];
    }
    late SpeakerProfile created;
    _db.transaction(() {
      created = _createInTransaction(name, embeddingModel, samples, 'label');
      _db.raw
        ..execute('UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE cluster_id = ?', [created.id, clusterId])
        ..execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
    });
    return profile(created.id)!;
  }

  void deleteCluster(String clusterId) {
    _db.raw.execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
  }

  // ---- internals ---------------------------------------------------------

  void _insertSample(String speakerId, Float32List e, String source, int now) {
    _db.raw.execute(
      'INSERT INTO speaker_samples(speaker_id, embedding, source, created_at) VALUES (?, ?, ?, ?)',
      [speakerId, floatsToBlob(e), source, now],
    );
  }

  void _rebuildCentroid(String speakerId) {
    _db.raw.execute(
      'DELETE FROM speaker_samples WHERE speaker_id = ? AND id NOT IN '
      '(SELECT id FROM speaker_samples WHERE speaker_id = ? ORDER BY id DESC LIMIT ?)',
      [speakerId, speakerId, maxSamplesPerSpeaker],
    );
    final rows = _db.raw.select(
      'SELECT embedding FROM speaker_samples WHERE speaker_id = ?',
      [speakerId],
    );
    if (rows.isEmpty) return;
    final vectors = rows.map((r) => blobToFloats(r['embedding'] as Uint8List)).toList();
    _db.raw.execute(
      'UPDATE speakers SET centroid = ?, sample_count = ? WHERE id = ?',
      [floatsToBlob(meanEmbedding(vectors)), vectors.length, speakerId],
    );
  }

  SpeakerProfile _profileFromRow(Map<String, Object?> r) => SpeakerProfile(
        id: r['id']! as String,
        name: r['name']! as String,
        embeddingModel: r['embedding_model']! as String,
        centroid: blobToFloats(r['centroid']! as Uint8List),
        sampleCount: r['sample_count']! as int,
        createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
      );

  UnknownCluster _clusterFromRow(Map<String, Object?> r) => UnknownCluster(
        id: r['id']! as String,
        label: r['label']! as String,
        centroid: blobToFloats(r['centroid']! as Uint8List),
        count: r['count']! as int,
        updatedAt: DateTime.fromMillisecondsSinceEpoch(r['updated_at']! as int),
      );

  static final Random _rng = Random.secure();

  /// Random 128-bit hex id.
  static String newId() => List.generate(16, (_) => _rng.nextInt(256).toRadixString(16).padLeft(2, '0')).join();
}
