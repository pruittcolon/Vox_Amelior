import 'dart:math';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';
import 'package:vox_amelior_mobile/speakers/voice_patterns.dart';

/// Persistence for enrolled speakers, their voice samples and the
/// automatically discovered "Guest N" clusters.
class SpeakerRepository {
  SpeakerRepository(this._db, {this.clock = systemClock});

  final AppDatabase _db;
  final Clock clock;

  /// Cap on stored samples per person. When exceeded, the oldest samples
  /// learned from corrections are dropped; enrollment samples are kept.
  static const int maxSamplesPerSpeaker = 1000;

  /// Cap on "not this person" examples per person (newest kept).
  static const int maxNegativesPerSpeaker = 100;

  /// A "not this person" example this close to a line the user later
  /// confirms as that person was probably them after all, so it is removed.
  static const double negativeConflict = 0.75;

  List<SpeakerProfile> profiles() {
    final rows = _db.raw.select('SELECT * FROM speakers ORDER BY name COLLATE NOCASE');
    final negatives = <String, List<Float32List>>{};
    for (final r in _db.raw.select('SELECT speaker_id, embedding FROM speaker_negatives ORDER BY id')) {
      negatives.putIfAbsent(r['speaker_id']! as String, () => []).add(blobToFloats(r['embedding']! as Uint8List));
    }
    return [for (final r in rows) _profileFromRow(r, negatives[r['id']] ?? const [])];
  }

  SpeakerProfile? profile(String id) {
    final rows = _db.raw.select('SELECT * FROM speakers WHERE id = ?', [id]);
    if (rows.isEmpty) return null;
    final negatives = [
      for (final r in _db.raw.select('SELECT embedding FROM speaker_negatives WHERE speaker_id = ? ORDER BY id', [id]))
        blobToFloats(r['embedding']! as Uint8List),
    ];
    return _profileFromRow(rows.first, negatives);
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

  SpeakerProfile _createInTransaction(
    String name,
    String embeddingModel,
    List<Float32List> samples,
    String source, {
    List<int?>? segmentIds,
  }) {
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
    for (var i = 0; i < samples.length; i++) {
      _insertSample(id, samples[i], source, now, segmentId: segmentIds?[i]);
    }
    _rebuild(id);
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
      _rebuild(speakerId);
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
        'SELECT id, embedding FROM segments WHERE cluster_id = ? AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?',
        [clusterId, maxSamplesPerSpeaker],
      );
      for (final r in rows) {
        _insertSample(speakerId, blobToFloats(r['embedding'] as Uint8List), 'label', now, segmentId: r['id'] as int);
      }
      _db.raw
        ..execute(
          'UPDATE voice_clips SET speaker_id = ? WHERE segment_id IN (SELECT id FROM segments WHERE cluster_id = ?)',
          [speakerId, clusterId],
        )
        ..execute(
          'UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE cluster_id = ?',
          [speakerId, clusterId],
        )
        ..execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
      _rebuild(speakerId);
    });
  }

  /// Corrects a single segment's speaker and learns from it.
  ///
  /// Fixing the same line again moves its sample to the new person instead
  /// of leaving a copy behind, so a wrong tap can always be undone.
  void assignSegmentToSpeaker(int segmentId, String speakerId) {
    final now = clock().millisecondsSinceEpoch;
    _db.transaction(() {
      final rows = _db.raw.select('SELECT embedding FROM segments WHERE id = ?', [segmentId]);
      if (rows.isEmpty) return;
      _forgetSegment(segmentId);
      _db.raw
        ..execute('UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE id = ?', [speakerId, segmentId])
        ..execute('UPDATE voice_clips SET speaker_id = ? WHERE segment_id = ?', [speakerId, segmentId]);
      final blob = rows.first['embedding'] as Uint8List?;
      if (blob != null) {
        final e = blobToFloats(blob);
        _insertSample(speakerId, e, 'label', now, segmentId: segmentId);
        _dropConflictingNegatives(speakerId, e);
      }
      _rebuild(speakerId);
    });
  }

  /// "Not [name]": the line was wrongly given to its current person.
  ///
  /// The person's voiceprint is not changed. The line's voice is kept as a
  /// "not this person" example, which only stops future voices that sound
  /// more like that example than like the person from getting their name.
  /// The line becomes a guest. Returns the guest label, or null if the line
  /// had no person.
  String? markNotSpeaker(int segmentId, {double clusterThreshold = 0.6, int maxClusterWeight = 50}) {
    String? label;
    final now = clock();
    _db.transaction(() {
      final rows = _db.raw.select('SELECT speaker_id, embedding FROM segments WHERE id = ?', [segmentId]);
      if (rows.isEmpty) return;
      final speakerId = rows.first['speaker_id'] as String?;
      final blob = rows.first['embedding'] as Uint8List?;
      if (speakerId == null) return;
      _forgetSegment(segmentId);
      String? clusterId;
      if (blob != null) {
        final e = blobToFloats(blob);
        _db.raw.execute(
          'INSERT INTO speaker_negatives(speaker_id, embedding, segment_id, created_at) VALUES (?, ?, ?, ?)',
          [speakerId, floatsToBlob(e), segmentId, now.millisecondsSinceEpoch],
        );
        _db.raw.execute(
          'DELETE FROM speaker_negatives WHERE speaker_id = ? AND id NOT IN '
          '(SELECT id FROM speaker_negatives WHERE speaker_id = ? ORDER BY id DESC LIMIT ?)',
          [speakerId, speakerId, maxNegativesPerSpeaker],
        );
        final cluster = _guestFor(e, clusterThreshold, maxClusterWeight, now);
        clusterId = cluster.id;
        label = cluster.label;
      }
      _db.raw
        ..execute('UPDATE segments SET speaker_id = NULL, cluster_id = ? WHERE id = ?', [clusterId, segmentId])
        ..execute('UPDATE voice_clips SET speaker_id = NULL WHERE segment_id = ?', [segmentId]);
      _rebuild(speakerId);
    });
    return label;
  }

  /// Marks or unmarks a guest voice as TV / background. Its lines, past and
  /// future (new lines that match this voice join it), are hidden when
  /// reading conversations back.
  void setClusterBackground(String clusterId, bool background) =>
      _db.raw.execute('UPDATE unknown_clusters SET background = ? WHERE id = ?', [background ? 1 : 0, clusterId]);

  /// "This is TV / background": the voice of line [segmentId] becomes a
  /// background voice. A line wrongly given to a person is taken off them
  /// first (as with [markNotSpeaker]). Returns the voice's label, or null
  /// when the line is too short to have a voice to remember.
  String? markBackground(int segmentId, {double clusterThreshold = 0.6, int maxClusterWeight = 50}) {
    final row = _db.raw.select('SELECT speaker_id FROM segments WHERE id = ?', [segmentId]);
    if (row.isEmpty) return null;
    if (row.first['speaker_id'] != null) {
      markNotSpeaker(segmentId, clusterThreshold: clusterThreshold, maxClusterWeight: maxClusterWeight);
    }
    String? label;
    _db.transaction(() {
      final r = _db.raw.select('SELECT cluster_id, embedding FROM segments WHERE id = ?', [segmentId]).first;
      var clusterId = r['cluster_id'] as String?;
      final blob = r['embedding'] as Uint8List?;
      if (clusterId == null) {
        if (blob == null) return;
        final cluster = _guestFor(blobToFloats(blob), clusterThreshold, maxClusterWeight, clock());
        clusterId = cluster.id;
        _db.raw.execute('UPDATE segments SET cluster_id = ? WHERE id = ?', [clusterId, segmentId]);
      }
      setClusterBackground(clusterId, true);
      label = _db.raw.select('SELECT label FROM unknown_clusters WHERE id = ?', [clusterId]).first['label'] as String?;
    });
    return label;
  }

  /// Names a new person from one line ("New person…").
  SpeakerProfile createFromSegment(int segmentId, String name, {required String embeddingModel}) {
    final rows = _db.raw.select('SELECT embedding FROM segments WHERE id = ?', [segmentId]);
    final blob = rows.isEmpty ? null : rows.first['embedding'] as Uint8List?;
    if (blob == null) throw StateError('This line is too short to learn a voice from');
    late SpeakerProfile created;
    _db.transaction(() {
      _forgetSegment(segmentId);
      created = _createInTransaction(name, embeddingModel, [blobToFloats(blob)], 'label', segmentIds: [segmentId]);
      _db.raw
        ..execute('UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE id = ?', [created.id, segmentId])
        ..execute('UPDATE voice_clips SET speaker_id = ? WHERE segment_id = ?', [created.id, segmentId]);
    });
    return profile(created.id)!;
  }

  /// Builds voice patterns for people enrolled before patterns existed.
  /// Returns how many people were updated.
  int ensurePatterns() {
    final ids = [
      for (final r in _db.raw.select('SELECT id FROM speakers WHERE patterns IS NULL AND sample_count >= 20'))
        r['id']! as String,
    ];
    for (final id in ids) {
      _db.transaction(() => _rebuild(id));
    }
    return ids.length;
  }

  int negativeCount(String speakerId) =>
      _db.raw.select('SELECT COUNT(*) AS c FROM speaker_negatives WHERE speaker_id = ?', [speakerId]).first['c']! as int;

  /// Removes every "not this person" example for [speakerId].
  void clearNegatives(String speakerId) =>
      _db.raw.execute('DELETE FROM speaker_negatives WHERE speaker_id = ?', [speakerId]);

  /// Number of voice patterns currently learned for [speakerId].
  int patternCount(String speakerId) => profile(speakerId)?.patterns.length ?? 0;

  /// Turns an unnamed voice into a new person, learning from everything
  /// they have already said.
  SpeakerProfile promoteCluster(String clusterId, String name, {required String embeddingModel}) {
    final rows = _db.raw.select(
      'SELECT id, embedding FROM segments WHERE cluster_id = ? AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?',
      [clusterId, maxSamplesPerSpeaker],
    );
    var samples = rows.map((r) => blobToFloats(r['embedding']! as Uint8List)).toList();
    List<int?> segmentIds = [for (final r in rows) r['id']! as int];
    if (samples.isEmpty) {
      final c = _db.raw.select('SELECT centroid FROM unknown_clusters WHERE id = ?', [clusterId]);
      if (c.isEmpty) throw StateError('That voice no longer exists');
      samples = [blobToFloats(c.first['centroid']! as Uint8List)];
      segmentIds = [null];
    }
    late SpeakerProfile created;
    _db.transaction(() {
      created = _createInTransaction(name, embeddingModel, samples, 'label', segmentIds: segmentIds);
      _db.raw
        ..execute(
          'UPDATE voice_clips SET speaker_id = ? WHERE segment_id IN (SELECT id FROM segments WHERE cluster_id = ?)',
          [created.id, clusterId],
        )
        ..execute('UPDATE segments SET speaker_id = ?, cluster_id = NULL WHERE cluster_id = ?', [created.id, clusterId])
        ..execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
    });
    return profile(created.id)!;
  }

  void deleteCluster(String clusterId) {
    _db.raw.execute('DELETE FROM unknown_clusters WHERE id = ?', [clusterId]);
  }

  // ---- internals ---------------------------------------------------------

  void _insertSample(String speakerId, Float32List e, String source, int now, {int? segmentId}) {
    _db.raw.execute(
      'INSERT INTO speaker_samples(speaker_id, embedding, source, created_at, segment_id) VALUES (?, ?, ?, ?, ?)',
      [speakerId, floatsToBlob(e), source, now, segmentId],
    );
  }

  /// Removes what earlier corrections of this line taught (its sample or
  /// its "not this person" example), rebuilding the affected people.
  void _forgetSegment(int segmentId) {
    final affected = {
      for (final r in _db.raw.select('SELECT DISTINCT speaker_id FROM speaker_samples WHERE segment_id = ?', [segmentId]))
        r['speaker_id']! as String,
    };
    _db.raw
      ..execute('DELETE FROM speaker_samples WHERE segment_id = ?', [segmentId])
      ..execute('DELETE FROM speaker_negatives WHERE segment_id = ?', [segmentId]);
    for (final id in affected) {
      _rebuild(id);
    }
  }

  void _dropConflictingNegatives(String speakerId, Float32List confirmed) {
    final rows = _db.raw.select('SELECT id, embedding FROM speaker_negatives WHERE speaker_id = ?', [speakerId]);
    for (final r in rows) {
      if (cosine(confirmed, blobToFloats(r['embedding']! as Uint8List)) >= negativeConflict) {
        _db.raw.execute('DELETE FROM speaker_negatives WHERE id = ?', [r['id']]);
      }
    }
  }

  /// Attaches a voice to the closest guest, or starts a new guest.
  UnknownCluster _guestFor(Float32List e, double threshold, int maxWeight, DateTime now) {
    UnknownCluster? best;
    var bestScore = -1.0;
    for (final c in clusters()) {
      final s = cosine(e, c.centroid);
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best != null && bestScore >= threshold) {
      final w = best.count.clamp(1, maxWeight);
      final n = l2Normalize(e);
      final merged = Float32List(best.centroid.length);
      for (var i = 0; i < merged.length; i++) {
        merged[i] = best.centroid[i] * w + n[i];
      }
      best
        ..centroid = l2Normalize(merged)
        ..count = best.count + 1
        ..updatedAt = now;
      saveCluster(best);
      return best;
    }
    final fresh = UnknownCluster(id: newId(), label: nextGuestLabel(), centroid: l2Normalize(e), count: 1, updatedAt: now);
    saveCluster(fresh);
    return fresh;
  }

  /// Recomputes a person's average and voice patterns from their samples,
  /// first trimming the oldest correction samples above the cap.
  void _rebuild(String speakerId) {
    final total = _db.raw.select('SELECT COUNT(*) AS c FROM speaker_samples WHERE speaker_id = ?', [speakerId]).first['c']! as int;
    if (total > maxSamplesPerSpeaker) {
      _db.raw.execute(
        "DELETE FROM speaker_samples WHERE id IN (SELECT id FROM speaker_samples WHERE speaker_id = ? AND source != 'enroll' "
        'ORDER BY id LIMIT ?)',
        [speakerId, total - maxSamplesPerSpeaker],
      );
    }
    final rows = _db.raw.select('SELECT embedding FROM speaker_samples WHERE speaker_id = ? ORDER BY id', [speakerId]);
    if (rows.isEmpty) return;
    final vectors = rows.map((r) => blobToFloats(r['embedding'] as Uint8List)).toList();
    _db.raw.execute(
      'UPDATE speakers SET centroid = ?, sample_count = ?, patterns = ? WHERE id = ?',
      [floatsToBlob(meanEmbedding(vectors)), vectors.length, patternsToBlob(voicePatterns(vectors)), speakerId],
    );
  }

  SpeakerProfile _profileFromRow(Map<String, Object?> r, List<Float32List> negatives) {
    final centroid = blobToFloats(r['centroid']! as Uint8List);
    return SpeakerProfile(
      id: r['id']! as String,
      name: r['name']! as String,
      embeddingModel: r['embedding_model']! as String,
      centroid: centroid,
      sampleCount: r['sample_count']! as int,
      createdAt: DateTime.fromMillisecondsSinceEpoch(r['created_at']! as int),
      patterns: patternsFromBlob(r['patterns'] as Uint8List?, centroid.length),
      negatives: negatives,
    );
  }

  UnknownCluster _clusterFromRow(Map<String, Object?> r) => UnknownCluster(
        id: r['id']! as String,
        label: r['label']! as String,
        centroid: blobToFloats(r['centroid']! as Uint8List),
        count: r['count']! as int,
        updatedAt: DateTime.fromMillisecondsSinceEpoch(r['updated_at']! as int),
        background: (r['background'] as int? ?? 0) != 0,
      );

  static final Random _rng = Random.secure();

  /// Random 128-bit hex id.
  static String newId() => List.generate(16, (_) => _rng.nextInt(256).toRadixString(16).padLeft(2, '0')).join();
}
