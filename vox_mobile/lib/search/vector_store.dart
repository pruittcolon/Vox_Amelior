import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';

/// A line waiting for its vector, with the text to embed.
class PendingLine {
  const PendingLine(this.id, this.text);

  final int id;

  /// What is embedded, or null when the line is too short to mean anything
  /// (it is then marked done without a vector).
  final String? text;
}

/// Every stored vector of one model, packed for fast scoring.
class VectorIndex {
  const VectorIndex(this.ids, this.data, this.scales);

  static final VectorIndex empty = VectorIndex(Int32List(0), Int8List(0), Float32List(0));

  final Int32List ids;

  /// [EmbeddingCodec.dims] bytes per line, in the order of [ids].
  final Int8List data;
  final Float32List scales;

  int get length => ids.length;
}

/// Stores one compact vector per transcript line (see [EmbeddingCodec]).
/// Vectors disappear with their line, and when its text changes (a database
/// trigger), so they are made again.
class VectorStore {
  VectorStore(this._db);

  final AppDatabase _db;

  /// Lines without a vector from [model], newest first (what people search
  /// for most). TV / background voices are left out.
  List<PendingLine> pending(String model, {int limit = 16}) {
    final rows = _db.raw.select(
      '''
SELECT s.id, s.text,
  (SELECT p.text FROM segments p WHERE p.conversation_id = s.conversation_id AND p.id < s.id
   ORDER BY p.id DESC LIMIT 1) AS prev
FROM segments s
LEFT JOIN segment_vectors v ON v.segment_id = s.id AND v.model = ?
LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
WHERE v.segment_id IS NULL AND COALESCE(uc.background, 0) = 0
ORDER BY s.id DESC LIMIT ?''',
      [model, limit],
    );
    return [for (final r in rows) PendingLine(r['id']! as int, documentText(r['text']! as String, r['prev'] as String?))];
  }

  /// What is embedded for a line: the line itself, with the line before it
  /// when it is short ("yes, Tuesday" means little alone). Null for lines
  /// with fewer than two words.
  static String? documentText(String text, String? previous) {
    final words = RegExp(r'[\p{L}\p{N}]+', unicode: true).allMatches(text).length;
    if (words < 2) return null;
    if (words < 6 && previous != null && previous.trim().isNotEmpty) return '${previous.trim()}\n${text.trim()}';
    return text.trim();
  }

  /// Saves vectors (null: nothing worth embedding) for [model].
  void put(String model, List<(int, Float32List?)> vectors) {
    _db.transaction(() {
      for (final (id, v) in vectors) {
        if (v == null) {
          _db.raw.execute('INSERT OR REPLACE INTO segment_vectors(segment_id, model, scale, vec) VALUES (?, ?, 0, NULL)', [id, model]);
          continue;
        }
        final q = EmbeddingCodec.quantize(EmbeddingCodec.shorten(v));
        _db.raw.execute(
          'INSERT OR REPLACE INTO segment_vectors(segment_id, model, scale, vec) VALUES (?, ?, ?, ?)',
          [id, model, q.scale, Uint8List.view(q.bytes.buffer, q.bytes.offsetInBytes, q.bytes.length)],
        );
      }
    });
  }

  /// Lines with a vector (or marked done) from [model], and lines in total
  /// that should have one.
  ({int done, int total}) progress(String model) {
    final r = _db.raw.select(
      '''
SELECT
  (SELECT COUNT(*) FROM segment_vectors v JOIN segments s ON s.id = v.segment_id WHERE v.model = ?) AS done,
  (SELECT COUNT(*) FROM segments s LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id
   WHERE COALESCE(uc.background, 0) = 0) AS total''',
      [model],
    ).first;
    final done = r['done']! as int;
    final total = r['total']! as int;
    return (done: done < total ? done : total, total: total);
  }

  /// Changes whenever vectors are added or removed (cheap to check).
  String signature(String model) {
    final r = _db.raw.select(
      'SELECT COUNT(*) AS n, COALESCE(MAX(segment_id), 0) AS m, COALESCE(SUM(segment_id), 0) AS t '
      'FROM segment_vectors WHERE model = ? AND vec IS NOT NULL',
      [model],
    ).first;
    return '${r['n']}:${r['m']}:${r['t']}';
  }

  /// All vectors of [model], packed.
  VectorIndex load(String model) {
    final rows = _db.raw.select(
      'SELECT segment_id AS id, scale, vec FROM segment_vectors WHERE model = ? AND vec IS NOT NULL ORDER BY segment_id',
      [model],
    );
    const dims = EmbeddingCodec.dims;
    final ids = Int32List(rows.length);
    final data = Int8List(rows.length * dims);
    final scales = Float32List(rows.length);
    var i = 0;
    for (final r in rows) {
      final blob = r['vec']! as Uint8List;
      if (blob.length != dims) continue;
      ids[i] = r['id']! as int;
      scales[i] = (r['scale']! as num).toDouble();
      data.setRange(i * dims, (i + 1) * dims, Int8List.view(blob.buffer, blob.offsetInBytes, dims));
      i++;
    }
    return i == rows.length
        ? VectorIndex(ids, data, scales)
        : VectorIndex(Int32List.sublistView(ids, 0, i), Int8List.sublistView(data, 0, i * dims), Float32List.sublistView(scales, 0, i));
  }
}
