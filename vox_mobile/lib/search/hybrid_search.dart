import 'dart:async';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/search/vector_store.dart';

/// How to search.
enum SearchMode {
  /// Words and meaning together (the default).
  smart,

  /// Only lines containing the words (full-text, ranked by BM25).
  words,

  /// Only lines that mean something similar (EmbeddingGemma).
  meaning,
}

/// Narrows a search: people, tones, a period. TV / background voices are
/// always left out.
class SearchFilters {
  const SearchFilters({this.speakerIds = const {}, this.emotions = const {}, this.from, this.to});

  final Set<String> speakerIds;
  final Set<String> emotions;
  final DateTime? from;
  final DateTime? to;
}

/// A line found by a search, and how it was found.
class SearchHit {
  const SearchHit(this.segment, {required this.byWords, required this.byMeaning, this.similarity});

  final SegmentView segment;
  final bool byWords;
  final bool byMeaning;

  /// Cosine similarity to the query (meaning hits only).
  final double? similarity;
}

/// Search over every line: full-text (SQLite FTS5, BM25) and meaning
/// (EmbeddingGemma vectors), combined with Reciprocal Rank Fusion. RRF only
/// uses each list's ranks, so the two very different scores never have to
/// be put on one scale. Also picks the lines Gemma reads for a question (RAG).
class HybridSearch {
  HybridSearch({
    required this._db,
    required this.transcripts,
    required this.vectors,
    required this.embedder,
    required this.modelId,
  });

  final AppDatabase _db;
  final TranscriptRepository transcripts;
  final VectorStore vectors;

  /// The embedder, or null when meaning search is off or not installed.
  final AsyncEmbedder? Function() embedder;
  final String modelId;

  /// RRF's constant: the usual 60 keeps one list's top hit from drowning the other.
  static const int rrfK = 60;

  /// Below this cosine similarity a line is never considered related.
  static const double minSimilarity = 0.30;

  /// Lines more than this below the best match are left out. EmbeddingGemma's
  /// scores are compressed (in CI: related lines 0.44–0.62, unrelated up to
  /// 0.43), so a cut relative to the best match separates them better than
  /// any fixed threshold.
  static const double relativeMargin = 0.12;

  VectorIndex _index = VectorIndex.empty;
  String _signature = '';

  /// Meaning search can run (model on, some lines embedded).
  bool get meaningReady => embedder() != null && _currentIndex().length > 0;

  VectorIndex _currentIndex() {
    final sig = vectors.signature(modelId);
    if (sig != _signature) {
      _index = vectors.load(modelId);
      _signature = sig;
    }
    return _index;
  }

  /// Lines containing [query]'s words, best first.
  List<SegmentView> words(String query, SearchFilters f, {int limit = 50}) {
    final keywords = query.trim().split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
    if (keywords.isEmpty) return const [];
    return transcripts.search(SegmentQuery(
      keywords: keywords,
      speakerIds: f.speakerIds,
      emotions: f.emotions,
      from: f.from,
      to: f.to,
      limit: limit,
    ));
  }

  /// Lines meaning something like [query] (id and similarity), best first.
  /// Empty when meaning search is not available.
  Future<List<(int, double)>> meaning(String query, SearchFilters f, {int limit = 50, double min = minSimilarity}) async {
    final e = embedder();
    if (e == null || query.trim().isEmpty) return const [];
    final index = _currentIndex();
    if (index.length == 0) return const [];
    final q = EmbeddingCodec.shorten((await e.embed([query.trim()], task: EmbedTask.query)).single);
    final allowed = _allowed(f);
    const dims = EmbeddingCodec.dims;
    final scored = <(int, double)>[];
    for (var i = 0; i < index.length; i++) {
      final id = index.ids[i];
      if (!allowed.contains(id)) continue;
      final s = EmbeddingCodec.score(q, index.data, i * dims, index.scales[i]);
      if (s >= min) scored.add((id, s));
    }
    scored.sort((a, b) => b.$2.compareTo(a.$2));
    if (scored.isEmpty) return const [];
    final cut = scored.first.$2 - relativeMargin;
    return scored.where((e) => e.$2 >= cut).take(limit).toList();
  }

  /// Ids of lines the filters allow (not TV, chosen people, tones, period).
  Set<int> _allowed(SearchFilters f) {
    final where = <String>['COALESCE(uc.background, 0) = 0'];
    final args = <Object?>[];
    if (f.speakerIds.isNotEmpty) {
      where.add('s.speaker_id IN (${List.filled(f.speakerIds.length, '?').join(',')})');
      args.addAll(f.speakerIds);
    }
    if (f.emotions.isNotEmpty) {
      final marks = List.filled(f.emotions.length, '?').join(',');
      where.add('(s.emotion IN ($marks) OR s.sound IN ($marks))');
      args
        ..addAll(f.emotions)
        ..addAll(f.emotions);
    }
    if (f.from != null) {
      where.add('s.started_at >= ?');
      args.add(f.from!.millisecondsSinceEpoch);
    }
    if (f.to != null) {
      where.add('s.started_at < ?');
      args.add(f.to!.millisecondsSinceEpoch);
    }
    return {
      for (final r in _db.raw.select(
        'SELECT s.id FROM segments s LEFT JOIN unknown_clusters uc ON uc.id = s.cluster_id WHERE ${where.join(' AND ')}',
        args,
      ))
        r['id']! as int,
    };
  }

  /// Reciprocal Rank Fusion of several best-first id lists: each id scores
  /// the sum of 1 / ([k] + rank) over the lists it appears in.
  static List<(int, double)> fuse(List<List<int>> rankings, {int k = rrfK}) {
    final scores = <int, double>{};
    for (final list in rankings) {
      for (var r = 0; r < list.length; r++) {
        scores[list[r]] = (scores[list[r]] ?? 0) + 1 / (k + r + 1);
      }
    }
    final out = scores.entries.map((e) => (e.key, e.value)).toList()
      ..sort((a, b) => b.$2 != a.$2 ? b.$2.compareTo(a.$2) : b.$1.compareTo(a.$1));
    return out;
  }

  /// Searches in [mode]. Without meaning search, smart is words only.
  Future<List<SearchHit>> search(String query, SearchFilters f, {SearchMode mode = SearchMode.smart, int limit = 50}) async {
    final byWords = mode == SearchMode.meaning ? const <SegmentView>[] : words(query, f, limit: limit);
    var byMeaning = const <(int, double)>[];
    if (mode != SearchMode.words) {
      try {
        byMeaning = await meaning(query, f, limit: limit);
      } on Object {
        // The model failed: words still work.
        if (mode == SearchMode.meaning) rethrow;
      }
    }
    final wordIds = {for (final s in byWords) s.id};
    final similarity = {for (final (id, s) in byMeaning) id: s};
    final ranked = fuse([
      [for (final s in byWords) s.id],
      [for (final (id, _) in byMeaning) id],
    ]).take(limit).map((e) => e.$1).toList();
    final known = {for (final s in byWords) s.id: s};
    final missing = [for (final id in ranked) if (!known.containsKey(id)) id];
    for (final s in transcripts.segmentsByIds(missing)) {
      known[s.id] = s;
    }
    return [
      for (final id in ranked)
        if (known[id] case final s?)
          SearchHit(s, byWords: wordIds.contains(id), byMeaning: similarity.containsKey(id), similarity: similarity[id]),
    ];
  }

  /// Lines worth reading for [question] by meaning, best first, for the
  /// assistant (RAG). Empty when meaning search is not available or too slow.
  Future<List<int>> hintsFor(String question, {int limit = 12, Duration timeout = const Duration(seconds: 6)}) async {
    try {
      final hits = await meaning(question, const SearchFilters(), limit: limit).timeout(timeout);
      return [for (final (id, _) in hits) id];
    } on Object {
      return const [];
    }
  }
}

/// Embeds lines that have no vector yet, in small batches, newest first, so
/// meaning search catches up in the background.
class SearchIndexer {
  SearchIndexer({
    required this.store,
    required this.embedder,
    required this.modelId,
    this.batch = 8,
    this.pause = const Duration(milliseconds: 150),
  });

  final VectorStore store;
  final AsyncEmbedder? Function() embedder;
  final String modelId;
  final int batch;

  /// Shortest rest between batches. The rest is at least as long as the
  /// batch took, so indexing uses at most half the CPU it could and
  /// listening (transcribing at the same time) is never starved.
  final Duration pause;

  /// Lines done and in total; updated as it works.
  final StreamController<({int done, int total})> _progress = StreamController.broadcast();
  Stream<({int done, int total})> get progress => _progress.stream;

  bool _running = false;
  bool _again = false;
  bool _stopped = false;

  /// Last error, if indexing stopped because of one.
  Object? error;

  bool get isRunning => _running;

  /// Starts (or continues) indexing. Returns when nothing is left to do.
  Future<void> run() async {
    if (_running) {
      _again = true;
      return;
    }
    _running = true;
    _stopped = false;
    error = null;
    try {
      do {
        _again = false;
        while (!_stopped) {
          final e = embedder();
          if (e == null) return;
          final lines = store.pending(modelId, limit: batch);
          if (lines.isEmpty) break;
          final texts = [for (final l in lines) if (l.text != null) l.text!];
          final watch = Stopwatch()..start();
          final vectors = texts.isEmpty ? const <Float32List>[] : await e.embed(texts, task: EmbedTask.document);
          final took = watch.elapsed;
          var next = 0;
          store.put(modelId, [for (final l in lines) (l.id, l.text == null ? null : vectors[next++])]);
          if (!_progress.isClosed) _progress.add(store.progress(modelId));
          await Future<void>.delayed(took > pause ? took : pause);
        }
      } while (_again && !_stopped);
    } on Object catch (e) {
      error = e;
    } finally {
      _running = false;
    }
  }

  void stop() => _stopped = true;

  void dispose() {
    stop();
    unawaited(_progress.close());
  }
}
