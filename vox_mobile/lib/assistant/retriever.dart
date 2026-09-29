import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

/// Finds the parts of past conversations relevant to a question.
class Retriever {
  Retriever(this._transcripts, {this.maxHits = 8, this.contextLines = 2, this.maxChars = 6000});

  final TranscriptRepository _transcripts;

  /// Number of best-matching utterances to expand.
  final int maxHits;

  /// Neighbouring utterances included on each side of a hit.
  final int contextLines;

  /// Budget for the text handed to the model (small on-device context).
  final int maxChars;

  /// Returns chronologically ordered, de-duplicated segments.
  List<SegmentView> retrieve(ParsedQuery q) {
    final query = SegmentQuery(
      keywords: q.keywords,
      speakerId: q.speaker?.id,
      from: q.from,
      to: q.to,
      limit: maxHits,
    );

    var hits = _transcripts.search(query);
    if (hits.isEmpty && q.keywords.isNotEmpty && (q.speaker != null || q.from != null)) {
      // Keywords may be paraphrased; fall back to "what happened in that window".
      hits = _transcripts.search(SegmentQuery(
        speakerId: q.speaker?.id,
        from: q.from,
        to: q.to,
        limit: maxHits * 2,
      ));
    }
    if (hits.isEmpty) return const [];

    final byId = <int, SegmentView>{};
    for (final h in hits) {
      for (final s in _transcripts.around(h, before: contextLines, after: contextLines)) {
        byId[s.id] = s;
      }
    }
    final ordered = byId.values.toList()..sort((a, b) => a.id.compareTo(b.id));
    return _trim(ordered, hits);
  }

  /// Keeps the total under [maxChars], preferring segments that were direct hits.
  List<SegmentView> _trim(List<SegmentView> ordered, List<SegmentView> hits) {
    var total = ordered.fold<int>(0, (a, s) => a + s.text.length + 30);
    if (total <= maxChars) return ordered;
    final hitIds = hits.map((h) => h.id).toSet();
    final keep = List<SegmentView>.of(ordered);
    for (final s in ordered.where((s) => !hitIds.contains(s.id)).toList().reversed) {
      if (total <= maxChars) break;
      keep.remove(s);
      total -= s.text.length + 30;
    }
    while (total > maxChars && keep.length > 1) {
      final removed = keep.removeAt(0);
      total -= removed.text.length + 30;
    }
    return keep;
  }
}
