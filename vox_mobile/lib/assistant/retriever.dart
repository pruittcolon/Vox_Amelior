import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

/// Finds the parts of past conversations relevant to a question.
class Retriever {
  Retriever(this._transcripts, {this.maxHits = 8, this.contextLines = 2, this.maxChars = 7000});

  final TranscriptRepository _transcripts;

  /// Number of best-matching utterances to expand.
  final int maxHits;

  /// Neighbouring utterances included on each side of a hit.
  final int contextLines;

  /// Budget for the text handed to the model (small on-device context).
  final int maxChars;

  /// Returns chronologically ordered, de-duplicated segments.
  List<SegmentView> retrieve(ParsedQuery q) {
    final window = q.window;
    if (q.intent == QueryIntent.overview && window != null) {
      return timeline(window.from, window.to, speakerId: q.speaker?.id);
    }

    final hits = _transcripts.search(SegmentQuery(
      keywords: q.keywords,
      speakerId: q.speaker?.id,
      from: q.from,
      to: q.to,
      limit: maxHits,
    ));

    final byId = <int, SegmentView>{};
    for (final h in hits) {
      for (final s in _transcripts.around(h, before: contextLines, after: contextLines)) {
        byId[s.id] = s;
      }
    }

    // Few direct matches inside a time window: the words were probably
    // paraphrased, so also read across that period.
    if (window != null && hits.length < 3) {
      final budget = maxChars - _size(byId.values);
      for (final s in timeline(window.from, window.to, speakerId: q.speaker?.id, maxChars: budget)) {
        byId[s.id] = s;
      }
    }
    final ordered = byId.values.toList()..sort((a, b) => a.id.compareTo(b.id));
    return _trim(ordered, hits);
  }

  /// An even sample of everything said between [from] and [to], so every
  /// day and every conversation in the period is represented.
  List<SegmentView> timeline(DateTime from, DateTime to, {String? speakerId, int? maxChars}) {
    final budget = maxChars ?? this.maxChars;
    if (budget <= 0) return const [];
    var all = _transcripts.between(from, to, limit: 20000);
    if (speakerId != null) all = all.where((s) => s.speakerId == speakerId).toList();
    if (all.isEmpty || _size(all) <= budget) return all;

    final byConversation = <int, List<SegmentView>>{};
    for (final s in all) {
      byConversation.putIfAbsent(s.conversationId, () => []).add(s);
    }
    // Each conversation gets a share of the budget proportional to its
    // length, but never less than its opening and closing lines.
    final total = _size(all);
    final picked = <SegmentView>[];
    for (final lines in byConversation.values) {
      final share = (budget * _size(lines) / total).floor();
      picked.addAll(_sample(lines, share));
    }
    picked.sort((a, b) => a.id.compareTo(b.id));
    while (_size(picked) > budget && picked.length > byConversation.length) {
      // Drop from the longest conversation first.
      final counts = <int, int>{};
      for (final s in picked) {
        counts[s.conversationId] = (counts[s.conversationId] ?? 0) + 1;
      }
      final longest = counts.entries.reduce((a, b) => a.value >= b.value ? a : b).key;
      final inLongest = picked.where((s) => s.conversationId == longest).toList();
      picked.remove(inLongest[inLongest.length ~/ 2]);
    }
    return picked;
  }

  List<SegmentView> _sample(List<SegmentView> lines, int budget) {
    if (lines.length <= 2 || _size(lines) <= budget) return lines;
    final avg = _size(lines) / lines.length;
    final count = (budget / avg).floor().clamp(2, lines.length);
    if (count >= lines.length) return lines;
    final step = (lines.length - 1) / (count - 1);
    return {for (var i = 0; i < count; i++) lines[(i * step).round()]}.toList();
  }

  /// Keeps the total under [maxChars], preferring segments that were direct hits.
  List<SegmentView> _trim(List<SegmentView> ordered, List<SegmentView> hits) {
    var total = _size(ordered);
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

  static int _size(Iterable<SegmentView> s) => s.fold<int>(0, (a, x) => a + x.text.length + 30);
}
