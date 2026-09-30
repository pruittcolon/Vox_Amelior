import 'dart:async';
import 'dart:convert';

import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/idle_timeout.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

/// A finding parsed from one line of Gemma's answer.
class ParsedFinding {
  const ParsedFinding({required this.line, required this.category, required this.quote, required this.note});

  /// The part's line number it points at (null if Gemma left it out).
  final int? line;
  final String category;
  final String quote;
  final String note;
}

/// "Go through everything": reads a whole period one part at a time.
///
/// Each part holds about half the context window of transcript. Gemma gets
/// the task, the answer format, a short "found so far" note and the part;
/// its answer is saved straight away. For list reviews the app parses and
/// counts every finding itself (exact totals, links to the exact line);
/// Gemma then writes a short overview. Summary reviews are merged in rounds
/// until one answer is left. Every step is small and saved, so a review
/// survives restarts and can be paused at any time.
class ReviewEngine {
  ReviewEngine({
    required this.reviews,
    required this.transcripts,
    required this.llm,
    this.clock = systemClock,
    this.idleTimeout = const Duration(minutes: 4),
  });

  final ReviewRepository reviews;
  final TranscriptRepository transcripts;
  final LlmEngine llm;
  final Clock clock;
  final Duration idleTimeout;

  static const String system =
      "You are Vox, reviewing transcripts of the user's own conversations. They come from automatic speech "
      'recognition and may contain mistakes. Follow the task and the answer format exactly. Use only the '
      'transcript; never invent lines, quotes or line numbers.';

  /// Plans a review over [from]–[to] and saves it. Returns its id, or null
  /// when nothing was said in that period.
  int? start({
    required String title,
    required String prompt,
    required String format,
    required ReviewKind kind,
    required String periodLabel,
    required DateTime from,
    required DateTime to,
    required ContextBudget budget,
    List<String> focus = const [],
  }) {
    final segments = transcripts.between(from, to, limit: 1000000);
    if (segments.isEmpty) return null;
    final parts = plan(segments, budget.chunkTokens);
    return reviews.create(
      title: title,
      prompt: prompt,
      format: format,
      kind: kind,
      periodLabel: periodLabel,
      from: from,
      to: to,
      focus: focus,
      chunkTokens: budget.chunkTokens,
      contextTokens: budget.contextTokens,
      parts: parts,
    );
  }

  /// Splits [segments] into parts of about [chunkTokens] each.
  static List<List<SegmentView>> plan(List<SegmentView> segments, int chunkTokens) {
    final parts = <List<SegmentView>>[];
    var current = <SegmentView>[];
    var size = 0;
    for (final s in segments) {
      final t = ContextBudget.estimateTokens(_line(99, s)) + 1;
      if (current.isNotEmpty && size + t > chunkTokens) {
        parts.add(current);
        current = [];
        size = 0;
      }
      current.add(s);
      size += t;
    }
    if (current.isNotEmpty) parts.add(current);
    return parts;
  }

  /// Does one unit of work on run [runId]. Returns true while work remains.
  Future<bool> step(int runId) async {
    final run = reviews.run(runId);
    if (run == null || !run.isActive) return false;
    final budget = ContextBudget(run.contextTokens, chunkTokens: run.chunkTokens);
    try {
      final chunk = reviews.nextPending(runId);
      if (chunk != null) {
        await _part(run, chunk, budget);
        return true;
      }
      return run.kind == ReviewKind.list ? await _finishList(run, budget) : await _mergeStep(run, budget);
    } on LlmUnavailable catch (e) {
      // The model is missing or cannot start: pause rather than burn through
      // every part with the same error.
      reviews.setStatus(runId, ReviewStatus.paused, error: e.message);
      return false;
    }
  }

  // ---- parts -------------------------------------------------------------

  Future<void> _part(ReviewRun run, ReviewChunk chunk, ContextBudget budget) async {
    final byId = {for (final s in transcripts.segmentsByIds(chunk.segmentIds)) s.id: s};
    final lines = [for (final id in chunk.segmentIds) ?byId[id]];
    if (lines.isEmpty) {
      reviews.completeChunk(run.id, chunk.idx, 'Nothing left in this part (deleted).', const []);
      return;
    }
    final prompt = partPrompt(run, chunk.idx, lines, carry: _carry(run, budget));
    String answer;
    try {
      answer = await _ask(prompt, budget.replyTokens);
    } on LlmUnavailable {
      rethrow;
    } on Object catch (e) {
      reviews.failChunk(run.id, chunk.idx, friendlyLlmError(e));
      return;
    }
    final found = run.kind == ReviewKind.list ? _findings(run, chunk.idx, answer, lines) : const <ReviewItem>[];
    reviews.completeChunk(run.id, chunk.idx, answer.trim(), found);
  }

  String partPrompt(ReviewRun run, int idx, List<SegmentView> lines, {String carry = ''}) {
    final b = StringBuffer()
      ..writeln('Task: ${run.prompt.trim()}')
      ..writeln();
    if (run.focus.isNotEmpty) {
      final names = {for (final s in lines) if (run.focus.contains(s.speakerId)) s.speakerLabel};
      b
        ..writeln('Only report things said by: ${names.isEmpty ? 'the chosen people' : names.join(', ')}.')
        ..writeln();
    }
    b
      ..writeln('Answer format:')
      ..writeln(run.format.trim())
      ..writeln();
    if (carry.isNotEmpty) {
      b
        ..writeln(run.kind == ReviewKind.list ? 'Found in earlier parts (do not repeat these): $carry' : 'Notes from earlier parts: $carry')
        ..writeln();
    }
    b.writeln('Transcript, part ${idx + 1} of ${run.totalChunks} (${run.periodLabel}):');
    String? lastHeader;
    for (var i = 0; i < lines.length; i++) {
      final s = lines[i];
      final header = _dayHeader(s.startedAt);
      if (header != lastHeader) {
        b.writeln('--- $header ---');
        lastHeader = header;
      }
      b.writeln(_line(i + 1, s));
    }
    return b.toString().trim();
  }

  String _carry(ReviewRun run, ContextBudget budget) {
    final maxChars = ContextBudget.charsFor(budget.carryTokens);
    if (run.kind == ReviewKind.summary) {
      final previous = reviews.chunks(run.id).where((c) => c.status == 'done' && (c.answer ?? '').isNotEmpty).lastOrNull;
      final text = previous?.answer ?? '';
      return text.length <= maxChars ? text : '${text.substring(0, maxChars)}…';
    }
    final counts = countByCategory(reviews.items(run.id));
    if (counts.isEmpty) return '';
    final b = StringBuffer();
    for (final e in counts.entries) {
      final piece = '${e.key} ×${e.value}; ';
      if (b.length + piece.length > maxChars) break;
      b.write(piece);
    }
    return b.toString().trim();
  }

  List<ReviewItem> _findings(ReviewRun run, int idx, String answer, List<SegmentView> lines) {
    final out = <ReviewItem>[];
    for (final f in parseFindings(answer)) {
      final seg = f.line != null && f.line! >= 1 && f.line! <= lines.length ? lines[f.line! - 1] : null;
      if (run.focus.isNotEmpty && seg != null && !run.focus.contains(seg.speakerId)) continue;
      out.add(ReviewItem(
        chunkIdx: idx,
        segmentId: seg?.id,
        category: f.category,
        quote: f.quote.isEmpty && seg != null ? seg.text : f.quote,
        note: f.note,
        speaker: seg?.speakerLabel,
        saidAt: seg?.startedAt,
      ));
    }
    return out;
  }

  static final RegExp _numbered = RegExp(r'^\s*(?:[-*•]\s*|\d+[.)]\s*)?\[\s*(?:#|line\s*)?(\d+)\s*\]\s*(.*)$', caseSensitive: false);
  static final RegExp _bulleted = RegExp(r'^\s*[-*•]\s+(.*)$');
  static final RegExp _separator = RegExp(r'\s+[—–-]{1,2}\s+|\s*—\s*');

  /// Reads "- [12] "quote" — Category — note" lines (tolerating the usual
  /// variations). "NONE" and chatter are ignored.
  static List<ParsedFinding> parseFindings(String answer) {
    final out = <ParsedFinding>[];
    for (final raw in const LineSplitter().convert(answer)) {
      if (raw.trim().isEmpty || RegExp(r'^\W*none\W*$', caseSensitive: false).hasMatch(raw)) continue;
      int? line;
      String rest;
      final m = _numbered.firstMatch(raw);
      if (m != null) {
        line = int.tryParse(m.group(1)!);
        rest = m.group(2)!;
      } else {
        final b = _bulleted.firstMatch(raw);
        if (b == null || !_separator.hasMatch(b.group(1)!)) continue;
        rest = b.group(1)!;
      }
      // A quoted quote may itself contain dashes: take it whole first.
      String? quote;
      final t = rest.trimLeft();
      const closers = {'"': '"', '“': '”'};
      final close = t.isEmpty ? null : closers[t[0]];
      if (close != null) {
        final end = t.indexOf(close, 1);
        if (end > 0) {
          quote = t.substring(1, end).trim();
          rest = t.substring(end + 1).replaceFirst(RegExp(r'^\s*[—–-]{1,2}\s*'), '');
        }
      }
      final parts = rest.split(_separator).map((p) => p.trim()).where((p) => p.isNotEmpty).toList();
      if (quote == null) {
        if (parts.isEmpty) continue;
        quote = _unquote(parts.removeAt(0));
      }
      final category = parts.isNotEmpty ? _tidyCategory(parts.first) : 'Other';
      final note = parts.length > 1 ? parts.sublist(1).join(' — ') : '';
      out.add(ParsedFinding(line: line, category: category, quote: quote, note: note));
    }
    return out;
  }

  /// Findings per category, most common first (case-insensitive grouping).
  static Map<String, int> countByCategory(List<ReviewItem> items) {
    final counts = <String, int>{};
    final display = <String, String>{};
    for (final it in items) {
      final key = it.category.toLowerCase();
      display.putIfAbsent(key, () => it.category);
      counts[key] = (counts[key] ?? 0) + 1;
    }
    final sorted = counts.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return {for (final e in sorted) display[e.key]!: e.value};
  }

  static String _unquote(String s) {
    var t = s.trim();
    const pairs = {'"': '"', '“': '”', "'": "'", '‘': '’'};
    for (final e in pairs.entries) {
      if (t.length >= 2 && t.startsWith(e.key) && t.endsWith(e.value)) {
        t = t.substring(1, t.length - 1);
        break;
      }
    }
    return t.trim();
  }

  static String _tidyCategory(String s) {
    final t = s.trim().replaceAll(RegExp(r'[.:;,]+$'), '').replaceAll(RegExp(r'^\*+|\*+$'), '').trim();
    if (t.isEmpty) return 'Other';
    return t[0].toUpperCase() + t.substring(1);
  }

  // ---- finishing -----------------------------------------------------------

  Future<bool> _finishList(ReviewRun run, ContextBudget budget) async {
    final items = reviews.items(run.id);
    final counts = countByCategory(items);
    final failed = reviews.chunks(run.id).where((c) => c.status == 'failed').length;
    if (items.isEmpty) {
      reviews.finish(run.id, failed == 0 ? 'Nothing found in ${run.periodLabel}.' : 'Nothing found in the parts that could be read.');
      return false;
    }
    final b = StringBuffer()
      ..writeln('Task: ${run.prompt.trim()}')
      ..writeln()
      ..writeln('Across ${run.periodLabel}, ${items.length} findings were counted in ${run.totalChunks} parts:');
    for (final e in counts.entries) {
      b.writeln('- ${e.key}: ${e.value}');
    }
    b
      ..writeln()
      ..writeln('Examples:');
    final exampleChars = ContextBudget.charsFor(budget.chunkTokens ~/ 2);
    var used = 0;
    for (final it in items) {
      final line = '- "${it.quote}" — ${it.category}${it.speaker == null ? '' : ' (${it.speaker})'}';
      if (used + line.length > exampleChars) break;
      b.writeln(line);
      used += line.length;
    }
    b
      ..writeln()
      ..writeln('Write a short overview (3 to 6 sentences): the most common patterns, who they came from, and '
          'anything notable. Do not list every finding again and do not change the counts.');
    String overview;
    try {
      overview = (await _ask(b.toString(), budget.replyTokens)).trim();
    } on LlmUnavailable {
      rethrow;
    } on Object {
      overview = '';
    }
    final summary = StringBuffer()
      ..writeln('${items.length} found:')
      ..writeln([for (final e in counts.entries) '${e.key} ${e.value}'].join(' · '));
    if (overview.isNotEmpty) {
      summary
        ..writeln()
        ..write(overview);
    }
    reviews.finish(run.id, summary.toString().trim());
    return false;
  }

  Future<bool> _mergeStep(ReviewRun run, ContextBudget budget) async {
    var level = <String>[];
    var next = <String>[];
    final saved = run.mergeJson == null ? null : jsonDecode(run.mergeJson!);
    if (saved is Map) {
      level = (saved['level'] as List<Object?>).whereType<String>().toList();
      next = (saved['next'] as List<Object?>).whereType<String>().toList();
    } else {
      level = [
        for (final c in reviews.chunks(run.id))
          if (c.status == 'done' && (c.answer ?? '').trim().isNotEmpty) 'Part ${c.idx + 1}: ${c.answer!.trim()}',
      ];
    }
    if (level.isEmpty) {
      if (next.length <= 1) {
        reviews.finish(run.id, next.isEmpty ? 'Nothing was said in ${run.periodLabel}.' : _stripPartLabel(next.single));
        return false;
      }
      level = next;
      next = [];
    }
    // Take as many notes as fit in one part's budget.
    final batch = <String>[level.removeAt(0)];
    var size = ContextBudget.estimateTokens(batch.first);
    while (level.isNotEmpty && size + ContextBudget.estimateTokens(level.first) <= budget.chunkTokens) {
      size += ContextBudget.estimateTokens(level.first);
      batch.add(level.removeAt(0));
    }
    if (batch.length == 1) {
      next.add(batch.single);
    } else {
      final prompt = StringBuffer()
        ..writeln('Task: ${run.prompt.trim()}')
        ..writeln()
        ..writeln('Answer format:')
        ..writeln(run.format.trim())
        ..writeln()
        ..writeln('Below are notes from consecutive parts of ${run.periodLabel}. Combine them into one answer to the task, '
            'in the answer format. Keep who said what and when. Do not invent anything.')
        ..writeln();
      for (final t in batch) {
        prompt.writeln(t);
      }
      next.add((await _ask(prompt.toString(), budget.replyTokens)).trim());
    }
    reviews.setMerge(run.id, jsonEncode({'level': level, 'next': next}));
    return true;
  }

  static String _stripPartLabel(String s) => s.replaceFirst(RegExp(r'^Part \d+: '), '');

  // ---- model -------------------------------------------------------------

  Future<String> _ask(String prompt, int replyTokens) async {
    await llm.ensureLoaded();
    try {
      return await _once(prompt, replyTokens);
    } on LlmUnavailable {
      rethrow;
    } on Object {
      // One more try in a safer mode (e.g. CPU) before giving up on this part.
      if (!await llm.recover()) rethrow;
      await llm.ensureLoaded();
      return _once(prompt, replyTokens);
    }
  }

  Future<String> _once(String prompt, int replyTokens) async {
    final session = await llm.openSession(system: system, maxReplyTokens: replyTokens);
    try {
      final out = StringBuffer();
      await for (final e in withIdleTimeout(session.send(prompt), idleTimeout)) {
        if (e is LlmText) out.write(e.text);
      }
      return out.toString();
    } finally {
      await session.close();
    }
  }

  static String _line(int n, SegmentView s) =>
      '[$n] ${PromptBuilder.hm(s.startedAt)} ${s.speakerLabel}${s.overlap ? ' (over someone else)' : ''}: ${s.text}';

  static String _dayHeader(DateTime d) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return '${days[d.weekday - 1]} ${d.day} ${months[d.month - 1]}';
  }
}
