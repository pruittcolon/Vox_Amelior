/// Reads plain text, Markdown or a Narrator script into a [Script].
///
/// The script format (see GUIDE.md) is plain text with a few extras:
///
///     ---
///     title: Why Is the Sky Blue?
///     style: podcast
///     music: curious
///     summary: Why sunlight paints the sky blue and sunsets orange.
///     topics: physics, light
///     voices:
///       HOST: Radio Host
///       GUEST: The Professor
///     index:
///       - Light is a wave | White light holds every colour.
///     ---
///     [music: intro]
///     HOST: Welcome back to Curious Minds!
///     ## Part 1: Light is a wave
///     [pause 2]
///     [sound: chime]
///
/// Anything else (a book, an article, notes) just reads as narration.
library;

import 'dart:math' as math;

import 'script.dart';
import 'text_cleaner.dart';

Script parseScript(String raw, {String fallbackTitle = 'Untitled'}) {
  var text = raw.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
  if (text.startsWith('﻿')) text = text.substring(1);

  final front = _FrontMatter.parse(text);
  text = front.body;
  final declared = {for (final k in front.voices.keys) k.toUpperCase()};

  final lines = text.split('\n');
  final speakerCounts = _countSpeakerTags(lines, declared);
  bool isSpeaker(String name) {
    final key = _canonical(name);
    if (declared.contains(key)) return true;
    if (_notSpeakers.contains(key)) return false;
    // An all-caps tag used at least twice (HOST:, DR. LEE:) is a speaker.
    return (speakerCounts[key] ?? 0) >= 2;
  }

  final hasSpeakers = declared.isNotEmpty || speakerCounts.values.any((n) => n >= 2);
  final blocks = <Block>[];
  String? title = front.title;
  final paragraph = <String>[];
  String? paragraphSpeaker;
  String? lastSpeaker;
  var inComment = false;

  void flush() {
    if (paragraph.isEmpty) return;
    final joined = displayText(paragraph.join(' '));
    paragraph.clear();
    if (joined.isEmpty || !isSpeakable(joined)) return;
    _addLine(blocks, joined, paragraphSpeaker);
    lastSpeaker = paragraphSpeaker;
  }

  for (var rawLine in lines) {
    // HTML comments are notes for the author, not for the listener.
    if (inComment) {
      final close = rawLine.indexOf('-->');
      if (close < 0) continue;
      rawLine = rawLine.substring(close + 3);
      inComment = false;
    }
    rawLine = rawLine.replaceAll(RegExp(r'<!--.*?-->'), '');
    final open = rawLine.indexOf('<!--');
    if (open >= 0) {
      rawLine = rawLine.substring(0, open);
      inComment = true;
    }

    final line = rawLine.trim();
    if (line.isEmpty) {
      flush();
      continue;
    }
    if (line.startsWith('//')) continue;

    // Scene break.
    if (RegExp(r'^([-*_])(\s*\1){2,}$').hasMatch(line)) {
      flush();
      blocks.add(const Block.pause(1.2));
      continue;
    }

    // Markdown heading.
    final h = RegExp(r'^(#{1,6})\s+(.+?)\s*#*$').firstMatch(line);
    if (h != null) {
      flush();
      final headingText = displayText(h[2]!);
      final level = h[1]!.length;
      if (level == 1 && title == null && blocks.isEmpty) {
        title = headingText;
      }
      if (headingText.isNotEmpty) blocks.add(Block.heading(headingText, level: level));
      paragraphSpeaker = null;
      lastSpeaker = null;
      continue;
    }

    // A tag on its own line: [pause 2], [music: intro], [sound: chime], [laughs].
    final tag = RegExp(r'^\[([^\]]+)\]$').firstMatch(line);
    if (tag != null) {
      flush();
      final block = _parseTag(tag[1]!);
      if (block != null) blocks.add(block);
      continue;
    }

    // Stage directions like "(Music fades.)" in a dialogue script.
    if (hasSpeakers && RegExp(r'^\(.{1,160}\)$').hasMatch(line)) {
      flush();
      continue;
    }

    // "Chapter 3" / "PART TWO" alone on a line in a book.
    if (paragraph.isEmpty && _looksLikeChapter(line)) {
      flush();
      blocks.add(Block.heading(displayText(line), level: 2));
      paragraphSpeaker = null;
      lastSpeaker = null;
      continue;
    }

    // HOST: Hello there.   **Host:** Hello.   Dr. Lee: Hi.
    final sp = _speakerLine.firstMatch(line);
    if (sp != null && isSpeaker(sp[1]!)) {
      flush();
      paragraphSpeaker = _canonical(sp[1]!);
      final rest = sp[2]!.trim();
      if (rest.isNotEmpty) _appendWithInlineTags(rest, paragraph, blocks, flush);
      continue;
    }

    // List items and quotes become their own short paragraphs.
    final item = RegExp(r'^([-*+•]|\d{1,3}[.)])\s+(.*)$').firstMatch(line);
    if (item != null) {
      flush();
      if (paragraphSpeaker == null && hasSpeakers) paragraphSpeaker = lastSpeaker;
      final marker = item[1]!;
      final body = RegExp(r'^\d').hasMatch(marker) ? '${marker.replaceAll(')', '.')} ${item[2]}' : item[2]!;
      _appendWithInlineTags(body, paragraph, blocks, flush);
      flush();
      continue;
    }
    final quoted = line.startsWith('>') ? line.replaceFirst(RegExp(r'^>+\s?'), '') : line;

    if (paragraph.isEmpty) {
      // A new untagged paragraph in a dialogue continues the last speaker
      // (Claude sometimes splits one long answer into paragraphs).
      paragraphSpeaker = hasSpeakers ? lastSpeaker : null;
    }
    _appendWithInlineTags(quoted, paragraph, blocks, flush);
  }
  flush();

  final finalTitle = (title == null || title.isEmpty) ? fallbackTitle : title;
  final tidy = _tidy(blocks);
  return Script(
    title: finalTitle,
    blocks: tidy,
    style: front.style,
    music: front.music,
    voices: {for (final e in front.voices.entries) e.key.toUpperCase(): e.value},
    description: front.description,
    topics: front.topics,
    chapters: _chapters(tidy, front.index),
  );
}

/// The episode's chapters: its top-level headings, plus the opening before
/// the first one, named and summarised by the script's `index:`.
///
/// Index entries are matched to headings by title; if the titles don't line
/// up, by position (with or without an entry for the opening).
List<Chapter> _chapters(List<Block> blocks, List<(String, String?)> index) {
  var heads = [
    for (var i = 0; i < blocks.length; i++)
      if (blocks[i].kind == BlockKind.heading) i,
  ];
  // A leading "# Title" that names the document is not a chapter.
  if (heads.length > 1 && heads.first == 0 && blocks[0].level == 1 && blocks.skip(1).any((b) => b.kind == BlockKind.heading && b.level > 1)) {
    heads = heads.sublist(1);
  }
  if (heads.isEmpty) return const [];
  final level = heads.map((i) => blocks[i].level).reduce(math.min);
  heads = heads.where((i) => blocks[i].level == level).toList();
  final opening = blocks.take(heads.first).any((b) => b.kind == BlockKind.line);

  // match[h] = index entry for heading h.
  final match = List<int?>.filled(heads.length, null);
  var from = 0;
  for (var h = 0; h < heads.length; h++) {
    for (var j = from; j < index.length; j++) {
      if (_sameTitle(blocks[heads[h]].text, index[j].$1)) {
        match[h] = j;
        from = j + 1;
        break;
      }
    }
  }
  int? openingEntry;
  if (index.isNotEmpty && match.contains(null)) {
    if (index.length == heads.length + (opening ? 1 : 0)) {
      final skip = opening ? 1 : 0;
      for (var h = 0; h < heads.length; h++) {
        match[h] = h + skip;
      }
      if (opening) openingEntry = 0;
    } else if (index.length == heads.length) {
      for (var h = 0; h < heads.length; h++) {
        match[h] = h;
      }
    }
  } else if (opening && index.isNotEmpty) {
    final first = match.whereType<int>().firstOrNull ?? index.length;
    if (first > 0) openingEntry = 0;
  }

  return [
    if (opening)
      Chapter(
        title: openingEntry == null ? 'Intro' : index[openingEntry].$1,
        block: 0,
        summary: openingEntry == null ? null : index[openingEntry].$2,
      ),
    for (var h = 0; h < heads.length; h++)
      Chapter(
        title: chapterTitle(blocks[heads[h]].text),
        // Nothing to hear before the first heading but the title: start there.
        block: h == 0 && !opening ? 0 : heads[h],
        heading: heads[h],
        summary: match[h] == null ? null : index[match[h]!].$2,
      ),
  ];
}

const _numberWords = 'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|'
    'sixteen|seventeen|eighteen|nineteen|twenty';

/// "Part 1: Sunlight is a mixture" -> "Sunlight is a mixture" (a chapter list
/// numbers its entries already). "Chapter 3" alone stays as it is.
String chapterTitle(String heading) {
  final t = heading.trim();
  final m = RegExp(
    r'^(?:(?:part|chapter|section|episode|act|scene|segment|headline|item|lesson)\s+(?:\d{1,3}|[ivxlc]{1,6}|' +
        _numberWords +
        r')\b\s*[:.\-–—]\s*|\d{1,3}[.):]\s+|headline\s*:\s*)(.+)$',
    caseSensitive: false,
  ).firstMatch(t);
  return m == null ? t : m[1]!.trim();
}

String _titleKey(String t) => chapterTitle(t).toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), '');

bool _sameTitle(String a, String b) {
  final x = _titleKey(a), y = _titleKey(b);
  if (x.isEmpty || y.isEmpty) return false;
  if (x == y) return true;
  final (short, long) = x.length < y.length ? (x, y) : (y, x);
  return short.length >= 6 && long.contains(short);
}

final _speakerLine = RegExp(
  r'''^(?:\*\*|__)?\s*([A-Za-z][A-Za-z0-9 .'’\-]{0,30}?)\s*(?:\*\*|__)?\s*:\s*(?:\*\*|__)?(?!//)\s*(.*)$''',
);

/// Words that are often followed by a colon but are not people.
const _notSpeakers = {
  'NOTE', 'NOTES', 'WARNING', 'TIP', 'TIPS', 'EXAMPLE', 'EXAMPLES', 'SOURCE', 'SOURCES', 'SUMMARY', 'TITLE',
  'STYLE', 'MUSIC', 'VOICES', 'QUESTION', 'ANSWER', 'STEP', 'TODO', 'FACT', 'KEY POINT', 'DEFINITION', 'HTTP',
  'HTTPS', 'PS', 'P.S', 'RE', 'FYI', 'TL;DR', 'TLDR', 'CHAPTER', 'PART', 'SECTION', 'EPISODE', 'SCENE',
};

/// Roles that are clearly people even when they appear only once.
const _commonSpeakers = {
  'NARRATOR', 'HOST', 'CO-HOST', 'COHOST', 'GUEST', 'INTERVIEWER', 'PROFESSOR', 'TEACHER', 'STUDENT', 'ANCHOR',
  'REPORTER', 'SPEAKER 1', 'SPEAKER 2', 'SPEAKER 3', 'HOST 1', 'HOST 2', 'EXPERT', 'STORYTELLER',
};

String _canonical(String name) =>
    name.trim().replaceAll(RegExp(r'\s+'), ' ').replaceAll('’', "'").replaceAll(RegExp(r'\.$'), '').toUpperCase();

Map<String, int> _countSpeakerTags(List<String> lines, Set<String> declared) {
  final counts = <String, int>{};
  for (final raw in lines) {
    final m = _speakerLine.firstMatch(raw.trim());
    if (m == null) continue;
    final name = m[1]!.trim();
    final key = _canonical(name);
    final line = raw.trim();
    final bold = line.startsWith('**') || line.startsWith('__');
    // Undeclared speakers must be in capitals (HOST:, DR. LEE:) or bold (**Host:**).
    final caps = name == name.toUpperCase() && RegExp(r'[A-Z]{2}').hasMatch(name);
    if (!declared.contains(key) && !caps && !bold) continue;
    if (key.split(' ').length > 3) continue;
    // Well-known roles count double, so a single "NARRATOR:" line is enough.
    counts[key] = (counts[key] ?? 0) + (_commonSpeakers.contains(key) ? 2 : 1);
  }
  return counts;
}

bool _looksLikeChapter(String line) {
  if (line.length > 70) return false;
  final r = RegExp(
    r'^(chapter|part|book|section|prologue|epilogue|introduction|preface|foreword|afterword|appendix|interlude)\b'
    r'([\s.:\-–—]+[\w\s,.:;\-–—’\x27"!?]*)?$',
    caseSensitive: false,
  );
  if (!r.hasMatch(line)) return false;
  // "Part of the problem is..." is prose, not a heading.
  final words = line.split(RegExp(r'\s+'));
  if (words.length <= 2) return true;
  return RegExp(r'^\w+\s+([0-9IVXLC]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
              caseSensitive: false)
          .hasMatch(line) ||
      line == line.toUpperCase();
}

Block? _parseTag(String inner) {
  final t = inner.trim();
  final pause = RegExp(r'^(?:pause|silence|beat|break)(?:[\s:]+(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds?)?)?$',
          caseSensitive: false)
      .firstMatch(t);
  if (pause != null) {
    if (pause[1] == null) return const Block.pause(1.0);
    var seconds = double.parse(pause[1]!);
    if (pause[2]?.toLowerCase() == 'ms') seconds /= 1000;
    return Block.pause(seconds.clamp(0.1, 30).toDouble());
  }
  final cue = RegExp(r'^(music|sound|sfx)\s*[:=]?\s*([\w -]+)$', caseSensitive: false).firstMatch(t);
  if (cue != null) {
    final kind = cue[1]!.toLowerCase() == 'music' ? 'music' : 'sound';
    final name = cue[2]!.trim().toLowerCase().replaceAll(' ', '-');
    return Block.cue('$kind:$name');
  }
  // Anything else ([laughs], [Intro], [Cut to...]) is a direction, not speech.
  return null;
}

/// Adds [text] to the paragraph; an inline [pause] splits it with a real pause.
void _appendWithInlineTags(String text, List<String> paragraph, List<Block> blocks, void Function() flush) {
  final parts = text.split(RegExp(r'\[(?=(?:pause|beat|silence)\b)', caseSensitive: false));
  for (var i = 0; i < parts.length; i++) {
    var part = parts[i];
    if (i > 0) {
      final close = part.indexOf(']');
      if (close >= 0) {
        final tag = _parseTag(part.substring(0, close));
        flush();
        if (tag != null) blocks.add(tag);
        part = part.substring(close + 1);
      }
    }
    if (part.trim().isNotEmpty) paragraph.add(part.trim());
  }
}

void _addLine(List<Block> blocks, String text, String? speaker) => blocks.add(Block.line(text, speaker: speaker));

/// Drops pauses at the very start and merges back-to-back pauses.
List<Block> _tidy(List<Block> blocks) {
  final out = <Block>[];
  for (final b in blocks) {
    if (b.kind == BlockKind.pause) {
      if (out.isEmpty) continue;
      if (out.last.kind == BlockKind.pause) {
        out[out.length - 1] = Block.pause(out.last.seconds + b.seconds);
        continue;
      }
    }
    out.add(b);
  }
  while (out.isNotEmpty && out.last.kind == BlockKind.pause) {
    out.removeLast();
  }
  return out;
}

class _FrontMatter {
  _FrontMatter(this.body);

  String body;
  String? title;
  String? style;
  String? music;
  String? description;
  final Map<String, String> voices = {};
  final List<String> topics = [];

  /// `index:` entries: (title, summary).
  final List<(String, String?)> index = [];

  static _FrontMatter parse(String text) {
    final trimmed = text.trimLeft();
    final m = RegExp(r'^---[ \t]*\n([\s\S]*?)\n---[ \t]*(?:\n|$)').firstMatch(trimmed);
    if (m == null) return _FrontMatter(text);
    final fm = _FrontMatter(trimmed.substring(m.end));
    String? section;
    for (final raw in m[1]!.split('\n')) {
      final noComment = raw.replaceFirst(RegExp(r'\s+#.*$'), '');
      if (noComment.trim().isEmpty) continue;
      final indented = RegExp(r'^\s+').hasMatch(noComment);
      // Lists may also sit flush under their key ("index:" then "- Intro | ...").
      final listItem = RegExp(r'^\s*(?:[-*•]|\d{1,3}[.)])\s+').hasMatch(raw);
      if ((section == 'index' || section == 'topics') && (indented || listItem)) {
        if (section == 'index') {
          fm._indexEntry(raw.trim());
        } else {
          fm._addTopics(noComment);
        }
        continue;
      }
      final kv = RegExp(r'^\s*-?\s*([^:]+?)\s*:\s*(.*)$').firstMatch(noComment);
      if (kv == null) continue;
      final key = kv[1]!.trim();
      final value = _unquote(kv[2]!.trim());
      if (indented && section != null) {
        if (section == 'voices' && value.isNotEmpty) fm.voices[key] = value;
        continue;
      }
      section = null;
      switch (key.toLowerCase()) {
        case 'title':
          fm.title = value;
        case 'style':
          fm.style = value.toLowerCase();
        case 'music':
          fm.music = value.toLowerCase();
        case 'description' || 'summary' || 'about':
          fm.description = value;
        case 'topics' || 'topic' || 'tags' || 'subjects':
          section = 'topics';
          fm._addTopics(value);
        case 'index' || 'chapters' || 'contents' || 'outline':
          section = 'index';
        case 'voices' || 'cast' || 'speakers':
          section = 'voices';
          // Also allow "voices: HOST=Radio Host, GUEST=The Professor".
          for (final pair in value.split(',')) {
            final p = pair.split(RegExp(r'[=:]'));
            if (p.length == 2 && p[0].trim().isNotEmpty) fm.voices[p[0].trim()] = _unquote(p[1].trim());
          }
      }
    }
    return fm;
  }

  /// "Sunlight is a mixture | White light holds every colour." (also the
  /// YAML form: "- title: ..." followed by "summary: ...").
  void _indexEntry(String line) {
    var t = line.replaceFirst(RegExp(r'^(?:[-*•]|\d{1,3}[.)])\s+'), '').trim();
    t = t.replaceFirst(RegExp(r'^\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?\s+'), ''); // a timestamp Claude guessed
    final yaml = RegExp(r'^(title|name|summary|about|description)\s*:\s*(.*)$', caseSensitive: false).firstMatch(t);
    if (yaml != null) {
      final value = _clean(yaml[2]!);
      if (value.isEmpty) return;
      if (const {'title', 'name'}.contains(yaml[1]!.toLowerCase())) {
        index.add((value, null));
      } else if (index.isNotEmpty) {
        index[index.length - 1] = (index.last.$1, value);
      }
      return;
    }
    var cut = t.indexOf('|');
    var width = 1;
    if (cut < 0) {
      final dash = RegExp(r'\s+(?:—|–|--|-)\s+').firstMatch(t);
      final colons = RegExp(r':\s+').allMatches(t).toList();
      if (dash != null) {
        cut = dash.start;
        width = dash.end - dash.start;
      } else if (colons.length == 1 && colons.first.start <= 60) {
        cut = colons.first.start;
        width = colons.first.end - colons.first.start;
      }
    }
    final name = _clean(cut < 0 ? t : t.substring(0, cut));
    final summary = cut < 0 ? '' : _clean(t.substring(cut + width));
    if (name.isEmpty) return;
    index.add((name, summary.isEmpty ? null : summary));
  }

  void _addTopics(String value) {
    var v = value.trim();
    if (v.startsWith('[') && v.endsWith(']')) v = v.substring(1, v.length - 1);
    v = v.replaceFirst(RegExp(r'^(?:[-*•]|\d{1,3}[.)])\s+'), '');
    for (final part in v.split(RegExp(r'[,;]'))) {
      final topic = _clean(part).replaceAll(RegExp(r'^#'), '');
      if (topic.isEmpty || topic.length > 32) continue;
      if (topics.any((t) => t.toLowerCase() == topic.toLowerCase())) continue;
      if (topics.length < 8) topics.add(topic);
    }
  }

  static String _clean(String v) => displayText(_unquote(v.trim()));

  static String _unquote(String v) {
    if (v.length >= 2 && ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'")))) {
      return v.substring(1, v.length - 1);
    }
    return v;
  }
}
