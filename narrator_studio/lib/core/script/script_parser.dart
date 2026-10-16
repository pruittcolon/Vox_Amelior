/// Reads plain text, Markdown or a Narrator script into a [Script].
///
/// The script format (see GUIDE.md) is plain text with a few extras:
///
///     ---
///     title: Why Is the Sky Blue?
///     style: podcast
///     music: curious
///     voices:
///       HOST: Radio Host
///       GUEST: The Professor
///     ---
///     [music: intro]
///     HOST: Welcome back to Curious Minds!
///     ## Part 1: Light is a wave
///     [pause 2]
///     [sound: chime]
///
/// Anything else (a book, an article, notes) just reads as narration.
library;

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

  return Script(
    title: (title == null || title.isEmpty) ? fallbackTitle : title,
    blocks: _tidy(blocks),
    style: front.style,
    music: front.music,
    voices: {for (final e in front.voices.entries) e.key.toUpperCase(): e.value},
    description: front.description,
  );
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
        case 'description' || 'summary':
          fm.description = value;
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

  static String _unquote(String v) {
    if (v.length >= 2 && ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'")))) {
      return v.substring(1, v.length - 1);
    }
    return v;
  }
}
