import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../core/render/music.dart';
import '../core/render/styles.dart';
import '../core/script/prompts.dart';
import 'scope.dart';

/// The script format, for you (and for Claude).
class GuideScreen extends StatelessWidget {
  const GuideScreen({super.key});

  static const _example = '''---
title: Why Is the Sky Blue?
style: podcast
music: curious
voices:
  HOST: Radio Host
  GUEST: The Professor
---

HOST: Welcome back to Curious Minds!
GUEST: Thanks for having me.

## Part 1: Sunlight is a mixture
HOST: So where does the blue come from?
[pause 1]
GUEST: It's already in the sunlight...

[sound: chime]
[music: outro]
HOST: That's all for today.''';

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final mono = const TextStyle(fontFamily: 'monospace', fontSize: 13, height: 1.45);
    Widget h(String s) => Padding(padding: const EdgeInsets.only(top: 20, bottom: 6), child: Text(s, style: text.titleMedium));
    Widget row(String code, String what) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(width: 150, child: Text(code, style: mono.copyWith(fontWeight: FontWeight.w600))),
              Expanded(child: Text(what)),
            ],
          ),
        );
    return Scaffold(
      appBar: AppBar(title: const Text('Script format')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
        children: [
          const Text(
            'Any text works: a plain book or article simply reads as narration. Scripts add speakers, pauses and '
            'music with a few simple marks. Claude knows them when you use the prompts on the Write tab.',
          ),
          h('Example'),
          Card(child: Padding(padding: const EdgeInsets.all(14), child: SelectableText(_example, style: mono))),
          h('Marks'),
          row('---  ...  ---', 'Optional header: title, style, music and which character voices each speaker.'),
          row('HOST: text', 'A line spoken by HOST. Tags are capitals (or **Bold:**). Untagged text is the narrator.'),
          row('## Title', 'A section. Read aloud in audiobook, lecture, story and briefing styles; a transition sound in podcasts.'),
          row('[pause 2]', 'Silence, in seconds (also [pause 500ms]).'),
          row('[sound: chime]', 'A cue: chime, ding, whoosh or pop.'),
          row('[music: intro]', 'Music: intro, outro or stop. With music on, intro and outro happen automatically.'),
          row('[laughs]', 'Any other bracket is a stage direction and is skipped.'),
          row('<!-- note -->', 'A note to yourself; never read.'),
          h('Styles'),
          for (final s in readingStyles) row(s.id, '${s.name}: ${s.tagline}'),
          h('Music themes'),
          for (final t in musicThemes) row(t.id, '${t.name}: ${t.mood}'),
          h('Writing tips'),
          const Text(
            '• Write for the ear: short sentences, contractions, "here\'s the surprising part".\n'
            '• Spell out how numbers and symbols are said: "nineteen oh five", "about three hundred thousand kilometres".\n'
            '• Avoid tables, bullet lists, URLs and footnotes; they don\'t survive being read aloud.\n'
            '• 150 words is about one minute of listening.\n'
            '• For a famous figure, write a character "inspired by" them and say so in the intro.',
          ),
          const SizedBox(height: 20),
          OutlinedButton.icon(
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: scriptFormatRules));
              if (context.mounted) context.toast('Format rules copied: add them to any request to Claude.');
            },
            icon: const Icon(Icons.copy),
            label: const Text('Copy the format rules for Claude'),
          ),
        ],
      ),
    );
  }
}
