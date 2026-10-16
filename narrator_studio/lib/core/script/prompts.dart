/// Ready-to-paste prompts that make Claude write scripts in the app's format.
library;

import '../render/music.dart';
import '../tts/voices.dart';

class ScriptPrompt {
  const ScriptPrompt({
    required this.style,
    required this.title,
    required this.when,
    required this.body,
    this.minutes = 10,
  });

  /// Style id (see render/styles.dart).
  final String style;
  final String title;

  /// One line: when to use it.
  final String when;

  /// The request; [TOPIC] and [MINUTES] are filled in by [fill].
  final String body;
  final int minutes;

  String fill({String topic = '', int? minutes}) {
    final m = minutes ?? this.minutes;
    final t = topic.trim().isEmpty ? '[YOUR TOPIC — or "the attached document"]' : topic.trim();
    return '${body.replaceAll('[TOPIC]', t).replaceAll('[MINUTES]', '$m').replaceAll('[WORDS]', '${m * 150}')}'
        '\n\n$scriptFormatRules';
  }
}

String get _characterList => builtInCharacters.map((c) => '${c.name} (${c.blurb.split('.').first.toLowerCase()})').join('; ');

String get _musicList => [...musicThemes.map((t) => '${t.id} (${t.mood.split('.').first.toLowerCase()})'), 'off'].join(', ');

/// The format, explained for Claude. Appended to every prompt.
String get scriptFormatRules => '''
FORMAT — this will be read aloud by text-to-speech voices in the Narrator Studio app, so follow it exactly:

1. Start with this header (YAML front matter):
---
title: <a short, catchy title>
style: <audiobook | podcast | lecture | interview | story | news>
music: <one of: $_musicList>
voices:
  <SPEAKER TAG>: <character name from the list below>
---

2. Every spoken paragraph starts with its speaker tag in CAPITALS and a colon, like "HOST: Welcome back!". Use the same tag every time. A paragraph without a tag is read by the narrator.
3. Split the script into parts with lines like "## Part 1: Why the sky is blue". In podcast and interview styles these are not read aloud (a transition sound plays); in the other styles they are announced.
4. Optional cues on their own line: [pause 2] for a 2-second pause, [sound: chime] or [sound: whoosh] for a transition, [music: intro] / [music: outro] / [music: stop] to control the music.
5. Write for the ear, not the eye: short sentences, natural spoken rhythm, contractions, signposting ("Here's the surprising part..."). Spell out symbols and tricky numbers the way you'd say them ("about three hundred thousand kilometres per second", "nineteen oh five"). No tables, bullet lists, URLs, citations, footnotes, emojis, or markdown bold/italics. Any stage direction goes in [brackets] and will be skipped.
6. Output ONLY the script, nothing before or after it, so I can paste it straight into the app.

Characters you can assign in "voices:" — $_characterList.
''';

const scriptPrompts = <ScriptPrompt>[
  ScriptPrompt(
    style: 'podcast',
    title: 'Two-host podcast',
    when: 'An entertaining explainer with banter. The most fun format.',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) educational podcast episode about [TOPIC].

Hosts: HOST (Radio Host — curious, upbeat, keeps it moving) and CO-HOST (Co-Host — sharp, funny, asks what listeners are thinking). If an expert would help, add GUEST (The Professor).
Make it genuinely entertaining AND accurate:
- Hook the listener in the first 20 seconds with a surprising question or fact.
- Real back-and-forth: short turns, reactions ("Wait, really?"), friendly disagreement, vivid analogies and one memorable story.
- 3 to 5 parts, each with a "## " title and one "aha" moment.
- End with a quick recap of the key ideas and a teaser for a follow-up topic.
Use style: podcast, music: curious.''',
  ),
  ScriptPrompt(
    style: 'interview',
    title: 'Interview with a great mind',
    when: 'A host interviews a character inspired by a famous thinker (a clearly labelled dramatization).',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) interview-style episode about [TOPIC].

INTERVIEWER (Radio Host) talks with GUEST, a character inspired by a famous thinker connected to the topic (for example, a character inspired by Albert Einstein for relativity). Rules:
- The interviewer opens by saying this is an imagined conversation — a dramatization, not the real person — and names who inspired the character.
- The guest explains ideas in first person, in the spirit of that thinker's actual writings, era and personality; keep the facts historically accurate and have the interviewer flag anything speculative.
- Warm, curious questions; the guest uses analogies and thought experiments; include one lighter, human moment.
- 3 to 4 parts with "## " titles. End with the guest's advice to listeners and the interviewer's sign-off.
Use style: interview, music: fireside, voices: INTERVIEWER: Radio Host, GUEST: The Old Sage (or The Professor).''',
  ),
  ScriptPrompt(
    style: 'lecture',
    title: 'Professor\'s lecture',
    when: 'A clear, structured lesson with room to think. Great for studying.',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) spoken lecture about [TOPIC].

PROFESSOR (The Professor) teaches; STUDENT (The Student) interrupts two or three times with a good question.
- Open with why this matters, then a roadmap of the parts.
- Build from intuition to detail; one concrete example per idea; a quick "pause and think" question now and then followed by [pause 2].
- "## " titles for each part (they are announced). Close with a 30-second summary of the key takeaways.
Use style: lecture, music: off.''',
  ),
  ScriptPrompt(
    style: 'audiobook',
    title: 'Audiobook chapter',
    when: 'One narrator, like a well-read book. Best for rewriting a PDF or article for listening.',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) narrated chapter about [TOPIC], like a beautifully written popular-science audiobook.

If I attached a document, rewrite it for listening: keep all the ideas and facts, but turn figures, tables, lists and references into flowing spoken prose, and drop page numbers, captions and citations.
Single narrator (no speaker tags needed). Use "## " for chapter and section titles (they are read aloud). Vary sentence length; let important ideas land.
Use style: audiobook, music: off, voices: NARRATOR: The Narrator (or The Storyteller).''',
  ),
  ScriptPrompt(
    style: 'story',
    title: 'Story that teaches',
    when: 'A short story where characters discover the idea. Great for kids or bedtime.',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) story that teaches [TOPIC] through an adventure.

NARRATOR (The Storyteller) narrates; give each character a short CAPITAL tag (for example MAYA, GRANDPA, ROBOT) for their dialogue lines and assign each a fitting character voice in the header.
- Vivid setting, a small mystery or problem that the idea solves, and a satisfying ending.
- The science or history must be accurate; explain it through what the characters see and do.
- Use [pause 1] before reveals. "## " titles for chapters.
Use style: story, music: cosmic.''',
  ),
  ScriptPrompt(
    style: 'news',
    title: 'Daily briefing',
    when: 'Fast, crisp rundown: "the 5 things to know about...".',
    body: '''Write a [MINUTES]-minute (about [WORDS] words) news-style briefing: "The five things to know about [TOPIC]".

ANCHOR (News Anchor) presents; REPORTER (Field Reporter) adds colour for two of the items.
- A punchy cold open, then each item under its own "## " headline (read aloud), 2 to 4 tight paragraphs each.
- Plain, confident, factual; end with "the one thing to remember".
Use style: news, music: newsroom.''',
    minutes: 6,
  ),
];

/// Short how-to shown in the app.
const howItWorks = [
  ('Pick a style', 'Choose a format below and tap Copy prompt. Type your topic first to fill it in.'),
  ('Ask Claude', 'Open Claude, paste the prompt. Attach a PDF or article if you want it turned into the episode.'),
  ('Bring it back', 'Copy Claude\'s whole reply, come back here and tap Paste script. Or share it to Narrator Studio.'),
  ('Listen or export', 'Check the cast, press play, then Export to save an M4A you can upload as a podcast.'),
];
