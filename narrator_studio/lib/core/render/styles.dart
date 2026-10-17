/// Reading styles: how a document is paced, voiced and dressed up.
library;

class ReadingStyle {
  const ReadingStyle({
    required this.id,
    required this.name,
    required this.tagline,
    required this.icon,
    required this.narrator,
    required this.speed,
    required this.sentenceGapMs,
    required this.paragraphGapMs,
    required this.speakerGapMs,
    required this.headingBeforeMs,
    required this.headingAfterMs,
    required this.readHeadings,
    required this.music,
    this.defaultCast = const {},
    this.sectionSound,
  });

  final String id;
  final String name;
  final String tagline;

  /// Material icon code point name (see ui/icons.dart).
  final String icon;

  /// Character used for narration / untagged text.
  final String narrator;
  final double speed;
  final int sentenceGapMs;
  final int paragraphGapMs;
  final int speakerGapMs;
  final int headingBeforeMs;
  final int headingAfterMs;
  final bool readHeadings;

  /// Default music theme (`off` for none).
  final String music;

  /// Sound played at each `##` section when headings are not read aloud.
  final String? sectionSound;

  /// Common speaker tags -> character, used when the script doesn't say.
  final Map<String, String> defaultCast;
}

const readingStyles = <ReadingStyle>[
  ReadingStyle(
    id: 'audiobook',
    name: 'Audiobook',
    tagline: 'Read like a book: one narrator, chapter titles, unhurried pauses.',
    icon: 'menu_book',
    narrator: 'narrator',
    speed: 0.95,
    sentenceGapMs: 260,
    paragraphGapMs: 700,
    speakerGapMs: 450,
    headingBeforeMs: 900,
    headingAfterMs: 900,
    readHeadings: true,
    music: 'off',
    defaultCast: {'NARRATOR': 'narrator'},
  ),
  ReadingStyle(
    id: 'podcast',
    name: 'Podcast',
    tagline: 'Two hosts chatting: intro music, quick back-and-forth, outro.',
    icon: 'podcasts',
    narrator: 'host',
    speed: 1.0,
    sentenceGapMs: 180,
    paragraphGapMs: 420,
    speakerGapMs: 260,
    headingBeforeMs: 500,
    headingAfterMs: 400,
    readHeadings: false,
    music: 'curious',
    sectionSound: 'sweep',
    defaultCast: {
      'HOST': 'host',
      'HOST 1': 'host',
      'CO-HOST': 'cohost',
      'COHOST': 'cohost',
      'HOST 2': 'cohost',
      'GUEST': 'professor',
      'EXPERT': 'professor',
      'NARRATOR': 'narrator',
    },
  ),
  ReadingStyle(
    id: 'lecture',
    name: 'Lecture',
    tagline: 'A professor explains, slowly, with room to think after big ideas.',
    icon: 'school',
    narrator: 'professor',
    speed: 0.92,
    sentenceGapMs: 320,
    paragraphGapMs: 900,
    speakerGapMs: 500,
    headingBeforeMs: 1100,
    headingAfterMs: 800,
    readHeadings: true,
    music: 'off',
    defaultCast: {'PROFESSOR': 'professor', 'STUDENT': 'student', 'TEACHER': 'professor'},
  ),
  ReadingStyle(
    id: 'interview',
    name: 'Interview',
    tagline: 'An interviewer asks, an expert answers. Great for "conversations with" a character.',
    icon: 'mic',
    narrator: 'host',
    speed: 0.98,
    sentenceGapMs: 200,
    paragraphGapMs: 500,
    speakerGapMs: 380,
    headingBeforeMs: 600,
    headingAfterMs: 500,
    readHeadings: false,
    music: 'fireside',
    sectionSound: 'chime',
    defaultCast: {
      'INTERVIEWER': 'host',
      'HOST': 'host',
      'GUEST': 'sage',
      'EXPERT': 'professor',
      'NARRATOR': 'narrator',
    },
  ),
  ReadingStyle(
    id: 'story',
    name: 'Story',
    tagline: 'A storyteller with character voices for the dialogue. Bedtime-friendly.',
    icon: 'auto_stories',
    narrator: 'storyteller',
    speed: 0.92,
    sentenceGapMs: 300,
    paragraphGapMs: 800,
    speakerGapMs: 350,
    headingBeforeMs: 1000,
    headingAfterMs: 1000,
    readHeadings: true,
    music: 'cosmic',
    defaultCast: {'NARRATOR': 'storyteller'},
  ),
  ReadingStyle(
    id: 'news',
    name: 'Briefing',
    tagline: 'Crisp news-style rundown: headlines, a chime between stories.',
    icon: 'newspaper',
    narrator: 'anchor',
    speed: 1.06,
    sentenceGapMs: 160,
    paragraphGapMs: 380,
    speakerGapMs: 250,
    headingBeforeMs: 400,
    headingAfterMs: 350,
    readHeadings: true,
    music: 'newsroom',
    sectionSound: 'chime',
    defaultCast: {'ANCHOR': 'anchor', 'REPORTER': 'reporter', 'HOST': 'anchor'},
  ),
];

ReadingStyle styleById(String? id) => readingStyles.firstWhere(
      (s) => s.id == id,
      orElse: () => readingStyles.first,
    );

/// Picks a style for a script that didn't name one.
String guessStyle({required bool hasSpeakers, required int speakerCount, required bool isPdf}) {
  if (!hasSpeakers) return 'audiobook';
  return speakerCount >= 2 ? 'podcast' : 'audiobook';
}
