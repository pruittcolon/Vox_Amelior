import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/script/pdf_cleanup.dart';
import 'package:narrator_studio/core/script/script.dart';
import 'package:narrator_studio/core/script/script_parser.dart';
import 'package:narrator_studio/core/script/sentences.dart';
import 'package:narrator_studio/core/script/text_cleaner.dart';

List<String> sentences(String t) => [for (final r in splitSentences(t)) t.substring(r.start, r.end)];

void main() {
  group('parseScript', () {
    test('front matter, speakers, headings, pauses and cues', () {
      final s = parseScript('''---
title: Why Is the Sky Blue?
style: podcast
music: curious  # the bright one
voices:
  HOST: Radio Host
  GUEST: "The Professor"
---
[music: intro]
HOST: Welcome back!
GUEST: Thanks for having me.

## Part 1: Light
HOST: So, light.
[pause 2]
[sound: chime]
[laughs]
GUEST: Indeed.
''');
      expect(s.title, 'Why Is the Sky Blue?');
      expect(s.style, 'podcast');
      expect(s.music, 'curious');
      expect(s.voices, {'HOST': 'Radio Host', 'GUEST': 'The Professor'});
      expect(s.speakers, ['HOST', 'GUEST']);
      expect(s.blocks.map((b) => b.toString()).toList(), [
        'C(music:intro)',
        'L[HOST](Welcome back!)',
        'L[GUEST](Thanks for having me.)',
        'H2(Part 1: Light)',
        'L[HOST](So, light.)',
        'P(2.0)',
        'C(sound:chime)',
        'L[GUEST](Indeed.)',
      ]);
    });

    test('plain prose is narration; lines join into paragraphs', () {
      final s = parseScript('It was a dark\nand stormy night.\n\nThe end.', fallbackTitle: 'Story');
      expect(s.title, 'Story');
      expect(s.speakers, isEmpty);
      expect(s.blocks.map((b) => b.text), ['It was a dark and stormy night.', 'The end.']);
    });

    test('"Note:" and "Time:" in prose are not speakers', () {
      final s = parseScript('Note: this is important.\nTime: 10am.\nNote: again.');
      expect(s.speakers, isEmpty);
    });

    test('bold speaker tags from Claude are understood', () {
      final s = parseScript('**Host:** Hi there.\n\n**Guest:** Hello!\n\n**Host:** Great.');
      expect(s.speakers, ['HOST', 'GUEST']);
      expect(s.blocks.first.text, 'Hi there.');
    });

    test('a single NARRATOR line is still a speaker', () {
      final s = parseScript('NARRATOR: Once upon a time.');
      expect(s.speakers, ['NARRATOR']);
      expect(s.blocks.single.text, 'Once upon a time.');
    });

    test('untagged paragraph in a dialogue continues the last speaker', () {
      final s = parseScript('HOST: One.\n\nGUEST: Two.\n\nAnd three.\n\n## Next\n\nFour.');
      expect(s.blocks.map((b) => '${b.speaker}:${b.text}'), ['HOST:One.', 'GUEST:Two.', 'GUEST:And three.', 'null:Next', 'null:Four.']);
    });

    test('markdown is removed from what is shown', () {
      final s = parseScript('# My **Title**\n\nSee [the docs](https://x.y) and *this* and `code`.[1]');
      expect(s.title, 'My Title');
      expect(s.blocks.last.text, 'See the docs and this and code.');
    });

    test('chapter lines in books become headings, prose starting with Part does not', () {
      final s = parseScript('CHAPTER ONE\n\nIt began.\n\nChapter 2: The Road\n\nPart of the problem was rain.');
      expect(s.blocks.map((b) => b.kind).toList(), [BlockKind.heading, BlockKind.line, BlockKind.heading, BlockKind.line]);
    });

    test('inline [pause] splits a line', () {
      final s = parseScript('HOST: Ready? [pause 1.5] Go!\nHOST: Done.');
      expect(s.blocks.map((b) => b.toString()).toList(), ['L[HOST](Ready?)', 'P(1.5)', 'L[HOST](Go!)', 'L[HOST](Done.)']);
    });

    test('comments and scene breaks', () {
      final s = parseScript('Hello.\n<!-- secret\nnote -->\n***\nBye.');
      expect(s.blocks.map((b) => b.toString()).toList(), ['L(Hello.)', 'P(1.2)', 'L(Bye.)']);
    });

    test('the built-in samples parse', () {
      final pod = parseScript(samplePodcast);
      expect(pod.speakers, ['HOST', 'CO-HOST', 'GUEST']);
      expect(pod.wordCount, greaterThan(400));
      final book = parseScript(sampleAudiobook);
      expect(book.title, 'The Library That Remembered Everything');
      expect(book.speakers, isEmpty);
    });
  });

  group('splitSentences', () {
    test('basic', () {
      expect(sentences('One. Two! Three? Four'), ['One.', 'Two!', 'Three?', 'Four']);
    });

    test('abbreviations, initials and decimals', () {
      expect(
        sentences('Dr. Smith met J. R. R. Tolkien at 3.30 p.m. on Monday. It went well.'),
        ['Dr. Smith met J. R. R. Tolkien at 3.30 p.m. on Monday.', 'It went well.'],
      );
      expect(sentences('See e.g. Fig. 3 for details. Then stop.'), ['See e.g. Fig. 3 for details.', 'Then stop.']);
    });

    test('quotes stay with their sentence', () {
      expect(sentences('He said "Go." Then he left.'), ['He said "Go."', 'Then he left.']);
    });

    test('long sentences are cut at clause boundaries', () {
      final long = List.filled(12, 'this clause keeps going for quite a while,').join(' ');
      final parts = sentences('$long and ends here.');
      expect(parts.length, greaterThan(1));
      for (final part in parts) {
        expect(part.length, lessThanOrEqualTo(maxSentenceChars));
      }
      expect(parts.join(' '), '$long and ends here.');
    });

    test('ranges point into the original text', () {
      const t = '  First one.   Second.  ';
      final r = splitSentences(t);
      expect(t.substring(r[0].start, r[0].end), 'First one.');
      expect(t.substring(r[1].start, r[1].end), 'Second.');
    });
  });

  group('speechText', () {
    test('expands what voices stumble over', () {
      expect(speechText('e.g. 1905–1910, about 40% at 20°C'), 'for example 1905 to 1910, about 40 percent at 20 degrees Celsius');
      expect(speechText('Dr. Lee vs. Mr. Kim & co.'), 'Doctor Lee versus Mister Kim and co.');
      expect(speechText('See https://example.com/x — it is “great”…'), 'See, it is "great"...');
      expect(speechText('No. 5 costs 1,200 dollars'), 'number 5 costs 1200 dollars');
      expect(speechText('[laughs] Okay 😀'), 'Okay');
    });

    test('isSpeakable', () {
      expect(isSpeakable('...'), isFalse);
      expect(isSpeakable('Hi'), isTrue);
    });
  });

  group('cleanPdfPages', () {
    test('drops page numbers and running headers, joins lines, fixes hyphens', () {
      const words = ['alpha', 'beta', 'gamma', 'delta'];
      final pages = [
        for (var i = 1; i <= 4; i++)
          'The Origin of Species\n'
              'This is the first line of a paragraph about ${words[i - 1]} that continues\n'
              'onto the next line about ${words[i - 1]} and photo-\n'
              'synthesis in ${words[i - 1]} plants.\n'
              'A short ${words[i - 1]} line.\n'
              '\n'
              'Another: ${words[i - 1]}.\n'
              '$i',
      ];
      final text = cleanPdfPages(pages);
      expect(text, isNot(contains('Origin of Species')));
      expect(text, contains('continues onto the next line'));
      expect(text, contains('photosynthesis'));
      expect(RegExp(r'^\d+$', multiLine: true).hasMatch(text), isFalse);
      expect(text.split('\n\n').length, greaterThan(3));
    });

    test('ligatures are expanded', () {
      expect(cleanPdfPages(['The ﬁrst ﬂoor.']), 'The first floor.');
    });
  });
}
