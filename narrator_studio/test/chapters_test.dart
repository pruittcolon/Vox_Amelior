import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/audio/exporter.dart';
import 'package:narrator_studio/core/audio/mp4_chapters.dart';
import 'package:narrator_studio/core/render/plan.dart';
import 'package:narrator_studio/core/render/renderer.dart';
import 'package:narrator_studio/core/render/styles.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/script/script_parser.dart';

const _episode = '''---
title: Black Holes
style: podcast
summary: What a black hole is, and why nothing escapes it.
topics: [Space, physics, space]
voices:
  HOST: Radio Host
index:
  - Cold open | A star collapses.
  - What is a black hole | Gravity so strong that light can't escape.
  - The event horizon — The point of no return.
  - title: Hawking radiation
    summary: Black holes slowly evaporate.
---
HOST: Imagine a star collapsing.
HOST: It keeps going.

## Part 1: What is a black hole?
HOST: Gravity wins.

## The event horizon
HOST: No way back.

## Hawking radiation
HOST: Even black holes fade.
''';

void main() {
  group('index', () {
    test('summary, topics and chapters with summaries', () {
      final s = parseScript(_episode);
      expect(s.description, 'What a black hole is, and why nothing escapes it.');
      expect(s.topics, ['Space', 'physics']);
      expect(s.chapters.map((c) => c.title), ['Cold open', 'What is a black hole?', 'The event horizon', 'Hawking radiation']);
      expect(s.chapters.map((c) => c.summary), [
        'A star collapses.',
        "Gravity so strong that light can't escape.",
        'The point of no return.',
        'Black holes slowly evaporate.',
      ]);
      expect(s.chapters.first.block, 0);
      expect(s.chapters.first.heading, isNull);
      expect(s.voices, {'HOST': 'Radio Host'}, reason: 'the index does not leak into the cast');
    });

    test('entries pair by position when the titles differ', () {
      final s = parseScript('''---
index:
  - Intro | Hello.
  - Alpha | First.
  - Beta | Second.
---
Opening words.

## One
Text.

## Two
More.
''');
      expect(s.chapters.map((c) => '${c.title}: ${c.summary}'), ['Intro: Hello.', 'One: First.', 'Two: Second.']);
    });

    test('without an index, headings are the chapters and the opening is "Intro"', () {
      final s = parseScript('Before.\n\n## A\nText.\n\n## B\nMore.');
      expect(s.chapters.map((c) => c.title), ['Intro', 'A', 'B']);
      expect(s.chapters.every((c) => c.summary == null), isTrue);
    });

    test('a book: the title heading is not a chapter, and chapter one starts at the top', () {
      final s = parseScript(sampleAudiobook);
      expect(s.title, 'The Library That Remembered Everything');
      expect(s.chapters.map((c) => c.title), ['A City of Books', 'What Was Lost']);
      expect(s.chapters.first.block, 0);
      expect(s.chapters.first.summary, startsWith('How Alexandria'));
    });

    test('plain text has no chapters', () {
      expect(parseScript('Just words. And more words.').chapters, isEmpty);
    });

    test('chapter titles lose their numbering', () {
      expect(chapterTitle('Part 1: Sunlight is a mixture'), 'Sunlight is a mixture');
      expect(chapterTitle('Chapter Two: What Was Lost'), 'What Was Lost');
      expect(chapterTitle('3. The road'), 'The road');
      expect(chapterTitle('Chapter 3'), 'Chapter 3');
      expect(chapterTitle('2001: A Space Odyssey'), '2001: A Space Odyssey');
    });

    test('the sample podcast has its index', () {
      final s = parseScript(samplePodcast);
      expect(s.chapters.map((c) => c.title), ['Welcome', 'Sunlight is a mixture', 'The scattering trick', 'Sunsets']);
      expect(s.chapters.every((c) => c.summary != null), isTrue);
      expect(s.topics, contains('physics'));
    });

    test('old copies of the samples are upgraded, edited ones are not', () {
      String strip(String t) {
        final x = t
            .replaceAll(RegExp(r'^(?:summary|topics):.*\n', multiLine: true), '')
            .replaceAll(RegExp(r'^index:\n(?:  .*\n)*', multiLine: true), '');
        return x.startsWith('---\n---\n') ? x.substring(8) : x;
      }

      expect(upgradedSample(strip(samplePodcast)), samplePodcast);
      expect(upgradedSample(strip(sampleAudiobook)), sampleAudiobook);
      expect(strip(sampleAudiobook), startsWith('# The Library'));
      expect(upgradedSample('${strip(samplePodcast)}\nHOST: my own line.'), isNull);
      expect(upgradedSample(samplePodcast), isNull);
    });
  });

  group('plan chapters', () {
    test('each chapter knows its utterances; times run forward', () {
      final plan = buildPlan(parseScript(_episode), styleById('podcast'));
      expect(plan.hasChapters, isTrue);
      expect(plan.chapters.map((c) => (c.first, c.last)), [(0, 1), (2, 2), (3, 3), (4, 4)]);
      expect(plan.chapterOf(3)!.title, 'The event horizon');
      final t = plan.times(1.0);
      expect(t.starts.first, greaterThanOrEqualTo(introSeconds * 1000), reason: 'after the intro');
      for (var i = 1; i < t.starts.length; i++) {
        expect(t.starts[i], greaterThan(t.starts[i - 1]));
      }
      expect(t.utteranceAt(t.starts[3] + 10), 3);
      expect(t.utteranceAt(0), 0);
      expect(t.utteranceAt(t.total + 1000), plan.utterances.length - 1);
      expect(plan.estimate(1.0).inMilliseconds, t.total);
    });

    test('read-aloud headings start their chapter', () {
      final plan = buildPlan(parseScript(sampleAudiobook), styleById('audiobook'));
      expect(plan.chapters.first.first, 0);
      expect(plan.utterances[plan.chapters[1].first].heading, isTrue);
    });
  });

  group('chapter list', () {
    test('YouTube description format', () {
      final list = chapterList(
        const [
          ChapterMark(Duration.zero, 'Intro'),
          ChapterMark(Duration(seconds: 65), 'Light'),
          ChapterMark(Duration(minutes: 10, seconds: 3), 'Sunsets'),
        ],
        title: 'Sky',
        summary: 'Why it is blue.',
      );
      expect(list, 'Sky\nWhy it is blue.\n\n0:00 Intro\n1:05 Light\n10:03 Sunsets');
      expect(chapterList(const [ChapterMark(Duration.zero, 'A'), ChapterMark(Duration(hours: 1, seconds: 2), 'B')]),
          '0:00:00 A\n1:00:02 B');
    });
  });

  group('mp4 chapters', () {
    late Directory tmp;
    setUp(() => tmp = Directory.systemTemp.createTempSync('chapters'));
    tearDown(() => tmp.deleteSync(recursive: true));

    const marks = [
      ChapterMark(Duration.zero, 'Welcome'),
      ChapterMark(Duration(seconds: 20), 'Sunlight is a mixture'),
      ChapterMark(Duration(seconds: 41, milliseconds: 500), 'Sunsets — and why ✓'),
    ];

    for (final moovFirst in [false, true]) {
      test('adds QuickTime and Nero chapters, audio offsets follow (moov ${moovFirst ? 'first' : 'last'})', () async {
        final (bytes, chunks) = _fakeM4a(moovFirst: moovFirst);
        final f = File('${tmp.path}/a.m4a')..writeAsBytesSync(bytes);
        expect(await writeMp4Chapters(f.path, marks), isTrue);
        final out = f.readAsBytesSync();
        // Every audio chunk offset still points at its own bytes.
        final offsets = _stco(out).first;
        expect(offsets, hasLength(chunks.length));
        for (var i = 0; i < chunks.length; i++) {
          expect(ascii.decode(out.sublist(offsets[i], offsets[i] + chunks[i].length)), chunks[i]);
        }
        final (qt, nero) = await readMp4Chapters(f.path);
        for (final got in [qt, nero]) {
          expect(got.map((c) => c.title), marks.map((c) => c.title));
          expect(got.map((c) => c.start.inMilliseconds), [0, 20000, 41500]);
        }
        expect(await writeMp4Chapters(f.path, marks), isFalse, reason: 'already has chapters');
        expect(File('${f.path}.chapters.tmp').existsSync(), isFalse);
      });
    }

    test('leaves anything else alone', () async {
      final wav = File('${tmp.path}/x.m4a')..writeAsBytesSync(List.filled(64, 7));
      expect(await writeMp4Chapters(wav.path, marks), isFalse);
      expect(wav.readAsBytesSync(), List.filled(64, 7));
      final (bytes, _) = _fakeM4a(moovFirst: false);
      final one = File('${tmp.path}/one.m4a')..writeAsBytesSync(bytes);
      expect(await writeMp4Chapters(one.path, marks.take(1).toList()), isFalse);
      expect(await writeMp4Chapters(one.path, const [ChapterMark(Duration.zero, 'A'), ChapterMark(Duration(minutes: 5), 'Past the end')]),
          isFalse);
      expect(one.readAsBytesSync(), bytes);
    });
  });
}

Uint8List _u32(int v) => Uint8List(4)..buffer.asByteData().setUint32(0, v);

Uint8List _box(String type, List<List<int>> parts) {
  final b = BytesBuilder()
    ..add(_u32(8 + parts.fold<int>(0, (a, p) => a + p.length)))
    ..add(ascii.encode(type));
  for (final p in parts) {
    b.add(p);
  }
  return b.takeBytes();
}

Uint8List _full(String type, List<List<int>> parts) => _box(type, [_u32(0), ...parts]);

/// A minimal M4A: one audio track, 60 s, two chunks of marker bytes.
(Uint8List, List<String>) _fakeM4a({required bool moovFirst}) {
  const chunks = ['AUDIO-CHUNK-ONE', 'AUDIO-CHUNK-TWO!'];
  final ftyp = _box('ftyp', [ascii.encode('M4A '), _u32(0), ascii.encode('M4A isom')]);
  final mdat = _box('mdat', [ascii.encode(chunks.join())]);
  Uint8List moov(List<int> offsets) {
    final mvhd = _full('mvhd', [_u32(0), _u32(0), _u32(1000), _u32(60000), _u32(0x00010000), Uint8List(2 + 10 + 36 + 24), _u32(2)]);
    final tkhd = _full('tkhd', [_u32(0), _u32(0), _u32(1), Uint8List(84 - 16)]);
    final hdlr = _full('hdlr', [_u32(0), ascii.encode('soun'), Uint8List(12), ascii.encode('Sound\x00')]);
    final stco = _full('stco', [_u32(offsets.length), for (final o in offsets) _u32(o)]);
    final stbl = _box('stbl', [_full('stsz', [_u32(0), _u32(0)]), stco]);
    return _box('moov', [
      mvhd,
      _box('trak', [tkhd, _box('mdia', [_full('mdhd', [_u32(0), _u32(0), _u32(48000), _u32(48000 * 60), _u32(0)]), hdlr, _box('minf', [stbl])])]),
    ]);
  }

  final size = moov(const [0, 0]).length;
  final dataAt = ftyp.length + (moovFirst ? size : 0) + 8;
  final m = moov([dataAt, dataAt + chunks[0].length]);
  final bytes = BytesBuilder()..add(ftyp);
  if (moovFirst) {
    bytes
      ..add(m)
      ..add(mdat);
  } else {
    bytes
      ..add(mdat)
      ..add(m);
  }
  return (bytes.takeBytes(), chunks);
}

/// The chunk offsets of every stco box, in file order.
List<List<int>> _stco(Uint8List b) {
  final d = ByteData.sublistView(b);
  final out = <List<int>>[];
  for (var i = 4; i + 4 <= b.length; i++) {
    if (b[i] == 0x73 && b[i + 1] == 0x74 && b[i + 2] == 0x63 && b[i + 3] == 0x6F) {
      final n = d.getUint32(i + 8);
      out.add([for (var k = 0; k < n; k++) d.getUint32(i + 12 + 4 * k)]);
    }
  }
  return out;
}
