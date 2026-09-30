import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

/// Activity with [spans] of (speaker, fromSeconds, toSeconds) at [fs] seconds per frame.
SpeakerActivity activity(double seconds, List<(int, double, double)> spans, {double fs = 0.01, int speakers = 4}) {
  final frames = (seconds / fs).round();
  final p = Float32List(frames * speakers);
  for (final (s, from, to) in spans) {
    for (var t = (from / fs).round(); t < (to / fs).round() && t < frames; t++) {
      p[t * speakers + s] = 0.9;
    }
  }
  return SpeakerActivity(p, frames: frames, speakers: speakers, frameSeconds: fs);
}

void main() {
  const turns = SpeakerTurns();

  group('speaker turns', () {
    for (final fs in [0.01, 0.08]) {
      test('one voice is one turn (frames of ${fs * 1000} ms)', () {
        final t = turns.turns(activity(6, [(0, 0.2, 5.8)], fs: fs), totalSeconds: 6);
        expect(t, hasLength(1));
        expect(t.single.overlap, isFalse);
        expect(t.single.start, 0);
        expect(t.single.end, 6);
      });

      test('a change of speaker splits at the change ($fs)', () {
        final t = turns.turns(activity(6, [(0, 0, 3), (1, 3.1, 6)], fs: fs), totalSeconds: 6);
        expect(t.map((x) => x.slot), [0, 1]);
        expect(t[0].end, closeTo(3.1, fs + 1e-9));
        expect(t[1].end, 6);
        expect(t.any((x) => x.overlap), isFalse);
      });
    }

    test('short blips and short pauses do not create turns', () {
      final t = turns.turns(activity(6, [(0, 0, 2.5), (0, 2.6, 6), (1, 4, 4.1)]), totalSeconds: 6);
      expect(t, hasLength(1));
    });

    test('talking at the same time is marked; a quick hand-over is not', () {
      final real = turns.turns(activity(6, [(0, 0, 3.5), (1, 2.5, 6)]), totalSeconds: 6);
      expect(real.map((x) => x.slot), [0, 1]);
      expect(real.any((x) => x.overlap), isTrue);
      final handOver = turns.turns(activity(6, [(0, 0, 3.05), (1, 2.95, 6)]), totalSeconds: 6);
      expect(handOver.any((x) => x.overlap), isFalse);
    });

    test('a very short turn is merged into its neighbour', () {
      final t = turns.turns(activity(6, [(0, 0, 3), (1, 3, 3.5), (0, 3.5, 6)]), totalSeconds: 6);
      expect(t, hasLength(1));
      expect(t.single.slot, 0);
    });

    test('back-and-forth gives alternating turns covering the whole audio', () {
      final t = turns.turns(activity(10, [(0, 0, 2), (1, 2.2, 4.5), (0, 4.7, 7), (1, 7.2, 10)]), totalSeconds: 10);
      expect(t.map((x) => x.slot), [0, 1, 0, 1]);
      expect(t.first.start, 0);
      expect(t.last.end, 10);
      for (var i = 1; i < t.length; i++) {
        expect(t[i].start, t[i - 1].end);
      }
    });

    test('joinSame merges neighbours that are the same person', () {
      const list = [
        SpeakerTurn(start: 0, end: 2, slot: 0),
        SpeakerTurn(start: 2, end: 4, slot: 1, overlap: true),
        SpeakerTurn(start: 4, end: 6, slot: 2),
      ];
      final joined = SpeakerTurns.joinSame(list, (t) => t.slot == 2 ? 'alex' : (t.slot == 1 ? 'alex' : 'sam'));
      expect(joined, hasLength(2));
      expect(joined[1].start, 2);
      expect(joined[1].end, 6);
      expect(joined[1].overlap, isTrue);
    });

    test('solo ranges exclude overlap', () {
      final a = activity(4, [(0, 0, 3), (1, 2, 4)]);
      expect(SpeakerTurns.soloRanges(a, 0, sampleRate: 16000), [(0, 32000)]);
      expect(SpeakerTurns.soloRanges(a, 1, sampleRate: 16000), [(48000, 64000)]);
    });
  });

  group('splitting lines', () {
    late AppDatabase db;
    late SpeakerRepository speakers;
    late TranscriptRepository transcripts;
    late String alex;
    late String sam;
    setUp(() {
      db = AppDatabase.inMemory();
      speakers = SpeakerRepository(db);
      transcripts = TranscriptRepository(db);
      alex = enrollFake(speakers, 'Alex', 1).id;
      sam = enrollFake(speakers, 'Sam', 2).id;
    });
    tearDown(() => db.close());

    SegmentProcessor processor(List<String> texts, {DiarizationEngine? diarizer}) => SegmentProcessor(
          asr: FakeAsr(texts),
          embedder: FakeEmbedder(),
          identifier: SpeakerIdentifier(
            profiles: speakers.profiles(),
            newClusterId: SpeakerRepository.newId,
            nextGuestLabel: speakers.nextGuestLabel,
          ),
          transcripts: transcripts,
          speakers: speakers,
          diarizer: diarizer,
        );

    // Alex talks for 3 s, then Sam for 3 s, with no pause between.
    Float32List twoPeople() => Float32List.fromList([...fakeAudio(1), ...fakeAudio(2)]);
    final t0 = DateTime(2026, 9, 30, 20);

    test('quick back-and-forth becomes one line per person, with names and times', () {
      final p = processor(
        ['are you coming', 'yes in a minute'],
        diarizer: _FakeDiarizer(activity(6, [(0, 0, 3), (1, 3, 6)])),
      );
      final saved = p.process(twoPeople(), t0);
      expect(saved.map((s) => s.text), ['are you coming', 'yes in a minute']);
      expect(saved.map((s) => s.speakerId), [alex, sam]);
      expect(saved[1].startedAt, t0.add(const Duration(seconds: 3)));
      expect(saved[0].duration, const Duration(seconds: 3));
      expect(p.stats.split, 1);
      expect(transcripts.conversation(saved.first.conversationId), hasLength(2));
    });

    test('talking at the same time is marked and stored', () {
      final p = processor(
        ['wait', 'no listen'],
        diarizer: _FakeDiarizer(activity(6, [(0, 0, 3.6), (1, 2.6, 6)])),
      );
      final saved = p.process(twoPeople(), t0);
      expect(saved.any((s) => s.overlap), isTrue);
      expect(transcripts.segment(saved.firstWhere((s) => s.overlap).id)!.overlap, isTrue);
    });

    test('two slots that are the same person stay one line', () {
      final same = Float32List.fromList([...fakeAudio(1), ...fakeAudio(1)]);
      final p = processor(['one long line'], diarizer: _FakeDiarizer(activity(6, [(0, 0, 3), (1, 3, 6)])));
      final saved = p.process(same, t0);
      expect(saved, hasLength(1));
      expect(saved.single.speakerId, alex);
    });

    test('without the model, switched off, or if it fails, the line stays whole', () {
      expect(processor(['whole']).process(twoPeople(), t0), hasLength(1));
      final off = processor(['whole'], diarizer: _FakeDiarizer(activity(6, [(0, 0, 3), (1, 3, 6)])))..splitSpeakers = false;
      expect(off.process(twoPeople(), t0), hasLength(1));
      expect(processor(['whole'], diarizer: _FakeDiarizer(null)).process(twoPeople(), t0), hasLength(1));
    });

    test('each saved line gets its own audio for voice clips', () {
      final clipLengths = <int>[];
      final p = SegmentProcessor(
        asr: FakeAsr(['a b', 'c d']),
        embedder: FakeEmbedder(),
        identifier: SpeakerIdentifier(profiles: speakers.profiles(), newClusterId: SpeakerRepository.newId, nextGuestLabel: speakers.nextGuestLabel),
        transcripts: transcripts,
        speakers: speakers,
        diarizer: _FakeDiarizer(activity(6, [(0, 0, 3), (1, 3, 6)])),
        onSaved: (_, samples) => clipLengths.add(samples.length),
      );
      p.process(twoPeople(), t0);
      expect(clipLengths, [48000, 48000]);
    });
  });
}

class _FakeDiarizer implements DiarizationEngine {
  _FakeDiarizer(this.result);
  final SpeakerActivity? result;

  @override
  SpeakerActivity analyze(Float32List samples, int sampleRate) => result ?? (throw StateError('diarizer crashed'));

  @override
  void dispose() {}
}
