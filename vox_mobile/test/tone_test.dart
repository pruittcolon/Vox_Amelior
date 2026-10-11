import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sqlite3/sqlite3.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

/// Hears a scripted tone per call; records how much audio it was given.
class FakeTone implements ToneEngine {
  FakeTone(this.script);

  final List<LineTone> script;
  final List<int> heardSamples = [];
  bool fail = false;
  bool disposed = false;

  @override
  LineTone analyze(Float32List samples, int sampleRate) {
    heardSamples.add(samples.length);
    if (fail) throw StateError('tone model crashed');
    return script.isEmpty ? const LineTone() : script.removeAt(0);
  }

  @override
  void dispose() => disposed = true;
}

class _TwoVoices implements DiarizationEngine {
  /// Speaker slot 0 for the first half of the audio, slot 1 for the second.
  @override
  SpeakerActivity analyze(Float32List samples, int sampleRate) {
    const fs = 0.01;
    const speakers = 4;
    final frames = (samples.length / sampleRate / fs).round();
    final p = Float32List(frames * speakers);
    for (var t = 0; t < frames; t++) {
      p[t * speakers + (t < frames ~/ 2 ? 0 : 1)] = 0.9;
    }
    return SpeakerActivity(p, frames: frames, speakers: speakers, frameSeconds: fs);
  }

  @override
  void dispose() {}
}

void main() {
  group('reading SenseVoice tags', () {
    test('emotions', () {
      expect(Tone.fromEmotionTag('<|ANGRY|>'), 'angry');
      expect(Tone.fromEmotionTag('<|HAPPY|>'), 'happy');
      expect(Tone.fromEmotionTag('<|SAD|>'), 'sad');
      expect(Tone.fromEmotionTag('<|NEUTRAL|>'), 'neutral');
      expect(Tone.fromEmotionTag('<|FEARFUL|>'), 'fearful');
      expect(Tone.fromEmotionTag('<|DISGUSTED|>'), 'disgusted');
      expect(Tone.fromEmotionTag('<|SURPRISED|>'), 'surprised');
      expect(Tone.fromEmotionTag('happy'), 'happy', reason: 'bare words work too');
      expect(Tone.fromEmotionTag('<|EMO_UNKNOWN|>'), isNull);
      expect(Tone.fromEmotionTag(''), isNull);
      expect(Tone.fromEmotionTag('<|zh|>'), isNull);
    });

    test('sounds', () {
      expect(Tone.fromEventTag('<|Laughter|>'), 'laughter');
      expect(Tone.fromEventTag('<|BGM|>'), 'music');
      expect(Tone.fromEventTag('<|Applause|>'), 'applause');
      expect(Tone.fromEventTag('<|Cry|>'), 'crying');
      expect(Tone.fromEventTag('<|Cough|>'), 'coughing');
      expect(Tone.fromEventTag('<|Sneeze|>'), 'sneezing');
      expect(Tone.fromEventTag('<|Speech|>'), isNull, reason: 'plain speech is not a sound');
      expect(Tone.fromEventTag('<|Breath|>'), isNull);
      expect(Tone.fromEventTag(''), isNull);
    });

    test('every tone has an emoji and a label', () {
      for (final t in [...Tone.names, ...Tone.sounds]) {
        expect(Tone.emoji(t), isNot('•'), reason: t);
        expect(Tone.label(t), startsWith(t[0].toUpperCase()));
      }
    });

    test('mood counts put the most common non-neutral tone first', () {
      const m = MoodCount({'neutral': 9, 'angry': 2, 'happy': 5});
      expect(m.total, 16);
      expect(m.notable.map((e) => e.key), ['happy', 'angry']);
      expect(const MoodCount({}).isEmpty, isTrue);
    });
  });

  group('the tone model', () {
    test('is an optional download that installs only its two files', () {
      const a = ModelCatalog.toneModel;
      expect(a.essential, isFalse);
      expect(a.installedFileNames, {'model.int8.onnx', 'tokens.txt'});
      expect(a.files.single.url, contains('sense-voice'));
      expect(a.files.single.sha256, hasLength(64));
      expect(ModelCatalog.all, contains(a));
      expect(ModelCatalog.speech, isNot(contains(a)), reason: 'not downloaded at setup');
      expect(ModelCatalog.speechExtras, isNot(contains(a)));
    });

    test('its paths reach the listening service and survive the JSON hand-over', () {
      const paths = SpeechModelPaths(
        encoder: 'e',
        decoder: 'd',
        joiner: 'j',
        tokens: 't',
        vad: 'v',
        speaker: 's',
        toneModel: '/m/model.int8.onnx',
        toneTokens: '/m/tokens.txt',
      );
      final back = SpeechModelPaths.fromJson(paths.toJson());
      expect(back.toneModel, '/m/model.int8.onnx');
      expect(back.toneTokens, '/m/tokens.txt');
      expect(back.missingFiles(), isNot(contains('/m/model.int8.onnx')), reason: 'optional, never blocks listening');
      final old = SpeechModelPaths.fromJson({'encoder': 'e', 'decoder': 'd', 'joiner': 'j', 'tokens': 't', 'vad': 'v', 'speaker': 's'});
      expect(old.toneModel, isNull);
    });

    test('the "hear tone" setting defaults on and round-trips', () {
      expect(const AppSettings().hearTone, isTrue);
      final off = const AppSettings().copyWith(hearTone: false);
      expect(AppSettings.fromJson(off.toJson()).hearTone, isFalse);
      expect(AppSettings.fromJson(const {}).hearTone, isTrue);
    });
  });

  group('lines get a tone when saved', () {
    late AppDatabase db;
    late SpeakerRepository speakers;
    late TranscriptRepository transcripts;
    final t0 = DateTime(2026, 10, 8, 20);
    setUp(() {
      db = AppDatabase.inMemory();
      speakers = SpeakerRepository(db);
      transcripts = TranscriptRepository(db);
      enrollFake(speakers, 'Pruitt', 1);
      enrollFake(speakers, 'Ericah', 2);
    });
    tearDown(() => db.close());

    SegmentProcessor processor(List<String> texts, {ToneEngine? tone, DiarizationEngine? diarizer}) => SegmentProcessor(
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
          tone: tone,
        );

    test('the tone is stored and returned with the line', () {
      final tone = FakeTone([const LineTone(emotion: 'angry', sound: 'laughter')]);
      final saved = processor(['you never listen'], tone: tone).process(fakeAudio(1), t0).single;
      expect(saved.emotion, 'angry');
      expect(saved.sound, 'laughter');
      expect(transcripts.segment(saved.id)!.emotion, 'angry');
    });

    test('a line shows at once; its tone is added in the second pass', () {
      final clips = <SegmentView>[];
      final tone = FakeTone([const LineTone(emotion: 'happy')]);
      final p = SegmentProcessor(
        asr: FakeAsr(['hello there']),
        embedder: FakeEmbedder(),
        identifier: SpeakerIdentifier(profiles: speakers.profiles(), newClusterId: SpeakerRepository.newId, nextGuestLabel: speakers.nextGuestLabel),
        transcripts: transcripts,
        speakers: speakers,
        tone: tone,
        onSaved: (s, _) => clips.add(s),
      );
      final fast = p.transcribe(fakeAudio(1), t0).single;
      expect(fast.emotion, isNull);
      expect(tone.heardSamples, isEmpty, reason: 'the first pass does not wait for the tone model');
      expect(clips.single.id, fast.id, reason: 'the clip is kept straight away');

      final finished = p.refine(t0.add(const Duration(minutes: 1)));
      expect(finished.single.emotion, 'happy');
      expect(transcripts.segment(fast.id)!.emotion, 'happy');
    });

    test('a line re-labelled by hand before the second pass still gets its tone', () {
      final p = processor(['hello there'], tone: FakeTone([const LineTone(emotion: 'surprised')]));
      final fast = p.transcribe(fakeAudio(1), t0).single;
      speakers.assignSegmentToSpeaker(fast.id, speakers.profiles().last.id);
      p.refine(t0.add(const Duration(minutes: 1)));
      expect(transcripts.segment(fast.id)!.emotion, 'surprised');
    });

    test('a line deleted before the second pass is skipped', () {
      final tone = FakeTone([const LineTone(emotion: 'sad')]);
      final p = processor(['hello there'], tone: tone);
      final fast = p.transcribe(fakeAudio(1), t0).single;
      transcripts.deleteConversation(fast.conversationId);
      expect(p.refine(t0.add(const Duration(minutes: 1))), isEmpty);
      expect(tone.heardSamples, isEmpty);
    });

    test('without the model, or switched off, lines have no tone', () {
      final none = processor(['first line']).process(fakeAudio(1), t0).single;
      expect(none.emotion, isNull);
      final tone = FakeTone([const LineTone(emotion: 'sad')]);
      final off = processor(['second line'], tone: tone)..hearTone = false;
      final line = off.process(fakeAudio(1), t0.add(const Duration(minutes: 1))).single;
      expect(line.emotion, isNull);
      expect(tone.heardSamples, isEmpty, reason: 'the model is not even run');
    });

    test('a crashing tone model never loses the line', () {
      final tone = FakeTone([])..fail = true;
      final saved = processor(['still saved'], tone: tone).process(fakeAudio(1), t0);
      expect(saved.single.text, 'still saved');
      expect(saved.single.emotion, isNull);
      expect(transcripts.count(), 1);
    });

    test('only the first 15 s of a long line are analysed', () {
      final tone = FakeTone([]);
      processor(['a long monologue'], tone: tone).process(fakeAudio(1, seconds: 40), t0);
      expect(tone.heardSamples.single, 15 * 16000);
    });

    test('a line split at a speaker change gets a tone per part', () {
      final tone = FakeTone([
        const LineTone(emotion: 'angry'), // Pruitt's part
        const LineTone(emotion: 'sad'), // Ericah's part
      ]);
      final audio = Float32List.fromList([...fakeAudio(1), ...fakeAudio(2)]);
      final parts = processor(['why did you do that I am sorry okay'], tone: tone, diarizer: _TwoVoices()).process(audio, t0);
      expect(parts, hasLength(2));
      expect(parts.map((p) => p.emotion), ['angry', 'sad']);
      expect(tone.heardSamples, [48000, 48000], reason: 'each part once; the uncut line is never analysed');
    });

    test('disposing the processor frees the tone model', () {
      final tone = FakeTone([]);
      processor(const [], tone: tone).dispose();
      expect(tone.disposed, isTrue);
    });
  });

  group('sorting and filtering by tone', () {
    late AppDatabase db;
    late SpeakerRepository speakers;
    late TranscriptRepository transcripts;
    late SpeakerProfile pruitt;
    late SpeakerProfile ericah;
    final t0 = DateTime(2026, 10, 1, 19);
    setUp(() {
      db = AppDatabase.inMemory();
      speakers = SpeakerRepository(db);
      transcripts = TranscriptRepository(db);
      pruitt = speakers.create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]);
      ericah = speakers.create(name: 'Ericah', embeddingModel: 'm', samples: [voiceprint(2)]);
    });
    tearDown(() => db.close());

    SegmentView say(String text, int minute, {String? speakerId, String? emotion, String? sound, Float32List? embedding}) {
      final s = transcripts.addSegment(
        text: text,
        startedAt: t0.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 3),
        speakerId: speakerId,
        embedding: embedding,
      );
      if (emotion != null || sound != null) transcripts.setTone(s.id, emotion: emotion, sound: sound);
      return transcripts.segment(s.id)!;
    }

    test('nothing has a tone until the model runs', () {
      say('hi', 0, speakerId: pruitt.id);
      expect(transcripts.hasTones, isFalse);
      say('ha', 1, speakerId: ericah.id, sound: 'laughter');
      expect(transcripts.hasTones, isTrue);
    });

    test('mood of a conversation counts tones, leaving out TV voices', () {
      say('you always do this', 0, speakerId: pruitt.id, emotion: 'angry');
      say('I do not', 1, speakerId: ericah.id, emotion: 'angry');
      say('fine', 2, speakerId: ericah.id, emotion: 'sad');
      say('ok', 3, speakerId: pruitt.id, emotion: 'neutral');
      final tv = say('Previously on', 4, speakerId: pruitt.id, emotion: 'happy', embedding: voiceprint(77));
      speakers.markBackground(tv.id);
      final mood = transcripts.mood(tv.conversationId);
      expect(mood.counts, {'angry': 2, 'sad': 1, 'neutral': 1});
      expect(mood.notable.first.key, 'angry');
    });

    test('conversations with a tone, optionally by some people', () {
      say('so happy', 0, speakerId: pruitt.id, emotion: 'happy'); // conversation 1
      say('stop it', 60, speakerId: ericah.id, emotion: 'angry'); // conversation 2
      say('ha ha', 120, speakerId: pruitt.id, sound: 'laughter'); // conversation 3
      expect(transcripts.conversationsWithTone({'angry'}).map((c) => c.preview), ['stop it']);
      expect(transcripts.conversationsWithTone({'angry', 'happy'}), hasLength(2));
      expect(transcripts.conversationsWithTone({'laughter'}).single.preview, 'ha ha', reason: 'sounds count too');
      expect(transcripts.conversationsWithTone({'angry'}, speakerIds: {pruitt.id}), isEmpty);
      expect(transcripts.conversationsWithTone({}), isEmpty);
    });

    test('with two people, both must have talked and one of them sounded that way', () {
      say('stop it', 0, speakerId: ericah.id, emotion: 'angry');
      say('sorry', 1, speakerId: pruitt.id, emotion: 'sad'); // conversation 1: both, Ericah angry
      say('why', 60, speakerId: ericah.id, emotion: 'angry'); // conversation 2: only Ericah
      say('fine', 120, speakerId: pruitt.id, emotion: 'neutral');
      say('ok', 121, speakerId: ericah.id, emotion: 'neutral'); // conversation 3: both, calm
      final both = {pruitt.id, ericah.id};
      expect(transcripts.conversationsWithTone({'angry'}, speakerIds: both).single.preview, 'stop it');
      expect(transcripts.conversationsWithTone({'sad', 'angry'}, speakerIds: both), hasLength(1));
      expect(transcripts.conversationsWithTone({'angry'}, speakerIds: {ericah.id}), hasLength(2));
      final first = transcripts.conversationsWithTone({'angry'}, speakerIds: {ericah.id}).first;
      expect(transcripts.conversationsWithTone({'angry'}, speakerIds: {ericah.id}, before: first.startedAt).single.preview, 'stop it');
    });

    test('search can be narrowed to tones', () {
      say('pizza is late', 0, speakerId: pruitt.id, emotion: 'angry');
      say('pizza is here', 1, speakerId: ericah.id, emotion: 'happy');
      final angry = transcripts.search(const SegmentQuery(keywords: ['pizza'], emotions: {'angry'}));
      expect(angry.map((s) => s.text), ['pizza is late']);
      expect(transcripts.search(const SegmentQuery(emotions: {'happy'})).single.text, 'pizza is here');
    });

    group('the last lines people said', () {
      setUp(() {
        for (var i = 0; i < 30; i++) {
          say('line $i', i * 2, speakerId: i % 3 == 2 ? null : (i.isEven ? pruitt.id : ericah.id), emotion: i % 5 == 0 ? 'angry' : 'neutral');
        }
      });

      test('newest N, returned oldest first', () {
        final last = transcripts.lastLines(count: 5);
        expect(last.map((s) => s.text), ['line 25', 'line 26', 'line 27', 'line 28', 'line 29']);
      });

      test('only the chosen people', () {
        final both = transcripts.lastLines(count: 100, speakerIds: {pruitt.id, ericah.id});
        expect(both.every((s) => s.speakerId == pruitt.id || s.speakerId == ericah.id), isTrue);
        expect(both, hasLength(20));
        final hers = transcripts.lastLines(count: 3, speakerIds: {ericah.id});
        // Ericah says the odd lines, except every third line (an unknown voice): 21, 25, 27 are her last three.
        expect(hers.map((s) => s.text), ['line 21', 'line 25', 'line 27']);
      });

      test('only some tones', () {
        final angry = transcripts.lastLines(count: 100, emotions: {'angry'});
        expect(angry.map((s) => s.text), ['line 0', 'line 5', 'line 10', 'line 15', 'line 20', 'line 25']);
      });

      test('within a period', () {
        final early = transcripts.lastLines(count: 100, from: t0, to: t0.add(const Duration(minutes: 10)));
        expect(early.map((s) => s.text), ['line 0', 'line 1', 'line 2', 'line 3', 'line 4']);
      });

      test('TV voices are left out', () {
        final tv = say('Breaking news', 200, speakerId: pruitt.id, embedding: voiceprint(77));
        speakers.markBackground(tv.id);
        expect(transcripts.lastLines(count: 1).single.text, 'line 29');
      });
    });
  });

  group('copying out and back in keeps the tone', () {
    test('export marks tones, import reads them back', () {
      final db = AppDatabase.inMemory();
      addTearDown(db.close);
      final speakers = SpeakerRepository(db);
      final transcripts = TranscriptRepository(db);
      final p = speakers.create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]);
      final a = transcripts.addSegment(text: 'why', startedAt: DateTime(2026, 10, 2, 20), duration: const Duration(seconds: 2), speakerId: p.id);
      transcripts.setTone(a.id, emotion: 'angry', sound: 'laughter');
      transcripts.addSegment(text: 'plain', startedAt: DateTime(2026, 10, 2, 20, 0, 5), duration: const Duration(seconds: 2), speakerId: p.id);
      final text = transcripts.exportText();
      expect(text, contains('Pruitt [angry, laughter]: why'));
      expect(text, contains('Pruitt: plain'));

      final fresh = AppDatabase.inMemory();
      addTearDown(fresh.close);
      final p2 = SpeakerRepository(fresh).create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]);
      final back = TranscriptRepository(fresh);
      expect(back.importText(text).lines, 2);
      final lines = back.lastLines(count: 10);
      expect(lines.first.speakerId, p2.id, reason: 'the name is still linked with the tone after it');
      expect(lines.first.emotion, 'angry');
      expect(lines.first.sound, 'laughter');
      expect(lines.last.emotion, isNull);
    });

    test('brackets that are not tones stay part of the name', () {
      final db = AppDatabase.inMemory();
      addTearDown(db.close);
      final transcripts = TranscriptRepository(db);
      transcripts.importText('--- 2026-09-01T10:00:00.000 ---\nDr Who [the doctor]: hello\nSam [sad]: bye\n');
      final lines = transcripts.lastLines(count: 10);
      expect(lines.first.speakerLabel, 'Dr Who [the doctor]');
      expect(lines.first.emotion, isNull);
      expect(lines.last.speakerLabel, 'Sam');
      expect(lines.last.emotion, 'sad');
    });
  });

  test('an older database keeps every line when upgraded, with no tones yet', () {
    final dir = Directory.systemTemp.createTempSync('vox_v5_');
    addTearDown(() => dir.deleteSync(recursive: true));
    final path = '${dir.path}/vox.db';
    final first = AppDatabase.open(path);
    TranscriptRepository(first).addSegment(text: 'kept', startedAt: DateTime(2026, 10, 1), duration: const Duration(seconds: 2));
    first.close();
    final old = sqlite3.open(path)
      ..execute('ALTER TABLE segments DROP COLUMN emotion')
      ..execute('ALTER TABLE segments DROP COLUMN sound')
      ..userVersion = 5;
    old.close();

    final upgraded = AppDatabase.open(path);
    addTearDown(upgraded.close);
    expect(upgraded.raw.userVersion, AppDatabase.schemaVersion);
    final line = TranscriptRepository(upgraded).search(const SegmentQuery()).single;
    expect(line.text, 'kept');
    expect(line.emotion, isNull);
  });

  group('Gemma reviews of chosen lines', () {
    late AppDatabase db;
    late TranscriptRepository transcripts;
    late SpeakerRepository speakers;
    late ReviewRepository reviews;
    late FakeLlm llm;
    late ReviewEngine engine;
    late String pruitt;
    late String ericah;
    final t0 = DateTime(2026, 10, 3, 21);

    setUp(() {
      db = AppDatabase.inMemory();
      transcripts = TranscriptRepository(db);
      speakers = SpeakerRepository(db);
      reviews = ReviewRepository(db);
      llm = FakeLlm();
      engine = ReviewEngine(reviews: reviews, transcripts: transcripts, llm: llm);
      pruitt = speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]).id;
      ericah = speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]).id;
      for (var i = 0; i < 150; i++) {
        final s = transcripts.addSegment(
          text: 'statement $i',
          startedAt: t0.add(Duration(minutes: i)),
          duration: const Duration(seconds: 3),
          speakerId: i % 10 == 9 ? null : (i.isEven ? pruitt : ericah),
        );
        transcripts.setTone(s.id, emotion: i % 4 == 0 ? 'angry' : 'neutral', sound: i == 149 ? 'laughter' : null);
      }
    });
    tearDown(() => db.close());

    Future<void> runToEnd(int id) async {
      for (var guard = 0; guard < 500; guard++) {
        if (!await engine.step(id)) return;
      }
      fail('review did not finish');
    }

    ReviewTemplate fights() => ReviewTemplate.builtIns.firstWhere((t) => t.id == 'fights');

    int startLast({int count = 100, Set<String>? people, Set<String> emotions = const {}}) => engine.startLast(
          title: fights().name,
          prompt: fights().prompt,
          format: fights().format,
          kind: fights().kind,
          count: count,
          speakerIds: people ?? {pruitt, ericah},
          emotions: emotions,
          budget: const ContextBudget(4096, chunkTokens: 600),
        )!;

    test('reads exactly the last 100 lines by the two of them, oldest first, with their tone', () async {
      llm.responder = (p) => p.contains('Write a short overview') ? 'One argument about the dishes.' : 'NONE';
      final id = startLast();
      final run = reviews.run(id)!;
      expect(run.periodLabel, 'Last 100 lines');
      final ids = [for (final c in reviews.chunks(id)) ...c.segmentIds];
      final lines = transcripts.segmentsByIds(ids);
      expect(lines, hasLength(100));
      expect(lines.every((l) => l.speakerId == pruitt || l.speakerId == ericah), isTrue);
      expect(lines.first.startedAt.isBefore(lines.last.startedAt), isTrue);
      expect(lines.last.text, 'statement 148', reason: 'line 149 was said by an unknown voice');

      await runToEnd(id);
      expect(reviews.run(id)!.status, ReviewStatus.done);
      final firstPart = llm.prompts.first;
      expect(firstPart, contains('Pruitt (angry): statement'));
      expect(firstPart, contains('Ericah: statement'), reason: 'neutral is not mentioned');
      expect(firstPart, isNot(contains('(neutral')));
    });

    test('can be limited to angry lines', () {
      final id = startLast(count: 500, emotions: {'angry'});
      final lines = transcripts.segmentsByIds([for (final c in reviews.chunks(id)) ...c.segmentIds]);
      expect(lines, isNotEmpty);
      expect(lines.every((l) => l.emotion == 'angry'), isTrue);
    });

    test('laughter reaches Gemma as a note on the line', () {
      final line = transcripts.lastLines(count: 1).single;
      expect(line.sound, 'laughter');
      final run = reviews.run(startLast(count: 5, people: const {}))!;
      final prompt = engine.partPrompt(run, 0, transcripts.lastLines(count: 5));
      expect(prompt, contains('(laughing)'));
    });

    test('nothing to read gives no review', () {
      final none = engine.startLast(
        title: 'x',
        prompt: 'x',
        format: 'x',
        kind: ReviewKind.list,
        count: 100,
        emotions: const {'fearful'},
        budget: const ContextBudget(4096),
      );
      expect(none, isNull);
    });

    test('findings from a fights review are saved in the list and link to their lines', () async {
      llm.responder = (p) => p.contains('Write a short overview')
          ? 'They argued twice and made up.'
          : '- [1] "statement" — Started — Pruitt raised his voice\n- [2] "statement" — Calmed down — Ericah apologised';
      final id = startLast(count: 40);
      await runToEnd(id);
      final run = reviews.run(id)!;
      expect(run.status, ReviewStatus.done);
      expect(run.finalAnswer, contains('argued twice'));
      final items = reviews.items(id);
      expect(items, isNotEmpty);
      expect(items.every((i) => i.segmentId != null), isTrue);
      expect(ReviewEngine.countByCategory(items).keys, containsAll(['Started', 'Calmed down']));
      expect(reviews.runs().first.id, id, reason: 'kept in the list of reviews');
    });

    test('the new templates use the list format the app parses', () {
      for (final id in ['fights', 'positions', 'kindness']) {
        final t = ReviewTemplate.builtIns.firstWhere((x) => x.id == id);
        expect(t.kind, ReviewKind.list);
        expect(t.format, contains('[line number]'));
        expect(t.categoryLabel, isNot('Type'));
      }
    });
  });
}
