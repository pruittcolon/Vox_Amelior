import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';
import 'package:vox_amelior_mobile/pipeline/compute_scheduler.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speech_capture.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

void main() {
  late Directory tmp;
  setUp(() => tmp = Directory.systemTemp.createTempSync('vox_queue_'));
  tearDown(() => tmp.deleteSync(recursive: true));

  group('ChunkQueue', () {
    test('keeps chunks on disk in time order with their timestamps', () {
      final q = ChunkQueue(Directory(p.join(tmp.path, 'q')));
      final t = DateTime(2026, 5, 1, 9);
      q.push(Float32List.fromList([0.25, -0.5]), t.add(const Duration(seconds: 5)));
      q.push(Float32List.fromList([1, 2, 3]), t);
      expect(q.length, 2);
      final first = q.take()!;
      expect(first.startedAt, t);
      expect(first.read(), [1, 2, 3]);
      q.done(first);
      final second = q.take()!;
      expect(second.read(), [0.25, -0.5]);
      q.done(second);
      expect(q.take(), isNull);
    });

    test('survives a restart; a chunk that was being processed is set aside', () {
      final dir = Directory(p.join(tmp.path, 'q'));
      final q = ChunkQueue(dir);
      q.push(Float32List(10), DateTime(2026));
      q.push(Float32List(10), DateTime(2026, 1, 2));
      q.take(); // "crashes" while working on it
      final reopened = ChunkQueue(dir);
      expect(reopened.length, 1);
      expect(dir.listSync().where((f) => f.path.endsWith('.failed')).length, 1);
    });

    test('drops the oldest audio when the backlog is too large', () {
      final q = ChunkQueue(Directory(p.join(tmp.path, 'q')), maxBytes: 100);
      for (var i = 0; i < 5; i++) {
        q.push(Float32List(10), DateTime(2026, 1, 1, 0, i)); // 40 bytes each
      }
      expect(q.length, 2);
      expect(q.take()!.startedAt, DateTime(2026, 1, 1, 0, 3));
    });
  });

  group('ComputeScheduler', () {
    late AppDatabase db;
    late ChunkQueue queue;
    late FakeAsr asr;
    late ComputeScheduler scheduler;
    late List<String> log;

    setUp(() {
      db = AppDatabase.inMemory();
      final speakers = SpeakerRepository(db);
      queue = ChunkQueue(Directory(p.join(tmp.path, 'q')));
      asr = FakeAsr([]);
      log = [];
      scheduler = ComputeScheduler(
        queue: queue,
        processor: SegmentProcessor(
          asr: asr,
          embedder: FakeEmbedder(),
          identifier: SpeakerIdentifier(profiles: const [], newClusterId: SpeakerRepository.newId, nextGuestLabel: speakers.nextGuestLabel),
          transcripts: TranscriptRepository(db),
          speakers: speakers,
        ),
        onSegment: (SegmentView s) => log.add('saved ${s.text}'),
      );
    });
    tearDown(() => db.close());

    Future<void> settle() async {
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(Duration.zero);
      }
    }

    test('transcribes queued speech in order', () async {
      asr.outputs.addAll(['first words', 'second words']);
      queue.push(fakeAudio(1), DateTime(2026, 1, 1, 9));
      queue.push(fakeAudio(1), DateTime(2026, 1, 1, 9, 1));
      scheduler.kick();
      await settle();
      expect(log, ['saved first words', 'saved second words']);
      expect(queue.length, 0);
    });

    test('an assistant job runs alone; speech waits and is transcribed afterwards', () async {
      final gate = Completer<void>();
      final job = scheduler.runExclusive(() async {
        log.add('assistant start');
        await gate.future;
        log.add('assistant end');
        return 42;
      });
      await settle();
      asr.outputs.add('said during thinking');
      queue.push(fakeAudio(1), DateTime(2026));
      scheduler.kick();
      await settle();
      expect(log, ['assistant start']);
      expect(scheduler.activity, SchedulerActivity.assistant);
      gate.complete();
      expect(await job, 42);
      await settle();
      expect(log, ['assistant start', 'assistant end', 'saved said during thinking']);
    });

    test('assistant jobs jump ahead of a transcription backlog', () async {
      asr.outputs.addAll(['a', 'bb', 'cc']);
      for (var i = 0; i < 3; i++) {
        queue.push(fakeAudio(1), DateTime(2026, 1, 1, 0, i));
      }
      scheduler.kick();
      unawaited(scheduler.runExclusive(() async => log.add('assistant')));
      await settle();
      expect(log.first, anyOf('saved a', 'assistant'));
      expect(log.indexOf('assistant'), lessThanOrEqualTo(1));
      expect(log.where((l) => l.startsWith('saved')).length, 2); // 'a' is too short and dropped
    });

    test('errors in an assistant job reach the caller and do not stop transcription', () async {
      await expectLater(scheduler.runExclusive<void>(() async => throw StateError('boom')), throwsStateError);
      asr.outputs.add('still working');
      queue.push(fakeAudio(1), DateTime(2026));
      scheduler.kick();
      await settle();
      expect(log, ['saved still working']);
    });

    test('holding transcription keeps audio queued until released', () async {
      scheduler.transcriptionPaused = true;
      asr.outputs.add('held words');
      queue.push(fakeAudio(1), DateTime(2026));
      scheduler.kick();
      await settle();
      expect(log, isEmpty);
      expect(queue.length, 1);
      scheduler.transcriptionPaused = false;
      await settle();
      expect(log, ['saved held words']);
    });
  });

  group('SpeechCapture', () {
    test('re-anchors timestamps when the microphone drops audio', () {
      var now = DateTime(2026, 1, 1, 12);
      final vad = FakeVad();
      final capture = SpeechCapture(vad, clock: () => now);
      capture.addSamples(Float32List(16000)); // 1 s, origin = 11:59:59
      now = now.add(const Duration(seconds: 11)); // 10 s of audio went missing
      vad.ready.add(SpeechChunkFake.at(1.0));
      final out = capture.addSamples(Float32List(16000));
      // Without re-anchoring the chunk would be stamped 10 s too early.
      expect(out.single.startedAt.difference(DateTime(2026, 1, 1, 12, 0, 10)).inMilliseconds.abs(), lessThan(50));
    });
  });
}
