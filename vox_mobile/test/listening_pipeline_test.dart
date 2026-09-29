import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/listening_pipeline.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late FakeVad vad;
  late FakeAsr asr;
  late ListeningPipeline pipeline;
  late List<SegmentView> notified;
  final t0 = DateTime(2026, 3, 1, 12);

  ListeningPipeline build(List<String> texts) {
    vad = FakeVad();
    asr = FakeAsr(texts);
    notified = [];
    final identifier = SpeakerIdentifier(
      profiles: speakers.profiles(),
      clusters: speakers.clusters(),
      newClusterId: SpeakerRepository.newId,
      nextGuestLabel: speakers.nextGuestLabel,
    );
    return ListeningPipeline(
      vad: vad,
      asr: asr,
      embedder: FakeEmbedder(),
      identifier: identifier,
      transcripts: transcripts,
      speakers: speakers,
      clock: () => t0,
      onSegment: notified.add,
    );
  }

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    enrollFake(speakers, 'Alex', 1);
    enrollFake(speakers, 'Sam', 2);
  });
  tearDown(() => db.close());

  SpeechChunk chunk(int speaker, double start, {double seconds = 3}) =>
      SpeechChunk(fakeAudio(speaker, seconds: seconds), start);

  test('transcribes, identifies speakers and stores segments with timestamps', () {
    pipeline = build(['what time is the plumber coming', 'around four I think']);
    vad.ready.addAll([chunk(1, 0), chunk(2, 4)]);

    final out = pipeline.addSamples(Float32List(16000)); // 1s of audio triggers processing
    expect(out.map((s) => s.speakerName), ['Alex', 'Sam']);
    expect(out.map((s) => s.text), ['what time is the plumber coming', 'around four I think']);
    // Audio started 1s before "now" (t0); chunk 2 starts 4s after that.
    expect(out[0].startedAt, t0.subtract(const Duration(seconds: 1)));
    expect(out[1].startedAt, t0.add(const Duration(seconds: 3)));
    expect(notified.length, 2);
    expect(transcripts.count(), 2);
  });

  test('utterances close together share a conversation; a long gap starts a new one', () {
    pipeline = build(['one', 'two', 'three']);
    vad.ready.addAll([chunk(1, 0), chunk(2, 10), chunk(1, 60 * 20)]);
    final out = pipeline.addSamples(Float32List(16000));
    expect(out[0].conversationId, out[1].conversationId);
    expect(out[2].conversationId, isNot(out[0].conversationId));
  });

  test('unknown voices become a persistent guest, reused on later utterances', () {
    pipeline = build(['hello there', 'me again']);
    vad.ready.addAll([chunk(7, 0), chunk(7, 5)]);
    final out = pipeline.addSamples(Float32List(16000));
    expect(out[0].clusterLabel, 'Guest 1');
    expect(out[1].clusterLabel, 'Guest 1');
    expect(speakers.clusters().single.count, 2);
  });

  test('naming a guest relabels history and improves future recognition', () {
    pipeline = build(['hello there', 'later on']);
    vad.ready.add(chunk(7, 0));
    pipeline.addSamples(Float32List(16000));
    final cluster = speakers.clusters().single;
    final jo = enrollFake(speakers, 'Jo', 7, samples: 3);
    speakers.assignClusterToSpeaker(cluster.id, jo.id);
    expect(transcripts.recent().single.speakerName, 'Jo');
    expect(speakers.clusters(), isEmpty);
  });

  test('empty or junk transcripts are dropped, not stored', () {
    pipeline = build(['', '...', 'ok']);
    vad.ready.addAll([chunk(1, 0), chunk(1, 4), chunk(1, 8)]);
    final out = pipeline.addSamples(Float32List(16000));
    expect(out.map((s) => s.text), ['ok']);
    expect(pipeline.stats.dropped, 2);
  });

  test('one failing chunk does not stop the following ones', () {
    pipeline = build(['fine']);
    asr.throwNext = true;
    vad.ready.addAll([chunk(1, 0), chunk(1, 4)]);
    final out = pipeline.addSamples(Float32List(16000));
    expect(out.single.text, 'fine');
    expect(pipeline.stats.errors, 1);
  });

  test('short utterances are stored but left unattributed', () {
    pipeline = build(['yes']);
    vad.ready.add(chunk(1, 0, seconds: 0.5));
    final out = pipeline.addSamples(Float32List(16000));
    expect(out.single.speakerId, isNull);
    expect(out.single.clusterId, isNull);
    expect(speakers.clusters(), isEmpty);
  });

  test('addPcm16 converts little-endian samples and carries odd bytes across calls', () {
    pipeline = build([]);
    // 0x4000 = 16384 -> 0.5 ; 0xC000 = -16384 -> -0.5 ; send split mid-sample.
    final bytes = Uint8List.fromList([0x00, 0x40, 0x00, 0xC0]);
    pipeline.addPcm16(Uint8List.sublistView(bytes, 0, 3));
    expect(vad.accepted, 1);
    pipeline.addPcm16(Uint8List.sublistView(bytes, 3));
    expect(vad.accepted, 2);
  });

  test('flush processes trailing speech and resets the stream', () {
    pipeline = build(['last words']);
    vad.onFlush.add(chunk(2, 1));
    final out = pipeline.flush();
    expect(out.single.speakerName, 'Sam');
    expect(vad.resets, 1);
  });
}
