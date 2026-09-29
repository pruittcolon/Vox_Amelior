import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';
import 'package:vox_amelior_mobile/pipeline/compute_scheduler.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late SpeakerRepository speakers;
  late TranscriptRepository transcripts;
  late FakeLlm llm;

  setUp(() {
    db = AppDatabase.inMemory();
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    llm = FakeLlm();
    transcripts.addSegment(
      text: 'the plumber is coming at four',
      startedAt: DateTime.now().subtract(const Duration(hours: 1)),
      duration: const Duration(seconds: 3),
    );
  });
  tearDown(() => db.close());

  AssistantService service({int context = 4096}) => AssistantService(
        llm: llm,
        transcripts: transcripts,
        speakers: speakers,
        toolbox: toolboxFor(db, transcripts, speakers),
        budget: () => ContextBudget(context),
      );

  group('Gemma resilience', () {
    test('every tool declares parameters (LiteRT-LM fails all requests otherwise)', () {
      final tools = toolboxFor(db, transcripts, speakers).specs;
      expect(tools.map((t) => t.name), contains('list_notes'));
      for (final t in tools) {
        expect(t.parameters['type'], 'object', reason: t.name);
        expect((t.parameters['properties'] as Map?)?.isNotEmpty, isTrue, reason: '${t.name} has no parameters');
      }
    });

    test('a failure with tools is retried without them', () async {
      llm.failWithTools = true;
      final a = await service().answer('When is the plumber coming?');
      expect(a.text, contains('plumber'));
      expect(llm.lastTools, isEmpty);
    });

    test('a generation failure is retried once in a safer mode (CPU)', () async {
      llm
        ..tools = false
        ..failSends = 1;
      final a = await service().answer('When is the plumber coming?');
      expect(a.text, isNotEmpty);
      expect(llm.recovers, 1);
    });

    test('when everything fails the user gets a plain explanation, not an exception dump', () async {
      llm
        ..tools = false
        ..canRecover = false
        ..failSends = 5;
      await expectLater(
        service().answer('When is the plumber coming?'),
        throwsA(isA<LlmUnavailable>().having((e) => e.message, 'message', contains('Test this phone'))),
      );
    });

    test('a silent model times out instead of hanging', () async {
      final silent = _SilentLlm();
      final s = AssistantService(
        llm: silent,
        transcripts: transcripts,
        speakers: speakers,
        idleTimeout: const Duration(milliseconds: 50),
      );
      await expectLater(s.answer('anything?'), throwsA(isA<LlmUnavailable>()));
      expect(silent.closed, isTrue);
    });

    test('the context size limits excerpts and replies', () async {
      for (var i = 0; i < 300; i++) {
        transcripts.addSegment(
          text: 'plumber note number $i with a fairly long sentence about pipes and appointments',
          startedAt: DateTime.now().subtract(Duration(minutes: 300 - i)),
          duration: const Duration(seconds: 3),
        );
      }
      llm.tools = false;
      await service(context: 2048).answer('Summarise today');
      final small = llm.lastPrompt!.length;
      final smallReply = llm.lastMaxReply;
      await service(context: 16384).answer('Summarise today');
      expect(llm.lastPrompt!.length, greaterThan(small));
      expect(llm.lastMaxReply, greaterThan(smallReply!));
      expect(small, lessThanOrEqualTo(ContextBudget.charsFor(const ContextBudget(2048).excerptTokens(agent: false)) + 400));
    });
  });

  group('context budget', () {
    test('recommends half the context per review part, within safe bounds', () {
      const b = ContextBudget(8192);
      expect(b.recommendedChunkTokens, 4096);
      expect(b.chunkTokens, 4096);
      expect(const ContextBudget(8192, chunkTokens: 100000).chunkTokens, b.maxChunkTokens);
      expect(const ContextBudget(2048).chunkTokens, greaterThanOrEqualTo(512));
      expect(b.maxChunkTokens + b.replyTokens + 700, lessThanOrEqualTo(8192));
    });
  });

  group('voice clips', () {
    late Directory tmp;
    late ClipStore clips;
    setUp(() {
      tmp = Directory.systemTemp.createTempSync('vox_clips_');
      clips = ClipStore(db, Directory(p.join(tmp.path, 'clips')));
    });
    tearDown(() => tmp.deleteSync(recursive: true));

    Float32List audio(double seconds) => Float32List.fromList(List.filled((seconds * 16000).round(), 0.1));

    test('saves nothing when off; saves WAV + text for everyone or chosen people', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final seg = transcripts.addSegment(text: 'hello there', startedAt: DateTime(2026, 9, 1), duration: const Duration(seconds: 2), speakerId: alex.id);
      final other = transcripts.addSegment(text: 'who am I', startedAt: DateTime(2026, 9, 1, 0, 1), duration: const Duration(seconds: 2));
      expect(clips.maybeSave(const ClipPolicy(), seg, audio(2)), isFalse);
      expect(clips.maybeSave(ClipPolicy(mode: ClipMode.chosen, people: [alex.id]), other, audio(2)), isFalse);
      expect(clips.maybeSave(ClipPolicy(mode: ClipMode.chosen, people: [alex.id]), seg, audio(2)), isTrue);
      expect(clips.maybeSave(const ClipPolicy(mode: ClipMode.everyone), other, audio(1)), isTrue);

      final stats = clips.stats();
      expect(stats.count, 2);
      expect(stats.duration, const Duration(seconds: 3));
      expect(stats.groups.map((g) => g.name), containsAll(['Alex', 'Not named']));
      final wav = File(p.join(clips.dir.path, '2026-09', '${seg.startedAt.millisecondsSinceEpoch}_${seg.id}.wav')).readAsBytesSync();
      expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
      expect(ByteData.sublistView(wav).getUint32(24, Endian.little), 16000);
      expect(wav.length, 44 + 2 * 16000 * 2);
      expect(clips.writeManifest().readAsLinesSync().first, contains('"text":"hello there"'));
    });

    test('stops at the size limit, and deletes per person or everything', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      const policy = ClipPolicy(mode: ClipMode.everyone, limitBytes: 100000);
      var saved = 0;
      for (var i = 0; i < 5; i++) {
        final seg = transcripts.addSegment(
          text: 'line $i',
          startedAt: DateTime(2026, 9, 2, 0, i),
          duration: const Duration(seconds: 1),
          speakerId: i.isEven ? alex.id : null,
        );
        if (clips.maybeSave(policy, seg, audio(1))) saved++;
      }
      expect(saved, 3); // 32 KB each, 100 KB limit
      expect(clips.deleteForSpeaker(alex.id), 2);
      expect(clips.stats().count, 1);
      clips.deleteAll();
      expect(clips.stats().count, 0);
      expect(clips.dir.listSync(recursive: true).whereType<File>(), isEmpty);
    });

    test('relabelling a line updates its clip; deleting a conversation removes its clips', () {
      final alex = speakers.create(name: 'Alex', embeddingModel: 'f', samples: [voiceprint(1)]);
      final seg = transcripts.addSegment(
        text: 'mine',
        startedAt: DateTime(2026, 9, 3),
        duration: const Duration(seconds: 1),
        embedding: voiceprint(1, variant: 2),
      );
      clips.maybeSave(const ClipPolicy(mode: ClipMode.everyone), seg, audio(1));
      speakers.assignSegmentToSpeaker(seg.id, alex.id);
      expect(clips.stats().groups.single.name, 'Alex');
      clips.deleteForConversation(seg.conversationId);
      expect(clips.stats().count, 0);
    });
  });

  test('review steps wait until live questions and transcription are done', () async {
    final tmp = Directory.systemTemp.createTempSync('vox_sched_');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final queue = ChunkQueue(Directory(p.join(tmp.path, 'q')));
    final order = <String>[];
    final scheduler = ComputeScheduler(
      queue: queue,
      processor: SegmentProcessor(
        asr: FakeAsr(['one', 'two']),
        embedder: FakeEmbedder(),
        identifier: SpeakerIdentifier(profiles: const [], newClusterId: () => 'c', nextGuestLabel: () => 'Guest'),
        transcripts: transcripts,
        speakers: speakers,
      ),
      onSegment: (s) => order.add('speech:${s.text}'),
    );
    queue
      ..push(fakeAudio(1), DateTime(2026))
      ..push(fakeAudio(1), DateTime(2026, 1, 1, 0, 1));
    final ask = scheduler.runExclusive(() async => order.add('ask'));
    final review = scheduler.runBackground(() async => order.add('review'));
    await Future.wait([review, ask]);
    expect(order, ['ask', 'speech:one', 'speech:two', 'review']);
  });

  test('new settings survive a round trip and bad values fall back', () {
    final s = const AppSettings().copyWith(
      multiPatterns: false,
      clipMode: ClipMode.chosen,
      clipPeople: ['a'],
      clipLimitMb: 5120,
      contextTokens: 8192,
      contextTested: 8192,
      contextTestNote: 'Works up to 8,192 tokens.',
      reviewChunkTokens: 3000,
      themeMode: 'dark',
      accent: 0xFF0B7285,
      textScale: 1.2,
      corners: 'soft',
    );
    final back = AppSettings.fromJson(s.toJson());
    expect(back.multiPatterns, isFalse);
    expect(back.identifierConfig.usePatterns, isFalse);
    expect(back.clipMode, ClipMode.chosen);
    expect(back.clipPolicy.allows('a'), isTrue);
    expect(back.clipPolicy.allows('b'), isFalse);
    expect(back.contextTokens, 8192);
    expect(back.budget.chunkTokens, 3000);
    expect(back.themeMode, 'dark');
    expect(back.textScale, 1.2);
    expect(back.corners, 'soft');
    final bad = AppSettings.fromJson({'contextTokens': 5, 'themeMode': 'neon', 'textScale': 9, 'clipMode': 'x', 'corners': 1});
    expect(bad.contextTokens, ContextBudget.defaultContext);
    expect(bad.themeMode, 'system');
    expect(bad.textScale, 1.4);
    expect(bad.clipMode, ClipMode.off);
    expect(bad.corners, 'rounded');
  });
}

/// A model that never says anything.
class _SilentLlm implements LlmEngine {
  bool closed = false;
  @override
  bool get isLoaded => true;
  @override
  bool get supportsTools => false;
  @override
  Future<void> ensureLoaded({int? contextTokens}) async {}
  @override
  Future<bool> recover() async => false;
  @override
  Future<void> unload() async {}
  @override
  Future<LlmSession> openSession({
    required String system,
    List<ToolSpec> tools = const [],
    int? maxReplyTokens,
    int? contextTokens,
  }) async =>
      _SilentSession(this);
}

class _SilentSession implements LlmSession {
  _SilentSession(this.llm);
  final _SilentLlm llm;
  @override
  Stream<LlmEvent> send(String text) => StreamController<LlmEvent>().stream;
  @override
  Stream<LlmEvent> sendToolResult(String name, Map<String, Object?> result) => const Stream.empty();
  @override
  Future<int?> countTokens(String text) async => null;
  @override
  Future<void> close() async => llm.closed = true;
}
