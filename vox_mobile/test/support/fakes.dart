import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/enrollment_service.dart';

const int kDim = 32;

/// Deterministic "voiceprint" for a synthetic speaker: a fixed base vector plus small noise.
Float32List voiceprint(int speaker, {int variant = 0, double noise = 0.08}) {
  final base = math.Random(1000 + speaker);
  final jitter = math.Random(variant * 7919 + speaker * 31 + 1);
  final v = Float32List(kDim);
  for (var i = 0; i < kDim; i++) {
    v[i] = (base.nextDouble() * 2 - 1) + (jitter.nextDouble() * 2 - 1) * noise;
  }
  return v;
}

/// Audio whose first sample encodes which synthetic speaker "said" it (speaker * 0.001).
/// Length is [seconds] at 16 kHz with audible amplitude.
Float32List fakeAudio(int speaker, {double seconds = 3, double amplitude = 0.1}) {
  final a = Float32List((seconds * 16000).round());
  for (var i = 0; i < a.length; i++) {
    a[i] = amplitude * (i.isEven ? 1 : -1);
  }
  a[0] = speaker * 0.001; // speaker code, at negligible amplitude
  return a;
}

/// Maps the speaker code embedded by [fakeAudio] back to a voiceprint.
class FakeEmbedder implements EmbeddingEngine {
  int calls = 0;

  @override
  String get modelId => 'fake';

  @override
  int get dimension => kDim;

  @override
  Float32List embed(Float32List samples, int sampleRate) {
    calls++;
    return voiceprint((samples[0] * 1000).round(), variant: calls);
  }

  @override
  void dispose() {}
}

// ---- pipeline fakes --------------------------------------------------------

/// Scripted voice-activity detector: releases pre-built chunks when told to.
class FakeVad implements VadEngine {
  final List<SpeechChunk> ready = [];
  final List<SpeechChunk> onFlush = [];
  int accepted = 0;
  int resets = 0;
  bool disposed = false;

  @override
  void accept(Float32List samples) => accepted += samples.length;

  @override
  List<SpeechChunk> takeSegments() {
    final out = List.of(ready);
    ready.clear();
    return out;
  }

  @override
  List<SpeechChunk> flush() {
    final out = List.of(onFlush);
    onFlush.clear();
    return out;
  }

  @override
  void reset() => resets++;

  @override
  void dispose() => disposed = true;
}

/// Returns queued transcripts in order (empty string when the queue runs out).
class FakeAsr implements AsrEngine {
  FakeAsr(this.outputs);

  final List<String> outputs;
  bool throwNext = false;

  @override
  String transcribe(Float32List samples, int sampleRate) {
    if (throwNext) {
      throwNext = false;
      throw StateError('asr failed');
    }
    return outputs.isEmpty ? '' : outputs.removeAt(0);
  }

  @override
  void dispose() {}
}

/// Enrolls synthetic [speaker] under [name] from [samples] fake recordings.
SpeakerProfile enrollFake(SpeakerRepository repo, String name, int speaker, {int samples = 4}) {
  final report = VoiceSampleAnalyzer(FakeEmbedder()).analyze([for (var i = 0; i < samples; i++) fakeAudio(speaker)]);
  return EnrollmentService(repo).enroll(name, report);
}

/// Scripted language model. Each call to `send`/`sendToolResult` plays the
/// next entry of [script]; a String is streamed as words, an [LlmToolCall]
/// is emitted as a tool call. With an empty script it answers [reply], or
/// [responder]'s answer to the prompt.
class FakeLlm implements LlmEngine {
  String reply = 'Sam said the plumber comes at four.';
  final List<Object> script = [];
  final List<String> prompts = [];
  final List<Map<String, Object?>> toolResults = [];
  String? lastSystem;
  List<ToolSpec> lastTools = const [];
  int? lastMaxReply;
  bool loaded = false;
  bool unavailable = false;
  bool tools = true;

  /// Answers computed from the prompt (used when [script] is empty).
  String Function(String prompt)? responder;

  /// The next [failSends] sends fail before producing anything.
  int failSends = 0;

  /// Sends fail while tools are offered (like LiteRT-LM's code 13 bug).
  bool failWithTools = false;

  /// Contexts above this fail (simulates the phone's real limit).
  int? contextLimit;
  int loadedContext = 0;
  int recovers = 0;
  bool canRecover = true;

  String? get lastPrompt => prompts.isEmpty ? null : prompts.last;

  @override
  bool get isLoaded => loaded;

  @override
  bool get supportsTools => tools;

  @override
  Future<void> ensureLoaded({int? contextTokens}) async {
    if (unavailable) throw const LlmUnavailable('Gemma is not downloaded');
    loaded = true;
    loadedContext = contextTokens ?? 4096;
  }

  @override
  Future<LlmSession> openSession({
    required String system,
    List<ToolSpec> tools = const [],
    int? maxReplyTokens,
    int? contextTokens,
  }) async {
    if (unavailable) throw const LlmUnavailable('Gemma is not downloaded');
    if (contextTokens != null) loadedContext = contextTokens;
    lastSystem = system;
    lastTools = tools;
    lastMaxReply = maxReplyTokens;
    return _FakeSession(this, tools.isNotEmpty);
  }

  @override
  Future<bool> recover() async {
    recovers++;
    return canRecover && recovers == 1;
  }

  @override
  Future<void> unload() async => loaded = false;

  Stream<LlmEvent> _next(String prompt, {required bool withTools}) async* {
    if (failSends > 0) {
      failSends--;
      throw Exception('Failed to start streaming (code: 13)');
    }
    if (failWithTools && withTools) throw Exception('Failed to start streaming (code: 13)');
    if (contextLimit != null && loadedContext > contextLimit!) {
      throw Exception('DYNAMIC_UPDATE_SLICE failed to allocate');
    }
    final step = script.isNotEmpty ? script.removeAt(0) : (responder?.call(prompt) ?? reply);
    if (step is LlmToolCall) {
      yield step;
    } else {
      for (final word in '$step'.split(' ')) {
        yield LlmText('$word ');
      }
    }
  }
}

class _FakeSession implements LlmSession {
  _FakeSession(this.llm, this.withTools);
  final FakeLlm llm;
  final bool withTools;

  @override
  Stream<LlmEvent> send(String text) {
    llm.prompts.add(text);
    return llm._next(text, withTools: withTools);
  }

  @override
  Stream<LlmEvent> sendToolResult(String name, Map<String, Object?> result) {
    llm.toolResults.add({'name': name, ...result});
    return llm._next('', withTools: withTools);
  }

  @override
  Future<int?> countTokens(String text) async => (text.length / 4).ceil();

  @override
  Future<void> close() async {}
}

AgentToolbox toolboxFor(AppDatabase db, TranscriptRepository transcripts, SpeakerRepository speakers, {AgentHooks hooks = const AgentHooks(), DateTime Function()? clock}) =>
    AgentToolbox(
      transcripts: transcripts,
      speakers: speakers,
      notes: NoteRepository(db),
      reminders: ReminderRepository(db),
      rules: RuleRepository(db),
      hooks: hooks,
      clock: clock ?? DateTime.now,
    );

/// Builds VAD output starting [seconds] into the stream.
abstract final class SpeechChunkFake {
  static SpeechChunk at(double seconds, {int speaker = 1}) => SpeechChunk(fakeAudio(speaker), seconds);
}
