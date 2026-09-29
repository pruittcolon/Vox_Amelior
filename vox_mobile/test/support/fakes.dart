import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
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

class FakeLlm implements LlmEngine {
  String reply = 'Sam said the plumber comes at four.';
  String? lastSystem;
  String? lastPrompt;
  bool loaded = false;
  bool unavailable = false;

  @override
  bool get isLoaded => loaded;

  @override
  Future<void> ensureLoaded() async {
    if (unavailable) throw const LlmUnavailable('Gemma is not downloaded');
    loaded = true;
  }

  @override
  Stream<String> generate({required String system, required String prompt}) async* {
    lastSystem = system;
    lastPrompt = prompt;
    for (final word in reply.split(' ')) {
      yield '$word ';
    }
  }

  @override
  Future<void> unload() async => loaded = false;
}

