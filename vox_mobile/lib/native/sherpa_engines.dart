import 'dart:io';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';

bool _initialized = false;

/// Loads the sherpa-onnx native library. Must run once in every isolate
/// that uses the engines below (the UI isolate and the listening service).
///
/// [libraryDir] overrides where `libsherpa-onnx-c-api.so` is loaded from
/// (desktop tests); on Android the bundled library is found automatically.
void ensureSherpaInitialized({String? libraryDir}) {
  if (_initialized) return;
  sherpa.initBindings(libraryDir ?? _envLibraryDir());
  _initialized = true;
}

String? _envLibraryDir() {
  if (!Platform.isLinux) return null;
  final dir = Platform.environment['SHERPA_LIB_DIR'];
  return (dir == null || dir.isEmpty) ? null : dir;
}

const int kSampleRate = 16000;

/// Silero voice-activity detection.
class SherpaVad implements VadEngine {
  SherpaVad({
    required String modelPath,
    double threshold = 0.5,
    double minSilenceSeconds = 0.6,
    double minSpeechSeconds = 0.3,
    double maxSpeechSeconds = 20,
  }) {
    ensureSherpaInitialized();
    _vad = sherpa.VoiceActivityDetector(
      config: sherpa.VadModelConfig(
        sileroVad: sherpa.SileroVadModelConfig(
          model: modelPath,
          threshold: threshold,
          minSilenceDuration: minSilenceSeconds,
          minSpeechDuration: minSpeechSeconds,
          maxSpeechDuration: maxSpeechSeconds,
        ),
        sampleRate: kSampleRate,
        numThreads: 1,
        debug: false,
      ),
      bufferSizeInSeconds: 60,
    );
  }

  late final sherpa.VoiceActivityDetector _vad;

  @override
  void accept(Float32List samples) => _vad.acceptWaveform(samples);

  @override
  List<SpeechChunk> takeSegments() => _drain();

  @override
  List<SpeechChunk> flush() {
    _vad.flush();
    return _drain();
  }

  List<SpeechChunk> _drain() {
    final out = <SpeechChunk>[];
    while (!_vad.isEmpty()) {
      final segment = _vad.front();
      out.add(SpeechChunk(segment.samples, segment.start / kSampleRate));
      _vad.pop();
    }
    return out;
  }

  @override
  void reset() => _vad.reset();

  @override
  void dispose() => _vad.free();
}

/// NVIDIA Parakeet RNNT (NeMo transducer) speech recognition.
class SherpaParakeetAsr implements AsrEngine {
  SherpaParakeetAsr(SpeechModelPaths paths, {int threads = 2}) {
    ensureSherpaInitialized();
    _recognizer = sherpa.OfflineRecognizer(
      sherpa.OfflineRecognizerConfig(
        model: sherpa.OfflineModelConfig(
          transducer: sherpa.OfflineTransducerModelConfig(
            encoder: paths.encoder,
            decoder: paths.decoder,
            joiner: paths.joiner,
          ),
          tokens: paths.tokens,
          numThreads: threads,
          debug: false,
          modelType: 'nemo_transducer',
        ),
      ),
    );
  }

  late final sherpa.OfflineRecognizer _recognizer;

  @override
  String transcribe(Float32List samples, int sampleRate) {
    final stream = _recognizer.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      _recognizer.decode(stream);
      return _recognizer.getResult(stream).text;
    } finally {
      stream.free();
    }
  }

  @override
  void dispose() => _recognizer.free();
}

/// NVIDIA TitaNet voiceprints.
class SherpaSpeakerEmbedder implements EmbeddingEngine {
  SherpaSpeakerEmbedder(String modelPath, {int threads = 1}) {
    ensureSherpaInitialized();
    _extractor = sherpa.SpeakerEmbeddingExtractor(
      config: sherpa.SpeakerEmbeddingExtractorConfig(model: modelPath, numThreads: threads, debug: false),
    );
  }

  late final sherpa.SpeakerEmbeddingExtractor _extractor;

  static const String modelIdConst = 'nemo-titanet-small';

  @override
  String get modelId => modelIdConst;

  @override
  int get dimension => _extractor.dim;

  @override
  Float32List embed(Float32List samples, int sampleRate) {
    final stream = _extractor.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      stream.inputFinished();
      if (!_extractor.isReady(stream)) {
        throw StateError('Not enough audio to compute a voiceprint');
      }
      return _extractor.compute(stream);
    } finally {
      stream.free();
    }
  }

  @override
  void dispose() => _extractor.free();
}

/// Reads a WAV file as 16 kHz mono samples (resampling if needed).
Float32List readWavAs16k(String path) {
  ensureSherpaInitialized();
  final wave = sherpa.readWave(path);
  if (wave.samples.isEmpty) {
    throw const FormatException('Could not read this audio file. Use a 16-bit WAV file.');
  }
  return resampleLinear(wave.samples, wave.sampleRate, kSampleRate);
}

/// Linear-interpolation resampler (adequate for speech voiceprints).
Float32List resampleLinear(Float32List input, int fromRate, int toRate) {
  if (fromRate == toRate || input.isEmpty) return input;
  final outLength = (input.length * toRate / fromRate).floor();
  final out = Float32List(outLength);
  final ratio = fromRate / toRate;
  for (var i = 0; i < outLength; i++) {
    final pos = i * ratio;
    final i0 = pos.floor();
    final i1 = i0 + 1 < input.length ? i0 + 1 : i0;
    final frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}
