import 'dart:typed_data';

import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';

/// A stretch of continuous speech found by voice-activity detection.
class SpeechChunk {
  const SpeechChunk(this.samples, this.startSeconds);

  /// 16 kHz mono float samples in [-1, 1].
  final Float32List samples;

  /// Offset of the first sample from the start of the audio stream.
  final double startSeconds;

  double duration(int sampleRate) => samples.length / sampleRate;
}

/// Finds speech in a live audio stream (implemented with Silero VAD).
abstract interface class VadEngine {
  void accept(Float32List samples);

  /// Speech chunks completed since the last call.
  List<SpeechChunk> takeSegments();

  /// Ends the stream, returning any speech still in progress.
  List<SpeechChunk> flush();

  void reset();

  void dispose();
}

/// Speech-to-text (implemented with NVIDIA Parakeet via sherpa-onnx).
/// A word and when it starts, in seconds from the start of the audio.
class TimedWord {
  const TimedWord(this.text, this.start);

  final String text;
  final double start;
}

/// Text plus word start times (empty when the engine gives none).
class Transcript {
  const Transcript(this.text, [this.words = const []]);

  final String text;
  final List<TimedWord> words;
}

abstract interface class AsrEngine {
  String transcribe(Float32List samples, int sampleRate);

  /// Same text as [transcribe], with word start times when available.
  Transcript transcribeTimed(Float32List samples, int sampleRate);

  void dispose();
}

/// Who speaks when inside one stretch of audio (implemented with NVIDIA's
/// Sortformer-family diarizer). Used to split a line where the speaker
/// changes and to notice people talking at the same time.
abstract interface class DiarizationEngine {
  SpeakerActivity analyze(Float32List samples, int sampleRate);

  void dispose();
}
