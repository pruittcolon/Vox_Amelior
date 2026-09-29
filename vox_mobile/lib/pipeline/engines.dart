import 'dart:typed_data';

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
abstract interface class AsrEngine {
  String transcribe(Float32List samples, int sampleRate);

  void dispose();
}
