import 'dart:typed_data';

/// Turns speech audio into a fixed-size voiceprint vector.
/// Implemented natively (sherpa-onnx TitaNet); faked in tests.
abstract interface class EmbeddingEngine {
  /// Identifies the model. Voiceprints from different models don't mix.
  String get modelId;

  int get dimension;

  Float32List embed(Float32List samples, int sampleRate);

  void dispose();
}
