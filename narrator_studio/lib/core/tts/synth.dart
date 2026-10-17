/// The speech engine as the rest of the app sees it.
library;

import 'dart:typed_data';

import 'voices.dart';

/// Turns text into audio at [outputRate] (see render/audio.dart).
abstract class Synth {
  Future<Float32List> speak(Voice voice, String text, {double speed = 1.0});

  /// Drops loaded models (memory) without forgetting settings.
  Future<void> unload() async {}

  /// Kokoro's voice table changed (voice mixes); reload before next use.
  Future<void> reloadKokoroVoices() async {}
}

/// Who speaks a sentence, and how fast.
class VoiceSpec {
  const VoiceSpec(this.voice, this.speed);
  final Voice voice;
  final double speed;
}
