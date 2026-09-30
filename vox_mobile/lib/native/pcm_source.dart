import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

/// A source of 16 kHz mono 16-bit PCM audio.
abstract interface class PcmSource {
  /// Starts capture. Throws if the microphone is unavailable.
  Future<Stream<Uint8List>> start();

  Future<void> stop();

  Future<void> dispose();
}

/// Microphone capture using Android's voice-recognition audio source, which
/// gives the cleanest signal for speech models.
class MicrophonePcmSource implements PcmSource {
  MicrophonePcmSource() : _recorder = AudioRecorder();

  final AudioRecorder _recorder;

  // No hasPermission() here: in the background service there is no Activity,
  // so it reports false even when granted. The UI checks before starting.
  @override
  Future<Stream<Uint8List>> start() async {
    return _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        echoCancel: false,
        noiseSuppress: false,
        autoGain: false,
        // Keep recording through calls/notifications rather than pausing.
        audioInterruption: AudioInterruptionMode.none,
        androidConfig: AndroidRecordConfig(audioSource: AndroidAudioSource.voiceRecognition),
      ),
    );
  }

  @override
  Future<void> stop() async {
    if (await _recorder.isRecording()) await _recorder.stop();
  }

  @override
  Future<void> dispose() => _recorder.dispose();
}
