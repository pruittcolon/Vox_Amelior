import 'dart:ffi';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/native/onnx_runtime.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';

/// NVIDIA Nemotron 3 Diarization (Sortformer family), converted to ONNX by
/// this repository's CI with NeMo's audio features built in: 16 kHz audio
/// in, speaker activity per short frame out (up to 8 speakers). Runs on the
/// ONNX Runtime that ships inside sherpa-onnx.
class SortformerDiarizer implements DiarizationEngine {
  SortformerDiarizer(String modelPath, {int threads = 2, String? libraryDir})
      : _ort = OnnxRuntime.load(libraryDir: libraryDir) {
    _session = _ort.createSession(modelPath, threads: threads);
  }

  final OnnxRuntime _ort;
  late final Pointer<Void> _session;
  bool _disposed = false;

  @override
  SpeakerActivity analyze(Float32List samples, int sampleRate) {
    if (_disposed) throw StateError('Diarizer was disposed');
    if (sampleRate != 16000) throw ArgumentError('Expected 16 kHz audio, got $sampleRate');
    final out = _ort.run(
      _session,
      inputs: {
        'audio': (data: samples, shape: [1, samples.length]),
        'length': (data: Int64List.fromList([samples.length]), shape: [1]),
      },
      output: 'preds',
    );
    // [1, frames, speakers]; the frame length follows from the audio length.
    final frames = out.shape[1];
    return SpeakerActivity(
      out.data,
      frames: frames,
      speakers: out.shape[2],
      frameSeconds: frames == 0 ? 0.01 : samples.length / sampleRate / frames,
    );
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _ort.releaseSession(_session);
  }
}
