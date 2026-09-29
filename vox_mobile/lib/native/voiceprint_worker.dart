import 'dart:isolate';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/speakers/enrollment_service.dart';

/// Runs voiceprint extraction in a background isolate so the UI stays smooth.
class VoiceprintWorker {
  const VoiceprintWorker(this.speakerModelPath);

  final String speakerModelPath;

  /// Checks and embeds recorded samples (16 kHz mono).
  Future<EnrollmentReport> analyze(List<Float32List> samples) {
    final path = speakerModelPath;
    return Isolate.run(() {
      final embedder = SherpaSpeakerEmbedder(path);
      try {
        return VoiceSampleAnalyzer(embedder).analyze(samples);
      } finally {
        embedder.dispose();
      }
    });
  }

  /// Reads WAV files, splits long ones into ~6 s pieces, then analyses them.
  /// Returns the report and how many pieces were found.
  Future<(EnrollmentReport, int)> analyzeWavFiles(List<String> paths) {
    final model = speakerModelPath;
    return Isolate.run(() {
      final embedder = SherpaSpeakerEmbedder(model);
      try {
        final analyzer = VoiceSampleAnalyzer(embedder);
        final pieces = <Float32List>[];
        for (final path in paths) {
          final audio = readWavAs16k(path);
          final split = analyzer.splitRecording(audio);
          pieces.addAll(split.isEmpty && audio.isNotEmpty ? [audio] : split);
        }
        return (analyzer.analyze(pieces), pieces.length);
      } finally {
        embedder.dispose();
      }
    });
  }
}
