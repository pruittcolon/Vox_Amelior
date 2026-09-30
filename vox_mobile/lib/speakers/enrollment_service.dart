import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

enum RejectReason { tooShort, tooQuiet, outlier }

class RejectedSample {
  const RejectedSample(this.index, this.reason);
  final int index;
  final RejectReason reason;
}

class EnrollmentReport {
  const EnrollmentReport({required this.embeddings, required this.rejected, required this.embeddingModel});

  /// Voiceprints of the samples that passed quality checks.
  final List<Float32List> embeddings;
  final List<RejectedSample> rejected;
  final String embeddingModel;
}

class EnrollmentException implements Exception {
  const EnrollmentException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Quality-checks voice recordings and turns them into voiceprints.
///
/// Pure computation with no database access, so it can run in a background
/// isolate. Outliers (another person talking, a TV, noise) are dropped.
class VoiceSampleAnalyzer {
  VoiceSampleAnalyzer(
    this._embedder, {
    this.sampleRate = 16000,
    this.minSampleSeconds = 2.0,
    this.minRms = 0.004,
    this.outlierSimilarity = 0.3,
  });

  final EmbeddingEngine _embedder;
  final int sampleRate;
  final double minSampleSeconds;
  final double minRms;

  /// A sample this dissimilar to the average of the others is discarded.
  final double outlierSimilarity;

  /// Splits one long recording into speech-sized samples, skipping silence.
  List<Float32List> splitRecording(Float32List audio, {double chunkSeconds = 6}) {
    final chunk = (chunkSeconds * sampleRate).round();
    final out = <Float32List>[];
    for (var start = 0; start < audio.length; start += chunk) {
      final end = math.min(start + chunk, audio.length);
      final part = Float32List.sublistView(audio, start, end);
      if (part.length >= (minSampleSeconds * sampleRate) && rms(part) >= minRms) {
        out.add(Float32List.fromList(part));
      }
    }
    return out;
  }

  EnrollmentReport analyze(List<Float32List> samples) {
    final rejected = <RejectedSample>[];
    final kept = <int>[];
    final embeddings = <Float32List>[];
    for (var i = 0; i < samples.length; i++) {
      final s = samples[i];
      if (s.length < minSampleSeconds * sampleRate) {
        rejected.add(RejectedSample(i, RejectReason.tooShort));
      } else if (rms(s) < minRms) {
        rejected.add(RejectedSample(i, RejectReason.tooQuiet));
      } else {
        kept.add(i);
        embeddings.add(_embedder.embed(s, sampleRate));
      }
    }

    if (embeddings.length < 4) {
      return EnrollmentReport(embeddings: embeddings, rejected: rejected, embeddingModel: _embedder.modelId);
    }
    final survivors = <Float32List>[];
    for (var i = 0; i < embeddings.length; i++) {
      final others = [
        for (var j = 0; j < embeddings.length; j++)
          if (j != i) embeddings[j],
      ];
      if (cosine(embeddings[i], meanEmbedding(others)) < outlierSimilarity) {
        rejected.add(RejectedSample(kept[i], RejectReason.outlier));
      } else {
        survivors.add(embeddings[i]);
      }
    }
    return EnrollmentReport(embeddings: survivors, rejected: rejected, embeddingModel: _embedder.modelId);
  }
}

/// Root-mean-square loudness of [a] (0 for empty input).
double rms(Float32List a) {
  if (a.isEmpty) return 0;
  var sum = 0.0;
  for (final x in a) {
    sum += x * x;
  }
  return math.sqrt(sum / a.length);
}

/// Saves analysed voice samples as people.
class EnrollmentService {
  EnrollmentService(this._speakers, {this.minAccepted = 3});

  final SpeakerRepository _speakers;
  final int minAccepted;

  /// Creates a new person. Throws [EnrollmentException] if too few samples
  /// were usable or the name is taken.
  SpeakerProfile enroll(String name, EnrollmentReport report) {
    if (report.embeddings.length < minAccepted) {
      throw EnrollmentException(
        'Only ${report.embeddings.length} usable sample(s); need at least $minAccepted. '
        'Speak for a few seconds each, in a quiet room.',
      );
    }
    try {
      return _speakers.create(name: name, embeddingModel: report.embeddingModel, samples: report.embeddings);
    } on StateError catch (e) {
      throw EnrollmentException(e.message);
    } on ArgumentError catch (e) {
      throw EnrollmentException('${e.message}');
    }
  }

  /// Improves an existing person's voice profile.
  void addSamples(String speakerId, EnrollmentReport report) {
    if (report.embeddings.isEmpty) {
      throw const EnrollmentException('None of those recordings were usable. Try again closer to the microphone.');
    }
    _speakers.addSamples(speakerId, report.embeddings);
  }
}
