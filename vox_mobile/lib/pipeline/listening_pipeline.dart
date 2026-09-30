import 'dart:typed_data';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speech_capture.dart';
import 'package:vox_amelior_mobile/speakers/embedding_engine.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

/// Capture and processing wired together synchronously: audio in, saved
/// utterances out. Used for tests and tools; the always-on service instead
/// puts a disk queue and scheduler between the two halves.
class ListeningPipeline {
  ListeningPipeline({
    required VadEngine vad,
    required AsrEngine asr,
    required EmbeddingEngine embedder,
    required SpeakerIdentifier identifier,
    required TranscriptRepository transcripts,
    required SpeakerRepository speakers,
    Clock clock = systemClock,
    ProcessorConfig config = const ProcessorConfig(),
    DiarizationEngine? diarizer,
    this.onSegment,
  })  : capture = SpeechCapture(vad, clock: clock, sampleRate: config.sampleRate),
        processor = SegmentProcessor(
          asr: asr,
          embedder: embedder,
          identifier: identifier,
          transcripts: transcripts,
          speakers: speakers,
          config: config,
          diarizer: diarizer,
        );

  final SpeechCapture capture;
  final SegmentProcessor processor;
  final void Function(SegmentView segment)? onSegment;

  SpeakerIdentifier get identifier => processor.identifier;
  ProcessorStats get stats => processor.stats;

  List<SegmentView> addPcm16(Uint8List bytes) => _process(capture.addPcm16(bytes));

  List<SegmentView> addSamples(Float32List samples) => _process(capture.addSamples(samples));

  List<SegmentView> flush() => _process(capture.flush());

  void dispose() {
    capture.dispose();
    processor.dispose();
  }

  List<SegmentView> _process(List<CapturedSpeech> chunks) {
    final saved = <SegmentView>[];
    for (final c in chunks) {
      try {
        for (final s in processor.process(c.samples, c.startedAt)) {
          saved.add(s);
          onSegment?.call(s);
        }
      } on Object catch (e, st) {
        processor.stats.errors++;
        Log.e('pipeline', 'failed to process speech chunk', e, st);
      }
    }
    return saved;
  }
}
