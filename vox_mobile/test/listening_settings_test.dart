import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';
import 'package:vox_amelior_mobile/pipeline/speech_capture.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

/// Records the loudest sample it was given.
class _PeakVad implements VadEngine {
  double peak = 0;
  int flushes = 0;
  bool disposed = false;
  final List<SpeechChunk> onFlush = [];

  @override
  void accept(Float32List samples) {
    for (final v in samples) {
      if (v.abs() > peak) peak = v.abs();
    }
  }

  @override
  List<SpeechChunk> takeSegments() => const [];

  @override
  List<SpeechChunk> flush() {
    flushes++;
    return List.of(onFlush);
  }

  @override
  void reset() {}

  @override
  void dispose() => disposed = true;
}

Float32List constant(double v, int n) => Float32List.fromList(List.filled(n, v));

void main() {
  group('microphone boost', () {
    test('the default is +15%', () {
      expect(const AppSettings().micGain, 1.15);
    });

    test('boosts what the detector and transcriber hear, and never past full scale', () {
      final vad = _PeakVad();
      final capture = SpeechCapture(vad, gain: 1.15);
      capture.addSamples(constant(0.1, 1600));
      expect(vad.peak, closeTo(0.115, 1e-6));
      capture.addSamples(constant(0.9, 1600));
      expect(vad.peak, 1.0, reason: 'clamped, not wrapped');
      expect(capture.level.clipping, isTrue);
    });

    test('no boost leaves the audio untouched; a changed gain applies to the next audio', () {
      final vad = _PeakVad();
      final capture = SpeechCapture(vad);
      capture.addSamples(constant(0.2, 1600));
      expect(vad.peak, closeTo(0.2, 1e-6));
      capture.gain = 2.0;
      capture.addSamples(constant(0.3, 1600));
      expect(vad.peak, closeTo(0.6, 1e-6));
    });

    test('16-bit microphone data is boosted too', () {
      final vad = _PeakVad();
      final capture = SpeechCapture(vad, gain: 2.0);
      final pcm = ByteData(2)..setInt16(0, 8192, Endian.little); // 0.25
      capture.addPcm16(pcm.buffer.asUint8List());
      expect(vad.peak, closeTo(0.5, 1e-4));
    });

    test('level meter: -60 dB is empty, full scale is full, -20 dB is two thirds', () {
      expect(AudioLevel.meter(0), 0);
      expect(AudioLevel.meter(0.0005), 0);
      expect(AudioLevel.meter(1), 1);
      expect(AudioLevel.meter(0.1), closeTo(2 / 3, 1e-9));
      final l = AudioLevel.of(Float32List.fromList([0.5, -0.5, 0.5, -0.5]));
      expect(l.rms, closeTo(0.5, 1e-9));
      expect(l.peak, 0.5);
      expect(l.clipping, isFalse);
      expect(AudioLevel.of(Float32List(0)).rms, 0);
    });

    test('swapping the speech detector hands back what the old one held and re-anchors time', () {
      final old = _PeakVad()..onFlush.add(SpeechChunk(constant(0.1, 1600), 0));
      final next = _PeakVad();
      final now = DateTime(2026, 9, 30, 12);
      final capture = SpeechCapture(old, clock: () => now);
      capture.addSamples(constant(0.1, 1600));
      final held = capture.swapVad(next);
      expect(held, hasLength(1));
      expect(old.disposed, isTrue);
      capture.addSamples(constant(0.4, 1600));
      expect(next.peak, closeTo(0.4, 1e-6));
      expect(old.peak, closeTo(0.1, 1e-6), reason: 'new audio goes to the new detector');
    });
  });

  group('listening settings', () {
    test('new settings survive a round trip', () {
      const s = AppSettings(
        micGain: 2.5,
        vadThreshold: 0.35,
        pauseSeconds: 0.9,
        minSpeechSeconds: 0.5,
        speechModel: 'fp16',
        splitMinSeconds: 2.0,
        splitMinWords: 3,
      );
      final back = AppSettings.fromJson(s.toJson());
      expect(back.micGain, 2.5);
      expect(back.vadThreshold, 0.35);
      expect(back.pauseSeconds, 0.9);
      expect(back.minSpeechSeconds, 0.5);
      expect(back.speechModel, 'fp16');
      expect(back.asrAsset.id, 'parakeet-tdt-0.6b-v2-fp16');
      expect(back.splitMinSeconds, 2.0);
      expect(back.splitMinWords, 3);
    });

    test('settings saved by an older version get the new defaults (including +15%)', () {
      final old = const AppSettings().toJson()
        ..remove('micGain')
        ..remove('pauseSeconds')
        ..remove('minSpeechSeconds')
        ..remove('speechModel')
        ..remove('splitMinSeconds')
        ..remove('splitMinWords');
      final s = AppSettings.fromJson(old);
      expect(s.micGain, 1.15);
      expect(s.pauseSeconds, 0.6);
      expect(s.minSpeechSeconds, 0.3);
      expect(s.speechModel, 'int8');
      expect(s.asrAsset.id, 'parakeet-tdt-0.6b-v2-int8');
      expect(s.splitMinSeconds, 1.5);
      expect(s.splitMinWords, 2);
    });

    test('out-of-range and wrong-type values are pulled back into range', () {
      final s = AppSettings.fromJson({
        'micGain': 99,
        'pauseSeconds': 0,
        'minSpeechSeconds': 'loud',
        'speechModel': 'fp64',
        'splitMinSeconds': -3,
        'splitMinWords': 40,
      });
      expect(s.micGain, 4.0);
      expect(s.pauseSeconds, 0.3);
      expect(s.minSpeechSeconds, 0.3);
      expect(s.speechModel, 'int8');
      expect(s.splitMinSeconds, 0.8);
      expect(s.splitMinWords, 2);
    });
  });

  group('speech model choice', () {
    late Directory tmp;
    late ModelStore store;
    setUp(() {
      tmp = Directory.systemTemp.createTempSync('vox_asr_');
      store = ModelStore(Directory(p.join(tmp.path, 'models')));
    });
    tearDown(() => tmp.deleteSync(recursive: true));

    void install(ModelAsset a) {
      store.dir(a).createSync(recursive: true);
      for (final n in a.installedFileNames) {
        store.file(a, n).writeAsBytesSync([1, 2, 3]);
      }
      store.markInstalled(a);
    }

    test('fp16 is an optional download with its own pinned files', () {
      final a = ModelCatalog.parakeetFp16;
      expect(a.essential, isFalse);
      expect(a.installedFileNames, {'encoder.fp16.onnx', 'decoder.fp16.onnx', 'joiner.fp16.onnx', 'tokens.txt'});
      expect(a.files.single.sha256, hasLength(64));
      expect(a.files.single.sizeBytes, 1120982957);
      expect(ModelCatalog.all, contains(a), reason: 'kept on disk by the upgrade cleanup');
    });

    test('the fp16 files are used once installed; until then the standard model keeps working', () {
      for (final a in ModelCatalog.speech) {
        install(a);
      }
      final before = SpeechModelPaths.fromStore(store, asr: ModelCatalog.parakeetFp16)!;
      expect(p.basename(before.encoder), 'encoder.int8.onnx', reason: 'fp16 not downloaded yet');
      install(ModelCatalog.parakeetFp16);
      final after = SpeechModelPaths.fromStore(store, asr: ModelCatalog.parakeetFp16)!;
      expect([after.encoder, after.decoder, after.joiner, after.tokens].map(p.basename),
          ['encoder.fp16.onnx', 'decoder.fp16.onnx', 'joiner.fp16.onnx', 'tokens.txt']);
      expect(p.basename(SpeechModelPaths.fromStore(store)!.encoder), 'encoder.int8.onnx', reason: 'default stays int8');
    });

    test('the fp16 folder survives the upgrade cleanup', () {
      install(ModelCatalog.parakeetFp16);
      store.removeExcept({for (final m in ModelCatalog.all) m.id, ModelCatalog.customLlmId});
      expect(store.isInstalled(ModelCatalog.parakeetFp16), isTrue);
    });
  });

  group('speaker-split caution from settings', () {
    late AppDatabase db;
    late SpeakerRepository speakers;
    late TranscriptRepository transcripts;
    setUp(() {
      db = AppDatabase.inMemory();
      speakers = SpeakerRepository(db);
      transcripts = TranscriptRepository(db);
      enrollFake(speakers, 'Alex', 1);
      enrollFake(speakers, 'Sam', 2);
    });
    tearDown(() => db.close());

    SegmentProcessor make(ProcessorConfig config) => SegmentProcessor(
          asr: FakeAsr(['one two three four']),
          embedder: FakeEmbedder(),
          identifier: SpeakerIdentifier(
            profiles: speakers.profiles(),
            newClusterId: SpeakerRepository.newId,
            nextGuestLabel: speakers.nextGuestLabel,
          ),
          transcripts: transcripts,
          speakers: speakers,
          diarizer: _Diarizer(),
          config: config,
        );

    test('a one-word part cuts only when the minimum words is lowered to 1', () {
      final audio = Float32List.fromList([...fakeAudio(1, seconds: 4.4), ...fakeAudio(2, seconds: 1.6)]);
      final t0 = DateTime(2026, 9, 30, 20);
      expect(make(const ProcessorConfig()).process(audio, t0).map((s) => s.text), ['one two three four']);
      final eager = make(const ProcessorConfig(minPartWords: 1)).process(audio, t0.add(const Duration(minutes: 5)));
      expect(eager.map((s) => s.text), ['one two three', 'four']);
    });

    test('the shortest part can be tuned', () {
      final activity = _activity([(0, 0, 3), (1, 3, 6)]);
      expect(const SpeakerTurns(minTurnSeconds: 2.5).turns(activity, totalSeconds: 6), hasLength(2), reason: '3 s halves pass a 2.5 s minimum');
      expect(const SpeakerTurns(minTurnSeconds: 3.5).turns(activity, totalSeconds: 6), hasLength(1), reason: 'but not a 3.5 s one');
    });
  });
}

SpeakerActivity _activity(List<(int, double, double)> spans, {double fs = 0.01, int speakers = 4, double seconds = 6}) {
  final frames = (seconds / fs).round();
  final probs = Float32List(frames * speakers);
  for (final (s, from, to) in spans) {
    for (var t = (from / fs).round(); t < (to / fs).round() && t < frames; t++) {
      probs[t * speakers + s] = 0.9;
    }
  }
  return SpeakerActivity(probs, frames: frames, speakers: speakers, frameSeconds: fs);
}

class _Diarizer implements DiarizationEngine {
  @override
  SpeakerActivity analyze(Float32List samples, int sampleRate) => _activity([(0, 0, 4.4), (1, 4.4, 6)]);

  @override
  void dispose() {}
}
