import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/speakers/enrollment_service.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

import 'support/fakes.dart';

void main() {
  late AppDatabase db;
  late SpeakerRepository repo;
  late VoiceSampleAnalyzer analyzer;
  late EnrollmentService service;

  setUp(() {
    db = AppDatabase.inMemory();
    repo = SpeakerRepository(db);
    analyzer = VoiceSampleAnalyzer(FakeEmbedder());
    service = EnrollmentService(repo);
  });
  tearDown(() => db.close());

  EnrollmentReport samplesOf(int speaker, int n) => analyzer.analyze([for (var i = 0; i < n; i++) fakeAudio(speaker)]);

  test('enrolls a person from several samples and can then recognise them', () {
    final profile = service.enroll('Alex', samplesOf(1, 5));
    service.enroll('Sam', samplesOf(2, 5));
    expect(profile.name, 'Alex');
    expect(profile.sampleCount, 5);
    expect(profile.embeddingModel, 'fake');

    final id = SpeakerIdentifier(profiles: repo.profiles(), newClusterId: () => 'x', nextGuestLabel: () => 'Guest 1');
    expect(id.identify(voiceprint(1, variant: 200), seconds: 3).speakerId, profile.id);
    expect(id.identify(voiceprint(2, variant: 201), seconds: 3).speakerId, isNot(profile.id));
  });

  test('rejects too-short and too-quiet samples', () {
    final report = analyzer.analyze([
      fakeAudio(1, seconds: 0.5),
      fakeAudio(1, amplitude: 0.0001),
      fakeAudio(1),
    ]);
    expect(report.embeddings.length, 1);
    expect(report.rejected.map((r) => r.reason), [RejectReason.tooShort, RejectReason.tooQuiet]);
  });

  test('drops an outlier (someone else talking) when there are enough samples', () {
    final report = analyzer.analyze([for (var i = 0; i < 5; i++) fakeAudio(1), fakeAudio(9)]);
    expect(report.embeddings.length, 5);
    expect(report.rejected.single.reason, RejectReason.outlier);
    expect(report.rejected.single.index, 5);
  });

  test('refuses to enroll with too few usable samples', () {
    expect(() => service.enroll('Alex', samplesOf(1, 2)), throwsA(isA<EnrollmentException>()));
    expect(repo.profiles(), isEmpty);
  });

  test('duplicate names are rejected with a readable message', () {
    service.enroll('Alex', samplesOf(1, 3));
    expect(
      () => service.enroll('alex', samplesOf(2, 3)),
      throwsA(isA<EnrollmentException>().having((e) => e.message, 'message', contains('already exists'))),
    );
  });

  test('adding samples updates the sample count; unusable batches are refused', () {
    final p = service.enroll('Alex', samplesOf(1, 3));
    service.addSamples(p.id, samplesOf(1, 2));
    expect(repo.profile(p.id)!.sampleCount, 5);
    expect(() => service.addSamples(p.id, analyzer.analyze([fakeAudio(1, seconds: 0.2)])), throwsA(isA<EnrollmentException>()));
  });

  test('splitRecording chunks long audio and skips silence', () {
    final speech = fakeAudio(1, seconds: 14);
    final silence = Float32List(16000 * 6);
    final parts = analyzer.splitRecording(Float32List.fromList([...speech, ...silence]));
    expect(parts.length, 3); // 6s + 6s + 2s of speech; the silent 6s is dropped
    expect(parts.every((p) => p.length >= 2 * 16000), isTrue);
  });

  test('rms handles empty input', () {
    expect(rms(Float32List(0)), 0);
    expect(rms(Float32List.fromList([0.5, -0.5])), closeTo(0.5, 1e-9));
  });
}
