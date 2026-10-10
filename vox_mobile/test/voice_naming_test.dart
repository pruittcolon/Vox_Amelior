import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

import 'support/fakes.dart';

/// Voices without a name: what each said, naming all of it at once, and
/// taking a naming back.
void main() {
  late AppDatabase db;
  late TranscriptRepository transcripts;
  late SpeakerRepository speakers;
  late String pruitt;
  final t0 = DateTime(2026, 10, 1, 19);

  setUp(() {
    db = AppDatabase.inMemory();
    transcripts = TranscriptRepository(db);
    speakers = SpeakerRepository(db);
    pruitt = speakers.create(name: 'Pruitt', embeddingModel: 'm', samples: [voiceprint(1)]).id;
  });
  tearDown(() => db.close());

  String guest(String label, int speaker, {bool background = false}) {
    final c = UnknownCluster(id: SpeakerRepository.newId(), label: label, centroid: voiceprint(speaker), count: 1, updatedAt: t0);
    speakers.saveCluster(c);
    if (background) speakers.setClusterBackground(c.id, true);
    return c.id;
  }

  SegmentView say(String text, int minute, {String? who, String? voice, int speaker = 2}) => transcripts.addSegment(
        text: text,
        startedAt: t0.add(Duration(minutes: minute)),
        duration: const Duration(seconds: 3),
        speakerId: who,
        clusterId: voice,
        embedding: voiceprint(speaker, variant: minute),
      );

  List<SegmentView> linesOf(String clusterOrSpeaker) =>
      transcripts.recent(limit: 100).where((l) => l.clusterId == clusterOrSpeaker || l.speakerId == clusterOrSpeaker).toList();

  test('voices to name: most lines first, with their clearest lines and who they were with; TV left out', () {
    final g1 = guest('Guest 1', 2);
    final g2 = guest('Guest 2', 3);
    final tv = guest('Guest 3', 4, background: true);
    say('Hi there', 0, who: pruitt, speaker: 1);
    say('Yes', 1, voice: g1);
    say('We cannot pay the electric bill this month at all', 2, voice: g1);
    say('Rent is due', 3, voice: g1);
    say('Breaking news tonight from the capital', 4, voice: tv, speaker: 4);
    say('Hello, is anyone home?', 90, voice: g2, speaker: 3);

    final voices = transcripts.voicesToName();
    expect(voices.map((v) => v.label), ['Guest 1', 'Guest 2']);
    expect(voices.first.lines, 3);
    expect(voices.first.samples.first.text, 'We cannot pay the electric bill this month at all', reason: 'the clearest line first');
    expect(voices.first.heardWith, ['Pruitt']);
    expect(voices.last.heardWith, isEmpty, reason: 'alone in its conversation');
    expect(voices.first.lastHeard, t0.add(const Duration(minutes: 3)));
    expect(transcripts.voicesToNameCount(), 2);
    expect(transcripts.voiceToName(tv)?.lines, 1, reason: 'a TV voice can still be looked up');
    expect(transcripts.voicesToName(detailed: 0).first.samples, isEmpty, reason: 'details only when asked for');
  });

  test('naming a voice names everything it said; undo puts every line back', () {
    final g1 = guest('Guest 1', 2);
    for (var i = 0; i < 3; i++) {
      say('Line $i from the guest', i, voice: g1);
    }
    final before = speakers.sampleCount(pruitt);

    final naming = speakers.nameVoice(g1, pruitt);
    expect(linesOf(pruitt), hasLength(3));
    expect(transcripts.voicesToName(), isEmpty);
    expect(speakers.clusters(), isEmpty);
    expect(speakers.sampleCount(pruitt), before + 3, reason: 'Pruitt learns from every line');
    expect(naming.voiceLabel, 'Guest 1');

    speakers.undoNaming(naming);
    expect(linesOf(g1), hasLength(3));
    expect(linesOf(g1).every((l) => l.speakerId == null && l.speakerLabel == 'Guest 1'), isTrue);
    expect(speakers.sampleCount(pruitt), before, reason: 'and forgets them again');
    expect(transcripts.voicesToName().single.lines, 3);
  });

  test('a line corrected after naming stays corrected when the naming is undone', () {
    final g1 = guest('Guest 1', 2);
    final lines = [for (var i = 0; i < 3; i++) say('Line $i from the guest', i, voice: g1)];
    final ericah = speakers.create(name: 'Ericah', embeddingModel: 'm', samples: [voiceprint(5)]).id;

    final naming = speakers.nameVoice(g1, pruitt);
    speakers.assignSegmentToSpeaker(lines[1].id, ericah);
    speakers.undoNaming(naming);

    expect(transcripts.segment(lines[1].id)!.speakerId, ericah);
    expect(linesOf(g1).map((l) => l.id), [lines[2].id, lines[0].id]);
  });

  test('naming a voice as someone new, and taking it back', () {
    final g1 = guest('Guest 1', 2);
    say('Line one from the guest', 0, voice: g1);
    say('Line two from the guest', 1, voice: g1);

    final naming = speakers.nameVoiceAsNew(g1, 'Grandma', embeddingModel: 'm');
    final grandma = speakers.profiles().singleWhere((p) => p.name == 'Grandma');
    expect(linesOf(grandma.id), hasLength(2));
    expect(naming.newPerson, isTrue);

    speakers.undoNaming(naming);
    expect(speakers.profiles().map((p) => p.name), ['Pruitt']);
    expect(linesOf(g1), hasLength(2));
  });

  test('a voice that is gone cannot be named', () {
    expect(() => speakers.nameVoice('missing', pruitt), throwsStateError);
  });
}
