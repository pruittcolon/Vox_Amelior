import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';
import 'package:vox_amelior_mobile/speakers/vector_math.dart';

import 'support/fakes.dart';

SpeakerProfile profile(String id, String name, int speaker) => SpeakerProfile(
      id: id,
      name: name,
      embeddingModel: 'fake',
      centroid: meanEmbedding([for (var i = 0; i < 5; i++) voiceprint(speaker, variant: i)]),
      sampleCount: 5,
      createdAt: DateTime(2026),
    );

SpeakerIdentifier build({List<SpeakerProfile>? profiles, IdentifierConfig? config}) {
  var n = 0;
  return SpeakerIdentifier(
    profiles: profiles ?? [profile('a', 'Alex', 1), profile('b', 'Sam', 2)],
    config: config ?? const IdentifierConfig(),
    newClusterId: () => 'c${++n}',
    nextGuestLabel: () => 'Guest $n',
  );
}

void main() {
  test('recognises each enrolled person from a fresh sample', () {
    final id = build();
    final alex = id.identify(voiceprint(1, variant: 99), seconds: 3);
    final sam = id.identify(voiceprint(2, variant: 98), seconds: 3);
    expect(alex.speakerId, 'a');
    expect(sam.speakerId, 'b');
    expect(alex.score, greaterThan(0.9));
  });

  test('very short utterances stay unattributed and create no guest', () {
    final id = build();
    final m = id.identify(voiceprint(1), seconds: 0.4);
    expect(m.speakerId, isNull);
    expect(m.cluster, isNull);
    expect(id.clusters, isEmpty);
  });

  test('an unknown voice becomes a guest and is recognised again', () {
    final id = build();
    final first = id.identify(voiceprint(7, variant: 1), seconds: 3);
    expect(first.cluster, isNotNull);
    expect(first.createdCluster, isTrue);
    final again = id.identify(voiceprint(7, variant: 2), seconds: 3);
    expect(again.cluster!.id, first.cluster!.id);
    expect(again.createdCluster, isFalse);
    expect(again.cluster!.count, 2);
    final other = id.identify(voiceprint(8, variant: 1), seconds: 3);
    expect(other.cluster!.id, isNot(first.cluster!.id));
  });

  test('ambiguous voices (two similar people) are not force-matched', () {
    // Both profiles share nearly the same centroid, so the margin rule must reject.
    final id = build(profiles: [profile('a', 'Alex', 1), profile('b', 'Twin', 1)]);
    final m = id.identify(voiceprint(1, variant: 50), seconds: 3);
    expect(m.speakerId, isNull);
  });

  test('threshold controls acceptance', () {
    final strict = build(config: const IdentifierConfig(threshold: 0.9999));
    expect(strict.identify(voiceprint(1, variant: 5), seconds: 3).speakerId, isNull);
  });

  test('guest count is capped', () {
    final id = build(profiles: const [], config: const IdentifierConfig(maxClusters: 2));
    for (var s = 10; s < 16; s++) {
      id.identify(voiceprint(s), seconds: 3);
    }
    expect(id.clusters.length, 2);
  });

  test('rank orders people by similarity', () {
    final id = build();
    final ranked = id.rank(voiceprint(2, variant: 3));
    expect(ranked.first.key.name, 'Sam');
    expect(ranked.first.value, greaterThan(ranked.last.value));
  });
}
