// Runs on an Android emulator in CI (narrator-device.yml): the real Kotlin
// side (speaker streaming, background service, AAC encoder, decoder) and the
// real voices, end to end. The host serves the voice files at 10.0.2.2:8765
// and takes a screenshot whenever this test logs "TOUR_SHOT <name>".
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:narrator_studio/core/app_state.dart';
import 'package:narrator_studio/core/audio/exporter.dart';
import 'package:narrator_studio/core/audio/native.dart';
import 'package:narrator_studio/core/audio/player.dart';
import 'package:narrator_studio/core/models/model_store.dart';
import 'package:narrator_studio/core/render/audio.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/tts/sherpa_synth.dart';
import 'package:narrator_studio/core/tts/voices.dart';
import 'package:narrator_studio/main.dart';
import 'package:narrator_studio/ui/preview.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

const host = 'http://10.0.2.2:8765';

Future<void> shot(WidgetTester tester, String name) async {
  await tester.pump(const Duration(milliseconds: 300));
  debugPrint('TOUR_SHOT $name');
  await Future<void>.delayed(const Duration(milliseconds: 1800));
}

void log(String s) => debugPrint('DEVICE $s');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('real voices on Android: download, speak, play, export', (tester) async {
    final support = await getApplicationSupportDirectory();
    final documents = await getApplicationDocumentsDirectory();
    final native = AndroidNative();
    final models = ModelStore(Directory(p.join(support.path, 'models')), baseUrl: host);
    late final AppState app;
    app = AppState(
      supportDir: support,
      documentsDir: documents,
      native: native,
      models: models,
      synth: (a) => SherpaSynth(
        () => EngineSetup(
          kokoroDir: a.models.packDir(a.kokoroPack).path,
          supertonicDir: a.models.packDir('supertonic').path,
          pocketDir: a.models.packDir('pocket').path,
          kokoroVoicesFile: a.kokoroVoicesFile,
          threads: a.settings.threads,
        ),
      ),
    );
    await app.init();
    final player = Player(app, native);
    final exporter = Exporter(app, native);
    final preview = VoicePreview(app, native, player);

    await tester.pumpWidget(NarratorApp(app: app, player: player, exporter: exporter, native: native, preview: preview));
    await tester.pump(const Duration(seconds: 1));
    await shot(tester, '01_first_launch');

    // 1. Downloads through the app's own downloader (sha256-checked).
    for (final pack in ['kokoro', 'pocket']) {
      final watch = Stopwatch()..start();
      await models.install(pack);
      final st = models.status(pack);
      log('install $pack: ${st.state} in ${watch.elapsed.inSeconds}s ${st.error ?? ''}');
      expect(st.state, PackState.installed, reason: st.error);
    }
    await tester.pump(const Duration(seconds: 1));
    await shot(tester, '02_voices_ready');

    // 2. Every engine speaks on the device; measure the speed.
    final heart = kokoroVoices.firstWhere((v) => v.id == 'kokoro:af_heart');
    for (final threads in [2, 4]) {
      app.settings.threads = threads;
      await app.synth.speak(heart, 'Warm up.');
      final watch = Stopwatch()..start();
      final pcm = await app.synth.speak(heart, 'Light from the sun looks white, but it is really a mixture of every colour.');
      final secs = pcm.length / outputRate;
      log('kokoro threads=$threads: ${secs.toStringAsFixed(2)}s audio in ${watch.elapsedMilliseconds}ms '
          '(${(secs / (watch.elapsedMilliseconds / 1000)).toStringAsFixed(2)}x real time)');
      expect(secs, greaterThan(2));
    }
    final bria = Voice(
      id: 'test:bria',
      name: 'Bria',
      engine: Engine.pocket,
      blurb: '',
      refPath: models.path('pocket', 'test_wavs/bria.wav'),
    );
    final watch = Stopwatch()..start();
    final cloned = await app.synth.speak(bria, 'This voice was cloned from a short sample clip.');
    log('pocket clone: ${(cloned.length / outputRate).toStringAsFixed(2)}s audio in ${watch.elapsedMilliseconds}ms');
    expect(cloned.length / outputRate, greaterThan(1.5));

    // 3. The sample podcast in the reader, playing through the real speaker.
    final doc = await app.library.add(title: 'Why Is the Sky Blue?', kind: 'sample', text: samplePodcast);
    final session = await app.openSession(doc);
    await player.play(session);
    await tester.pumpWidget(NarratorApp(
      key: UniqueKey(),
      app: app,
      player: player,
      exporter: exporter,
      native: native,
      preview: preview,
    ));
    await tester.pump(const Duration(seconds: 1));
    // Open the reader from the mini player.
    final mini = find.text('Why Is the Sky Blue?');
    if (mini.evaluate().isNotEmpty) {
      await tester.tap(mini.first);
    }
    final positions = <int>[];
    for (var i = 0; i < 24; i++) {
      await tester.pump(const Duration(milliseconds: 500));
      positions.add(await native.playerPosition());
    }
    log('player: state=${player.state} sentence=${player.current} positions=${positions.first}..${positions.last} '
        'render speed=${player.speedRatio.toStringAsFixed(2)}x');
    await shot(tester, '03_reader_playing');
    expect(positions.last, greaterThan(outputRate * 3), reason: 'the speaker played at least 3 s');
    expect(player.current, greaterThan(0), reason: 'the highlight followed the speech');
    await player.pause();
    final paused = await native.playerPosition();
    await Future<void>.delayed(const Duration(seconds: 1));
    expect(await native.playerPosition(), paused, reason: 'pause stops the speaker');
    await player.resume();
    await Future<void>.delayed(const Duration(seconds: 1));
    expect(await native.playerPosition(), greaterThan(paused));
    await player.stop();

    // 4. Export the whole episode to M4A, then decode it back.
    final exportWatch = Stopwatch()..start();
    final out = await exporter.export(session);
    log('export: ${exporter.state} in ${exportWatch.elapsed.inSeconds}s -> $out ${exporter.error ?? ''}');
    expect(exporter.state, ExportState.done, reason: exporter.error);
    final size = File(out!).lengthSync();
    final (pcm, rate) = await native.decodeAudio(out, maxSeconds: 400);
    log('m4a: $size bytes, decoded ${(pcm.length / rate).toStringAsFixed(1)}s at $rate Hz; '
        'expected ${exporter.length.inSeconds}s');
    expect((pcm.length / rate - exporter.length.inMilliseconds / 1000).abs(), lessThan(1.5));
    expect(size, greaterThan(200 * 1024));
    final saved = await native.saveToDownloads(out, 'narrator-test.m4a', 'audio/mp4');
    log('saved to $saved');

    // 5. Voice preview from the Voices tab.
    await preview.play(heart);
    expect(preview.error, isNull);
    await Future<void>.delayed(const Duration(seconds: 3));
    await tester.pump(const Duration(seconds: 1));
    await shot(tester, '04_after_export');

    log('ALL DONE');
  }, timeout: const Timeout(Duration(minutes: 40)));
}
