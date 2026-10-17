import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:narrator_studio/core/app_state.dart';
import 'package:narrator_studio/core/audio/exporter.dart';
import 'package:narrator_studio/core/audio/player.dart';
import 'package:narrator_studio/core/library/importer.dart';
import 'package:narrator_studio/core/library/library.dart';
import 'package:narrator_studio/core/library/settings.dart';
import 'package:narrator_studio/core/models/catalog.dart';
import 'package:narrator_studio/core/models/model_store.dart';
import 'package:narrator_studio/core/render/wav.dart';
import 'package:narrator_studio/core/samples.dart';
import 'package:narrator_studio/core/tts/vibevoice.dart';
import 'package:narrator_studio/core/tts/voice_mix.dart';
import 'package:narrator_studio/core/tts/voices.dart';
import 'package:path/path.dart' as p;

import 'helpers/fakes.dart';

void main() {
  group('catalog', () {
    test('every pack has files with sizes and checksums', () {
      for (final pack in voicePacks.values) {
        expect(pack.files, isNotEmpty, reason: pack.id);
        for (final f in pack.files) {
          expect(f.size, greaterThan(0));
          expect(f.sha256, hasLength(64));
        }
      }
      final kokoro = voicePacks['kokoro']!.files.map((f) => f.path).toSet();
      expect(kokoro, containsAll(['model.onnx', 'voices.bin', 'tokens.txt', 'lexicon-us-en.txt', 'espeak-ng-data']));
      expect(voicePacks['kokoro-lite']!.files.map((f) => f.path), contains('model.int8.onnx'));
      // Best quality: full-precision Supertonic and Pocket TTS (no int8 files), and Clarity.
      final supertonic = voicePacks['supertonic']!.files.map((f) => f.path).toSet();
      expect(supertonic, containsAll(['vocoder.onnx', 'vector_estimator.onnx', 'tts.json', 'voice.bin', 'unicode_indexer.bin']));
      expect(supertonic.where((f) => f.contains('int8')), isEmpty);
      final pocket = voicePacks['pocket']!.files.map((f) => f.path).toSet();
      expect(pocket, containsAll(['lm_main.onnx', 'lm_flow.onnx', 'decoder.onnx', 'encoder.onnx', 'text_conditioner.onnx']));
      expect(pocket.where((f) => f.contains('int8')), isEmpty);
      expect(voicePacks['clarity']!.files.map((f) => f.path), containsAll(['enhancer_backbone.onnx', 'enhancer_spec_head.onnx.data']));
      // The lifelike voices: everything the app's VibeVoice loop opens, and every voice.
      final vibe = voicePacks['vibevoice']!.files.map((f) => f.path).toSet();
      expect(vibe, containsAll(VibeVoice.files));
      for (final v in vibevoiceVoices) {
        expect(vibe, contains('voices/${v.id.split(':').last}.bin'), reason: v.id);
      }
    });

    test('voice tables line up with the models', () {
      expect(kokoroSpeakerNames, hasLength(54));
      expect(kokoroVoices.map((v) => v.sid).toSet(), hasLength(kokoroVoices.length));
      for (final slot in kokoroMixSlots) {
        expect(kokoroSpeakerNames, contains(slot));
      }
      expect(supertonicVoices, hasLength(10));
      for (final c in builtInCharacters) {
        // The natural voice is Kokoro (or a Gallery clip with a Kokoro stand-in).
        expect(c.voiceId, anyOf(startsWith('kokoro:'), startsWith('gallery:')), reason: c.id);
        if (!c.voiceId.startsWith('kokoro:')) expect(c.fallbackVoiceId, startsWith('kokoro:'), reason: c.id);
        if (c.studioVoiceId != null) expect(c.studioVoiceId, startsWith('supertonic:'), reason: c.id);
        if (c.lifelikeVoiceId != null) expect(c.lifelikeVoiceId, startsWith('vibevoice:'), reason: c.id);
        for (final id in [c.voiceId, ?c.fallbackVoiceId, ?c.studioVoiceId, ?c.lifelikeVoiceId]) {
          if (!id.startsWith('gallery:')) {
            expect([...kokoroVoices, ...supertonicVoices, ...vibevoiceVoices].any((v) => v.id == id), isTrue, reason: id);
          }
        }
      }
      // The main parts go to Kokoro's highest-graded voices.
      expect(builtInCharacters.firstWhere((c) => c.id == 'narrator').voiceId, 'kokoro:af_heart');
      expect(builtInCharacters.firstWhere((c) => c.id == 'cohost').voiceId, 'kokoro:af_bella');
    });
  });

  group('downloads', () {
    late HttpServer server;
    late Directory tmp;
    final payload = Uint8List.fromList(List.generate(300000, (i) => i * 7 % 251));
    var dropAt = 0;
    var requests = <String?>[];

    setUp(() async {
      tmp = await Directory.systemTemp.createTemp('dl');
      requests = [];
      dropAt = 0;
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((req) async {
        requests.add(req.headers.value('range'));
        if (req.uri.path == '/redirect') {
          req.response
            ..statusCode = 302
            ..headers.set('location', '/file');
          await req.response.close();
          return;
        }
        final range = req.headers.value('range');
        var start = 0;
        if (range != null) {
          start = int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
          req.response.statusCode = 206;
        }
        final end = dropAt > 0 && start == 0 ? dropAt : payload.length;
        req.response.contentLength = end - start;
        req.response.add(payload.sublist(start, end));
        await req.response.close();
      });
    });

    tearDown(() async => server.close(force: true));

    test('follows the redirect, keeps Range, resumes, verifies sha256', () async {
      final store = ModelStore(tmp);
      final target = File(p.join(tmp.path, 'out.bin'));
      // First attempt is cut short by the server: the retry resumes with Range.
      dropAt = 100000;
      await store.downloadFile(
        Uri.parse('http://127.0.0.1:${server.port}/redirect'),
        target,
        size: payload.length,
        sha256: sha256.convert(payload).toString(),
      );
      expect(await target.readAsBytes(), payload);
      expect(requests.whereType<String>().any((r) => r.startsWith('bytes=100000-')), isTrue);
    });

    test('a damaged file is rejected', () async {
      final store = ModelStore(tmp);
      final target = File(p.join(tmp.path, 'bad.bin'));
      await expectLater(
        store.downloadFile(
          Uri.parse('http://127.0.0.1:${server.port}/file'),
          target,
          size: payload.length,
          sha256: '0' * 64,
        ),
        throwsA(isA<StateError>()),
      );
      expect(target.existsSync(), isFalse);
    });
  });

  group('app state', () {
    test('casting: script voices, style defaults, rotation and NARRATOR', () async {
      final (app, _, _, _) = await makeApp();
      final doc = await app.library.add(title: 'x', kind: 'pasted', text: samplePodcast);
      final s = await app.openSession(doc);
      expect(s.style.id, 'podcast');
      expect(s.castRefs['HOST'], 'char:host');
      expect(s.castRefs['CO-HOST'], 'char:cohost');
      expect(s.castRefs['GUEST'], 'char:professor');
      expect(s.cast['GUEST']!.voice.id, 'kokoro:bm_george');
      expect(s.cast['HOST']!.voice.id, 'kokoro:am_michael');
      expect(s.cast['CO-HOST']!.voice.id, 'kokoro:af_bella');

      final doc2 = await app.library.add(
        title: 'y',
        kind: 'pasted',
        text: 'ALICE: Hi.\nBOB: Hello.\nALICE: Bye.\nBOB: Bye.\nNARRATOR: The end.',
      );
      final s2 = await app.openSession(doc2);
      expect(s2.castRefs['ALICE'], isNot(s2.castRefs['BOB']));
      expect(s2.castRefs['NARRATOR'], s2.castRefs[null]);
      expect(doc2.utterances, s2.plan.utterances.length);
    });

    test('names in the header match characters and voices', () async {
      final (app, _, _, _) = await makeApp(packs: ['kokoro', 'supertonic']);
      expect(app.matchRef('The Professor'), 'char:professor');
      expect(app.matchRef('professor'), 'char:professor');
      expect(app.matchRef('bm_george'), 'kokoro:bm_george');
      expect(app.matchRef('George'), 'kokoro:bm_george');
      expect(app.matchRef('Fiona'), 'supertonic:F1');
      expect(app.matchRef('nobody at all'), isNull);
    });

    test('natural voices by default, Studio with one switch, missing packs fall back', () async {
      final (both, _, _, _) = await makeApp(packs: ['kokoro', 'supertonic']);
      expect(both.settings.voiceSet, 'natural');
      expect(both.resolve('char:host').voice.id, 'kokoro:am_michael');
      expect(both.resolve('char:professor').voice.id, 'kokoro:bm_george');
      await both.settings.update((s) => s.voiceSet = 'studio');
      expect(both.resolve('char:host').voice.id, 'supertonic:M1');
      expect(both.resolve('char:professor').voice.id, 'supertonic:M3');
      expect(both.resolve('char:reporter').voice.id, 'kokoro:am_puck', reason: 'no Studio voice for this part');

      final (studioOnly, _, _, _) = await makeApp(packs: ['supertonic']);
      expect(studioOnly.resolve('char:professor').voice.id, 'supertonic:M3', reason: 'natural set, but only Studio downloaded');
      expect(studioOnly.resolve('char:reporter').voice.id, 'supertonic:F1', reason: 'a Kokoro-only character falls back');
      final (kokoroOnly, _, _, _) = await makeApp(packs: ['kokoro']);
      await kokoroOnly.settings.update((s) => s.voiceSet = 'studio');
      expect(kokoroOnly.resolve('char:host').voice.id, 'kokoro:am_michael', reason: 'Studio chosen, not downloaded');
      final (bare, _, _, _) = await makeApp(packs: []);
      expect(() => bare.resolve('char:narrator'), throwsA(isA<NoVoicesError>()));

      final (life, _, _, _) = await makeApp(packs: ['kokoro', 'vibevoice']);
      await life.settings.update((s) => s.voiceSet = 'lifelike');
      expect(life.resolve('char:host').voice.id, 'vibevoice:carter');
      expect(life.resolve('char:cohost').voice.id, 'vibevoice:grace');
      expect(life.resolve('char:narrator').voice.id, 'vibevoice:emma');
      expect(life.resolve('char:maestro').voice.id, 'kokoro:im_nicola', reason: 'no lifelike voice for this part');
      final (noPack, _, _, _) = await makeApp(packs: ['kokoro']);
      await noPack.settings.update((s) => s.voiceSet = 'lifelike');
      expect(noPack.resolve('char:host').voice.id, 'kokoro:am_michael', reason: 'Lifelike chosen, not downloaded');
    });

    test('gallery characters use their stand-in until the clip is downloaded', () async {
      final (app, _, _, _) = await makeApp(packs: ['kokoro', 'pocket']);
      expect(app.resolve('char:librarian').voice.id, 'kokoro:bf_isabella');
      final clip = File(app.models.path('gallery', 'voice--voice-zero-caro-davy.wav'))..createSync(recursive: true);
      clip.writeAsBytesSync(encodeWav(Float32List(24000), 24000));
      expect(app.resolve('char:librarian').voice.id, 'gallery:voice-zero-caro-davy');
    });

    test('settings and library survive a restart', () async {
      final (app, _, native, tmp) = await makeApp();
      await app.settings.update((s) {
        s.speed = 1.2;
        s.clones.add(ClonedVoice(id: 'c1', name: 'Me', file: 'c1.wav', created: DateTime(2026)));
      });
      final doc = await app.library.add(title: 'Book', kind: 'txt', text: 'Hello. World.');
      doc.position = 1;
      await app.library.changed(doc);
      final again = AppState(
        supportDir: app.supportDir,
        documentsDir: app.documentsDir,
        native: native,
        synth: (_) => FakeSynth(),
      );
      await again.init();
      expect(again.settings.speed, 1.2);
      expect(again.settings.clones.single.name, 'Me');
      expect(again.library.docs.single.position, 1);
      expect(await again.library.readText(again.library.docs.single), 'Hello. World.');
      expect(tmp.existsSync(), isTrue);
    });
  });

  group('voice mixes', () {
    test('blend is the weighted average, written into a spare slot', () async {
      final tmp = await Directory.systemTemp.createTemp('mix');
      final table = Float32List(kokoroVoiceFloats * kokoroSpeakerNames.length);
      for (var v = 0; v < kokoroSpeakerNames.length; v++) {
        for (var i = 0; i < kokoroVoiceFloats; i += 997) {
          table[v * kokoroVoiceFloats + i] = v.toDouble();
        }
      }
      final original = p.join(tmp.path, 'voices.bin');
      File(original).writeAsBytesSync(table.buffer.asUint8List());
      final mix = VoiceMix(id: 'm', name: 'Mix', parts: {'bm_george': 0.6, 'am_onyx': 0.4}, slot: 0);
      final out = await writeMixedVoices(original: original, out: p.join(tmp.path, 'mixed.bin'), mixes: [mix]);
      final bytes = File(out).readAsBytesSync();
      final mixed = bytes.buffer.asFloat32List();
      expect(mixed.length, table.length);
      final george = kokoroSpeakerNames.indexOf('bm_george');
      final onyx = kokoroSpeakerNames.indexOf('am_onyx');
      expect(mixed[mix.sid * kokoroVoiceFloats], closeTo(0.6 * george + 0.4 * onyx, 1e-4));
      expect(mixed[george * kokoroVoiceFloats], george.toDouble(), reason: 'other voices untouched');
      expect(mix.recipe, '60% George + 40% Onyx');
      expect(await writeMixedVoices(original: original, out: 'unused', mixes: []), original);
    });
  });

  group('player', () {
    test('plays through, follows sentences, saves the position, ends', () async {
      final (app, _, native, _) = await makeApp();
      final doc = await app.library.add(title: 'T', kind: 'txt', text: 'One. Two. Three. Four.');
      final s = await app.openSession(doc);
      native.timeScale = 3;
      final player = Player(app, native);
      final seen = <int>{};
      player.addListener(() => seen.add(player.current));
      await player.play(s);
      final done = Completer<void>();
      player.addListener(() {
        if (player.state == PlayState.ended && !done.isCompleted) done.complete();
      });
      await done.future.timeout(const Duration(seconds: 20));
      expect(seen, containsAll([0, 1, 2, 3]));
      expect(native.log, contains('open'));
      expect(doc.position, 0, reason: 'finished books start over');
      player.dispose();
    });

    test('pause, resume, jump and media buttons', () async {
      final (app, synth, native, _) = await makeApp();
      native.timeScale = 1; // real time, so it is still playing when we act
      final doc = await app.library.add(title: 'T', kind: 'txt', text: List.generate(20, (i) => 'Sentence number $i.').join(' '));
      final s = await app.openSession(doc);
      final player = Player(app, native);
      await player.play(s);
      await Future<void>.delayed(const Duration(milliseconds: 300));
      await player.pause();
      expect(player.state, PlayState.paused);
      expect(native.log, contains('pause'));
      final mark = synth.calls.length;
      await player.jump(10);
      expect(player.current, 10);
      expect(player.state, PlayState.paused, reason: 'jumping while paused stays paused');
      native.media.add('play');
      await Future<void>.delayed(const Duration(milliseconds: 300));
      expect(player.isPlaying, isTrue);
      expect(synth.calls.sublist(mark).map((c) => c.$2), contains('Sentence number 10.'));
      expect(synth.calls.sublist(mark).map((c) => c.$2), isNot(contains('Sentence number 2.')));
      native.media.add('stop');
      await Future<void>.delayed(const Duration(milliseconds: 300));
      expect(player.state, PlayState.idle);
      expect(doc.position, greaterThanOrEqualTo(10));
      player.dispose();
    });
  });

  group('chapters', () {
    test('the library learns the summary, topics and chapters when a script opens', () async {
      final (app, _, _, _) = await makeApp();
      final doc = await app.library.add(title: 'Sky', kind: 'pasted', text: samplePodcast);
      await app.openSession(doc);
      expect(doc.summary, startsWith('Why sunlight'));
      expect(doc.topics, ['science', 'physics', 'light']);
      expect(doc.chapters, 4);
      final again = Doc.fromJson(jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>);
      expect(again.topics, doc.topics);
      expect(again.chapters, 4);
    });

    test('headphone skip goes by chapter; back restarts the chapter first', () async {
      final (app, _, native, _) = await makeApp();
      native.timeScale = 1;
      final doc = await app.library.add(title: 'Sky', kind: 'pasted', text: samplePodcast);
      final s = await app.openSession(doc);
      final chapters = s.plan.chapters;
      final player = Player(app, native);
      await player.play(s);
      await player.pause();
      native.media.add('next');
      await Future<void>.delayed(const Duration(milliseconds: 200));
      expect(player.current, chapters[1].first);
      await player.jump(chapters[1].first + 3);
      native.media.add('previous');
      await Future<void>.delayed(const Duration(milliseconds: 200));
      expect(player.current, chapters[1].first, reason: 'back to the start of this chapter');
      native.media.add('previous');
      await Future<void>.delayed(const Duration(milliseconds: 200));
      expect(player.current, 0, reason: 'then the chapter before');
      expect(native.serviceText, contains('Chapter 1 of 4'));
      await player.stop();
      player.dispose();
    });

    test('an export records exactly where each chapter starts', () async {
      final (app, _, native, _) = await makeApp();
      final doc = await app.library.add(title: 'Sky', kind: 'pasted', text: samplePodcast);
      final s = await app.openSession(doc);
      final exporter = Exporter(app, native);
      await exporter.export(s);
      expect(exporter.state, ExportState.done);
      expect(exporter.chapters.map((c) => c.title), s.plan.chapters.map((c) => c.title));
      expect(exporter.chapters.first.start, Duration.zero);
      for (var i = 1; i < exporter.chapters.length; i++) {
        expect(exporter.chapters[i].start, greaterThan(exporter.chapters[i - 1].start));
      }
      expect(exporter.chapters.last.start, lessThan(exporter.length));
      expect(exporter.chaptersInFile, isFalse, reason: 'the fake encoder writes a WAV, not an MP4');
    });
  });

  group('export', () {
    test('renders the whole script to an m4a (via the native encoder)', () async {
      final (app, _, native, _) = await makeApp();
      final doc = await app.library.add(title: 'Sky: Episode 1?', kind: 'pasted', text: samplePodcast);
      final s = await app.openSession(doc);
      final exporter = Exporter(app, native);
      final out = await exporter.export(s);
      expect(exporter.state, ExportState.done, reason: exporter.error);
      expect(p.basename(out!), 'Sky Episode 1.m4a');
      final pcm = decodeWav(File(out).readAsBytesSync()); // the fake encoder copies the WAV
      expect(pcm.seconds, greaterThan(60));
      expect(native.log, containsAllInOrder(['screen:true', 'service:export', 'screen:false']));
      expect(Directory(app.cacheDir.path).listSync().whereType<File>().where((f) => f.path.endsWith('.wav')), isEmpty);
    });

    test('cancel stops early and leaves nothing behind', () async {
      final (app, synth, native, _) = await makeApp();
      synth.delay = const Duration(milliseconds: 20);
      final doc = await app.library.add(title: 'Long', kind: 'pasted', text: samplePodcast);
      final s = await app.openSession(doc);
      final exporter = Exporter(app, native);
      final f = exporter.export(s, format: 'wav');
      await Future<void>.delayed(const Duration(milliseconds: 150));
      exporter.cancel();
      expect(await f, isNull);
      expect(exporter.state, ExportState.cancelled);
      expect(app.exportsDir.existsSync() ? app.exportsDir.listSync() : const <FileSystemEntity>[], isEmpty);
    });
  });

  group('import', () {
    test('text encodings', () {
      expect(decodeText(utf8.encode('﻿héllo')), 'héllo');
      expect(decodeText([0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00]), 'hi');
      expect(decodeText([0x63, 0x61, 0x66, 0xE9]), 'café'); // Latin-1
    });

    test('txt and md files', () async {
      final tmp = await Directory.systemTemp.createTemp('imp');
      final f = File(p.join(tmp.path, 'my_great_notes.md'))..writeAsStringSync('# Notes\n\nHello.');
      final i = await importFile(f.path);
      expect(i.title, 'my great notes');
      expect(i.kind, 'md');
      expect(i.text, contains('Hello.'));
    });
  });
}
