import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:narrator_studio/core/app_state.dart';
import 'package:narrator_studio/core/audio/native.dart';
import 'package:narrator_studio/core/models/catalog.dart';
import 'package:narrator_studio/core/models/model_store.dart';
import 'package:narrator_studio/core/render/audio.dart';
import 'package:narrator_studio/core/tts/synth.dart';
import 'package:narrator_studio/core/tts/voices.dart';
import 'package:path/path.dart' as p;

/// Speaks a tone whose length follows the text (about 15 characters/second).
class FakeSynth implements Synth {
  final calls = <(String voiceId, String text, double speed)>[];
  Duration delay = Duration.zero;
  Object? failWith;

  @override
  Future<Float32List> speak(Voice voice, String text, {double speed = 1.0}) async {
    calls.add((voice.id, text, speed));
    if (delay > Duration.zero) await Future<void>.delayed(delay);
    if (failWith != null) throw failWith!;
    final seconds = math.max(0.3, text.length / 15 / speed);
    final n = (seconds * outputRate).round();
    final out = Float32List(n + outputRate ~/ 10); // trailing silence, trimmed later
    final f = 180.0 + (voice.sid % 7) * 20;
    for (var i = 0; i < n; i++) {
      out[i] = 0.3 * math.sin(2 * math.pi * f * i / outputRate);
    }
    return out;
  }

  @override
  Future<void> unload() async {}

  @override
  Future<void> reloadKokoroVoices() async {}
}

/// A speaker that "plays" in real time from when audio is written.
class FakeNative implements NativeApi {
  final media = StreamController<String>.broadcast();
  final shareEvents = StreamController<Map<String, Object?>>.broadcast();
  final encode = StreamController<double>.broadcast();
  final log = <String>[];
  int written = 0;
  bool playing = false;
  int _played = 0;
  DateTime? _since;

  /// Speed of the fake clock (tests run playback faster than real time).
  double timeScale = 50;

  int get _now {
    if (!playing || _since == null) return _played;
    final elapsed = DateTime.now().difference(_since!).inMicroseconds / 1e6 * timeScale;
    return math.min(written, _played + (elapsed * outputRate).round());
  }

  @override
  Future<void> playerOpen(int sampleRate) async {
    log.add('open');
    written = 0;
    _played = 0;
    playing = true;
    _since = DateTime.now();
  }

  @override
  Future<int> playerWrite(Uint8List pcm16) async {
    written += pcm16.length ~/ 2;
    return written;
  }

  @override
  Future<int> playerPosition() async => _now;

  @override
  Future<void> playerPause() async {
    log.add('pause');
    _played = _now;
    playing = false;
  }

  @override
  Future<void> playerResume() async {
    log.add('resume');
    _since = DateTime.now();
    playing = true;
  }

  @override
  Future<void> playerStop() async {
    log.add('stop');
    playing = false;
  }

  @override
  Future<void> serviceShow({required String kind, required String title, required String text, required bool playing}) async =>
      log.add('service:$kind');

  @override
  Future<void> serviceStop() async => log.add('service:stop');

  @override
  Future<void> encodeM4a(String wav, String out, {int bitrate = 96000}) async {
    encode.add(0.5);
    await File(wav).copy(out);
    encode.add(1);
  }

  @override
  Future<(Float32List, int)> decodeAudio(String path, {double maxSeconds = 30}) async =>
      (Float32List(outputRate * 5), outputRate);

  @override
  Future<String> saveToDownloads(String path, String name, String mime) async => 'Downloads/Narrator Studio/$name';

  @override
  Future<void> share(String path, String mime, {String? subject}) async => log.add('share:$path');

  @override
  Future<Map<String, Object?>?> pendingShare() async => null;

  @override
  Future<bool> requestPermission(String name) async => true;

  @override
  Future<void> recordStart({int sampleRate = 24000}) async {}

  @override
  Future<double> recordLevel() async => 0.4;

  @override
  Future<(Uint8List, int)> recordStop() async => (Uint8List(outputRate * 2 * 6), outputRate);

  @override
  Future<void> keepScreenOn(bool on) async => log.add('screen:$on');

  @override
  Future<String> openClaude() async => 'web';

  @override
  Future<void> openUrl(String url) async {}

  @override
  Stream<String> get mediaActions => media.stream;

  @override
  Stream<Map<String, Object?>> get shares => shareEvents.stream;

  @override
  Stream<double> get encodeProgress => encode.stream;
}

/// Marks packs as installed without downloading anything.
void fakeInstall(ModelStore models, String pack) {
  final spec = voicePacks[pack]!;
  final dir = models.packDir(pack)..createSync(recursive: true);
  File(p.join(dir.path, '.installed')).writeAsStringSync(spec.fingerprint);
}

Future<(AppState, FakeSynth, FakeNative, Directory)> makeApp({List<String> packs = const ['kokoro']}) async {
  final tmp = await Directory.systemTemp.createTemp('narrator');
  final synth = FakeSynth();
  final native = FakeNative();
  final app = AppState(
    supportDir: Directory(p.join(tmp.path, 'support'))..createSync(),
    documentsDir: Directory(p.join(tmp.path, 'docs'))..createSync(),
    native: native,
    synth: (_) => synth,
  );
  for (final pack in packs) {
    fakeInstall(app.models, pack);
  }
  await app.init();
  return (app, synth, native, tmp);
}
