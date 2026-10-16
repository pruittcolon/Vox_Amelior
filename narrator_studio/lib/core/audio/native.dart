/// The Android side: speaker output, background service, audio files, mic,
/// sharing. Tests use [FakeNative].
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

abstract class NativeApi {
  // Speech output.
  Future<void> playerOpen(int sampleRate);

  /// Returns total frames written so far (after this write is buffered).
  Future<int> playerWrite(Uint8List pcm16);
  Future<int> playerPosition();
  Future<void> playerPause();
  Future<void> playerResume();
  Future<void> playerStop();

  // Background service.
  Future<void> serviceShow({required String kind, required String title, required String text, required bool playing});
  Future<void> serviceStop();

  // Files.
  Future<void> encodeM4a(String wav, String out, {int bitrate = 96000});
  Future<(Float32List, int)> decodeAudio(String path, {double maxSeconds = 30});
  Future<String> saveToDownloads(String path, String name, String mime);
  Future<void> share(String path, String mime, {String? subject});
  Future<Map<String, Object?>?> pendingShare();

  // Mic.
  Future<bool> requestPermission(String name);
  Future<void> recordStart({int sampleRate = 24000});
  Future<double> recordLevel();
  Future<(Uint8List, int)> recordStop();

  // App.
  Future<void> keepScreenOn(bool on);
  Future<String> openClaude();
  Future<void> openUrl(String url);

  /// Notification / headset buttons: play, pause, next, previous, stop.
  Stream<String> get mediaActions;

  /// Text or files shared into the app while it is running.
  Stream<Map<String, Object?>> get shares;

  /// Progress of [encodeM4a], 0..1.
  Stream<double> get encodeProgress;
}

class AndroidNative implements NativeApi {
  AndroidNative() {
    _ch.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'media':
          _media.add(call.arguments as String);
        case 'shared':
          _shares.add((call.arguments as Map).cast<String, Object?>());
        case 'encodeProgress':
          _encode.add((call.arguments as num).toDouble());
      }
      return null;
    });
  }

  static const _ch = MethodChannel('narrator/native');
  final _media = StreamController<String>.broadcast();
  final _shares = StreamController<Map<String, Object?>>.broadcast();
  final _encode = StreamController<double>.broadcast();

  @override
  Stream<String> get mediaActions => _media.stream;
  @override
  Stream<Map<String, Object?>> get shares => _shares.stream;
  @override
  Stream<double> get encodeProgress => _encode.stream;

  @override
  Future<void> playerOpen(int sampleRate) => _ch.invokeMethod('player.open', {'sampleRate': sampleRate});
  @override
  Future<int> playerWrite(Uint8List pcm16) async =>
      ((await _ch.invokeMethod<num>('player.write', {'pcm': pcm16})) ?? 0).toInt();
  @override
  Future<int> playerPosition() async => ((await _ch.invokeMethod<num>('player.position')) ?? 0).toInt();
  @override
  Future<void> playerPause() => _ch.invokeMethod('player.pause');
  @override
  Future<void> playerResume() => _ch.invokeMethod('player.resume');
  @override
  Future<void> playerStop() => _ch.invokeMethod('player.stop');

  @override
  Future<void> serviceShow({required String kind, required String title, required String text, required bool playing}) =>
      _ch.invokeMethod('service.start', {'kind': kind, 'title': title, 'text': text, 'playing': playing});
  @override
  Future<void> serviceStop() => _ch.invokeMethod('service.stop');

  @override
  Future<void> encodeM4a(String wav, String out, {int bitrate = 96000}) =>
      _ch.invokeMethod('encode.m4a', {'wav': wav, 'out': out, 'bitrate': bitrate});

  @override
  Future<(Float32List, int)> decodeAudio(String path, {double maxSeconds = 30}) async {
    final r = (await _ch.invokeMethod<Map<Object?, Object?>>('decode.audio', {'path': path, 'maxSeconds': maxSeconds}))!;
    final bytes = r['pcm']! as Uint8List;
    final copy = Uint8List.fromList(bytes); // aligned for the float view
    return (copy.buffer.asFloat32List(0, copy.length ~/ 4), (r['sampleRate']! as num).toInt());
  }

  @override
  Future<String> saveToDownloads(String path, String name, String mime) async =>
      (await _ch.invokeMethod<String>('files.saveToDownloads', {'path': path, 'name': name, 'mime': mime}))!;
  @override
  Future<void> share(String path, String mime, {String? subject}) =>
      _ch.invokeMethod('files.share', {'path': path, 'mime': mime, 'subject': subject});
  @override
  Future<Map<String, Object?>?> pendingShare() async =>
      (await _ch.invokeMethod<Map<Object?, Object?>>('files.shared'))?.cast<String, Object?>();

  @override
  Future<bool> requestPermission(String name) async =>
      (await _ch.invokeMethod<bool>('perm.request', {'name': name})) ?? false;
  @override
  Future<void> recordStart({int sampleRate = 24000}) => _ch.invokeMethod('record.start', {'sampleRate': sampleRate});
  @override
  Future<double> recordLevel() async => ((await _ch.invokeMethod<num>('record.level')) ?? 0).toDouble();
  @override
  Future<(Uint8List, int)> recordStop() async {
    final r = (await _ch.invokeMethod<Map<Object?, Object?>>('record.stop'))!;
    return (r['pcm']! as Uint8List, (r['sampleRate']! as num).toInt());
  }

  @override
  Future<void> keepScreenOn(bool on) => _ch.invokeMethod('app.keepScreenOn', {'on': on});
  @override
  Future<String> openClaude() async => (await _ch.invokeMethod<String>('app.openClaude')) ?? 'web';
  @override
  Future<void> openUrl(String url) => _ch.invokeMethod('app.openUrl', {'url': url});
}
