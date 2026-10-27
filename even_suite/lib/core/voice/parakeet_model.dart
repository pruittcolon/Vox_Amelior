import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../log.dart';

/// One file of a downloadable speech model.
class ModelFile {
  const ModelFile(this.name, this.url, this.bytes);
  final String name;
  final String url;
  final int bytes;
}

enum ModelState { unknown, missing, downloading, ready, failed }

/// NVIDIA Parakeet TDT 0.6B v2 (int8, the same files as the Vox app) and
/// the Silero speech detector.
class ParakeetFiles {
  ParakeetFiles._();

  static const _int8 = 'https://github.com/pruittcolon/Vox_Amelior/releases/download/parakeet-tdt-0.6b-v2-int8-files';
  static const _fp16 = 'https://github.com/pruittcolon/Vox_Amelior/releases/download/parakeet-tdt-0.6b-v2-fp16-files';

  static const encoder = 'encoder.int8.onnx';
  static const decoder = 'decoder.int8.onnx';
  static const joiner = 'joiner.int8.onnx';
  static const tokens = 'tokens.txt';
  static const vad = 'silero_vad.onnx';

  static const all = [
    ModelFile(encoder, '$_int8/encoder.int8.onnx', 652184296),
    ModelFile(decoder, '$_int8/decoder.int8.onnx', 7257753),
    ModelFile(joiner, '$_int8/joiner.int8.onnx', 1739080),
    ModelFile(tokens, '$_fp16/tokens.txt', 9384),
    ModelFile(vad, 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx', 643854),
  ];
}

/// Downloads a speech model into the app's own files, picking up where it
/// left off after a cut connection, and says how far it is.
class SpeechModel extends ChangeNotifier {
  SpeechModel(this.filesDir, {this.files = ParakeetFiles.all, this.folder = 'parakeet'});

  /// The app's files folder (from Android).
  final Future<String?> Function() filesDir;
  final List<ModelFile> files;
  final String folder;

  ModelState state = ModelState.unknown;
  int received = 0;
  String? error;
  String? dir;

  bool _cancel = false;
  DateTime _lastNotify = DateTime.fromMillisecondsSinceEpoch(0);

  int get totalBytes => files.fold(0, (n, f) => n + f.bytes);
  double get progress => totalBytes == 0 ? 0 : (received / totalBytes).clamp(0.0, 1.0);
  bool get ready => state == ModelState.ready;
  String path(String name) => '$dir/$name';

  Future<String?> _dir() async {
    if (dir != null) return dir;
    final base = await filesDir();
    if (base == null || base.isEmpty) return null;
    return dir = '$base/$folder';
  }

  /// Look at what is on the phone already.
  Future<void> check() async {
    if (state == ModelState.downloading) return;
    final d = await _dir();
    if (d == null) return;
    var have = 0;
    var complete = true;
    for (final f in files) {
      final file = File('$d/${f.name}');
      final part = File('$d/${f.name}.part');
      if (file.existsSync() && file.lengthSync() == f.bytes) {
        have += f.bytes;
      } else {
        complete = false;
        if (part.existsSync()) have += part.lengthSync().clamp(0, f.bytes);
      }
    }
    received = have;
    state = complete ? ModelState.ready : ModelState.missing;
    notifyListeners();
  }

  /// Fetch whatever is missing. Safe to call again after a failure: it resumes.
  Future<void> download({HttpClient Function()? client}) async {
    if (state == ModelState.downloading) return;
    final d = await _dir();
    if (d == null) {
      _fail('No place to save it');
      return;
    }
    _cancel = false;
    error = null;
    state = ModelState.downloading;
    notifyListeners();
    final http = (client ?? HttpClient.new)()..connectionTimeout = const Duration(seconds: 20);
    final sw = Stopwatch()..start();
    try {
      await Directory(d).create(recursive: true);
      var done = 0;
      for (final f in files) {
        final file = File('$d/${f.name}');
        if (file.existsSync() && file.lengthSync() == f.bytes) {
          done += f.bytes;
          continue;
        }
        await _fetch(http, f, file, done);
        done += f.bytes;
      }
      received = done;
      state = ModelState.ready;
      EventLog.add('speech', 'modelReady', {'mb': totalBytes ~/ 1000000, 'secs': sw.elapsed.inSeconds});
      notifyListeners();
    } on _Cancelled {
      state = ModelState.missing;
      notifyListeners();
    } catch (e) {
      _fail('$e');
    } finally {
      http.close(force: true);
    }
  }

  Future<void> _fetch(HttpClient http, ModelFile f, File file, int before) async {
    final part = File('${file.path}.part');
    var have = part.existsSync() ? part.lengthSync() : 0;
    if (have > f.bytes) {
      part.deleteSync();
      have = 0;
    }
    var url = Uri.parse(f.url);
    HttpClientResponse? res;
    // Redirects by hand, so the Range header goes to the final host too.
    for (var hop = 0; hop < 6; hop++) {
      final req = await http.getUrl(url).timeout(const Duration(seconds: 30));
      req.followRedirects = false;
      if (have > 0) req.headers.set(HttpHeaders.rangeHeader, 'bytes=$have-');
      res = await req.close().timeout(const Duration(seconds: 30));
      if (res.isRedirect || (res.statusCode >= 300 && res.statusCode < 400)) {
        final loc = res.headers.value(HttpHeaders.locationHeader);
        await res.drain<void>();
        if (loc == null) throw HttpException('Redirect without a location', uri: url);
        url = url.resolve(loc);
        res = null;
        continue;
      }
      break;
    }
    if (res == null) throw HttpException('Too many redirects', uri: url);
    if (res.statusCode == 416) {
      // Already all there.
      await res.drain<void>();
    } else if (res.statusCode == 200 || res.statusCode == 206) {
      final append = res.statusCode == 206 && have > 0;
      if (!append) have = 0;
      final sink = part.openWrite(mode: append ? FileMode.append : FileMode.write);
      try {
        await for (final chunk in res.timeout(const Duration(seconds: 60))) {
          if (_cancel) throw _Cancelled();
          sink.add(chunk);
          have += chunk.length;
          received = before + have;
          final now = DateTime.now();
          if (now.difference(_lastNotify) > const Duration(milliseconds: 200)) {
            _lastNotify = now;
            notifyListeners();
          }
        }
      } finally {
        await sink.close();
      }
    } else {
      await res.drain<void>();
      throw HttpException('Download failed (${res.statusCode})', uri: url);
    }
    final got = part.lengthSync();
    if (got != f.bytes) {
      if (got > f.bytes) part.deleteSync();
      throw HttpException('${f.name}: got $got of ${f.bytes} bytes - tap Download to resume');
    }
    if (file.existsSync()) file.deleteSync();
    part.renameSync(file.path);
  }

  void cancel() => _cancel = true;

  /// Free the space again.
  Future<void> delete() async {
    final d = await _dir();
    if (d == null || state == ModelState.downloading) return;
    final folder = Directory(d);
    if (folder.existsSync()) folder.deleteSync(recursive: true);
    received = 0;
    state = ModelState.missing;
    notifyListeners();
  }

  void _fail(String why) {
    error = why;
    state = ModelState.failed;
    EventLog.add('speech', 'modelFailed', {'error': why});
    notifyListeners();
  }
}

class _Cancelled implements Exception {}
