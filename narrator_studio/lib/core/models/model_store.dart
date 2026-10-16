/// Downloads, verifies, unpacks and deletes the voice packs.
///
/// Files come one by one from the `narrator-models` GitHub release. Each is
/// downloaded to `<name>.part` with HTTP Range resume, checked against its
/// sha256, then moved into place; zips (espeak-ng-data) are unpacked.
library;

import 'dart:async';
import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import 'catalog.dart';

enum PackState { missing, downloading, installed, failed }

class PackStatus {
  const PackStatus(this.state, {this.received = 0, this.total = 0, this.error, this.file});
  final PackState state;
  final int received;
  final int total;
  final String? error;
  final String? file;

  double get progress => total == 0 ? 0 : (received / total).clamp(0, 1).toDouble();
}

class DownloadCancelled implements Exception {
  @override
  String toString() => 'Download cancelled';
}

class ModelStore extends ChangeNotifier {
  ModelStore(this.root, {HttpClient Function()? client, this.baseUrl = releaseBaseUrl})
      : _client = client ?? HttpClient.new;

  /// `<app support>/models`
  final Directory root;
  final String baseUrl;
  final HttpClient Function() _client;
  final Map<String, PackStatus> _status = {};
  final Map<String, bool> _cancel = {};

  Directory packDir(String pack) => Directory(p.join(root.path, pack));

  String path(String pack, String relative) => p.join(root.path, pack, relative);

  PackStatus status(String pack) => _status[pack] ?? PackStatus(_installedNow(pack) ? PackState.installed : PackState.missing);

  bool isInstalled(String pack) => status(pack).state == PackState.installed;

  File _marker(String pack) => File(p.join(root.path, pack, '.installed'));

  bool _installedNow(String pack) {
    final spec = voicePacks[pack];
    if (spec == null) return false;
    final m = _marker(pack);
    return m.existsSync() && m.readAsStringSync().trim() == spec.fingerprint;
  }

  /// Bytes the pack takes on disk now.
  int diskBytes(String pack) {
    final d = packDir(pack);
    if (!d.existsSync()) return 0;
    var n = 0;
    for (final e in d.listSync(recursive: true, followLinks: false)) {
      if (e is File) n += e.lengthSync();
    }
    return n;
  }

  void cancel(String pack) => _cancel[pack] = true;

  Future<void> delete(String pack) async {
    cancel(pack);
    final d = packDir(pack);
    if (d.existsSync()) await d.delete(recursive: true);
    _status.remove(pack);
    notifyListeners();
  }

  /// Downloads every file of [pack]; resumes what is already there.
  Future<void> install(String pack) async {
    final spec = voicePacks[pack]!;
    if (status(pack).state == PackState.downloading) return;
    _cancel[pack] = false;
    final dir = packDir(pack)..createSync(recursive: true);
    final total = spec.files.fold<int>(0, (a, f) => a + f.size);
    var done = 0;
    void report(int extra, String file) {
      _status[pack] = PackStatus(PackState.downloading, received: done + extra, total: total, file: file);
      notifyListeners();
    }

    try {
      for (final f in spec.files) {
        final target = File(p.join(dir.path, f.unzip ? '${f.path}.zip' : f.path));
        final finalPath = f.unzip ? Directory(p.join(dir.path, f.path)) : target;
        final already = f.unzip ? File(p.join(dir.path, '.${f.path}.ok')).existsSync() : target.existsSync();
        if (already && (f.unzip || target.lengthSync() == f.size)) {
          done += f.size;
          report(0, f.path);
          continue;
        }
        target.parent.createSync(recursive: true);
        await downloadFile(
          Uri.parse('$baseUrl/${f.asset}'),
          target,
          size: f.size,
          sha256: f.sha256,
          onProgress: (n) => report(n, f.path),
          cancelled: () => _cancel[pack] == true,
        );
        if (f.unzip) {
          if (finalPath is Directory && finalPath.existsSync()) finalPath.deleteSync(recursive: true);
          await extractFileToDisk(target.path, dir.path);
          target.deleteSync();
          File(p.join(dir.path, '.${f.path}.ok')).writeAsStringSync(f.sha256);
        }
        done += f.size;
        report(0, f.path);
      }
      _marker(pack).writeAsStringSync(spec.fingerprint);
      _status[pack] = const PackStatus(PackState.installed);
    } on DownloadCancelled {
      _status[pack] = const PackStatus(PackState.missing);
    } catch (e) {
      _status[pack] = PackStatus(PackState.failed, error: _friendly(e), received: done, total: total);
    }
    notifyListeners();
  }

  /// One gallery clip (a few hundred KB); returns its local path.
  Future<String> fetchGalleryClip(GalleryVoice v) async {
    final file = File(path('gallery', v.asset));
    if (file.existsSync() && file.lengthSync() == v.size) return file.path;
    file.parent.createSync(recursive: true);
    await downloadFile(Uri.parse('$baseUrl/${v.asset}'), file, size: v.size, sha256: v.sha256);
    return file.path;
  }

  /// The gallery list (downloaded once, then refreshed when possible).
  Future<List<GalleryVoice>> gallery({bool refresh = false}) async {
    final file = File(path('gallery', 'gallery.json'));
    if (refresh || !file.existsSync()) {
      try {
        file.parent.createSync(recursive: true);
        await downloadFile(Uri.parse('$baseUrl/gallery.json'), file);
      } catch (_) {
        if (!file.existsSync()) rethrow;
      }
    }
    return parseGallery(await file.readAsString());
  }

  Future<void> downloadFile(
    Uri url,
    File target, {
    int? size,
    String? sha256,
    void Function(int received)? onProgress,
    bool Function()? cancelled,
  }) async {
    final part = File('${target.path}.part');
    for (var attempt = 1;; attempt++) {
      try {
        await _fetch(url, part, size: size, onProgress: onProgress, cancelled: cancelled);
        // The connection can end early without an error: resume the rest.
        if (size == null || part.lengthSync() >= size || attempt >= 6) break;
      } on DownloadCancelled {
        rethrow;
      } catch (e) {
        // Flaky mobile data: retry with resume, a few times, backing off.
        if (attempt >= 4) rethrow;
        await Future<void>.delayed(Duration(seconds: attempt * 2));
      }
    }
    if (size != null && part.lengthSync() != size) {
      part.deleteSync();
      throw StateError('${p.basename(target.path)}: got ${part.lengthSync()} bytes, expected $size');
    }
    if (sha256 != null) {
      final digest = await _sha256(part);
      if (digest != sha256) {
        part.deleteSync();
        throw StateError('${p.basename(target.path)} is damaged (checksum mismatch). Try again.');
      }
    }
    if (target.existsSync()) target.deleteSync();
    part.renameSync(target.path);
  }

  Future<void> _fetch(
    Uri url,
    File part, {
    int? size,
    void Function(int received)? onProgress,
    bool Function()? cancelled,
  }) async {
    var have = part.existsSync() ? part.lengthSync() : 0;
    if (size != null && have > size) {
      part.deleteSync();
      have = 0;
    }
    if (size != null && have == size) return;
    final client = _client()
      ..connectionTimeout = const Duration(seconds: 20)
      ..idleTimeout = const Duration(seconds: 20);
    try {
      var uri = url;
      HttpClientResponse? res;
      // Follow GitHub's redirect to its file host ourselves so Range survives.
      for (var hop = 0; hop < 6; hop++) {
        final req = await client.getUrl(uri);
        req.followRedirects = false;
        if (have > 0) req.headers.set(HttpHeaders.rangeHeader, 'bytes=$have-');
        res = await req.close();
        if (res.isRedirect || (res.statusCode >= 300 && res.statusCode < 400)) {
          final loc = res.headers.value(HttpHeaders.locationHeader);
          await res.drain<void>();
          if (loc == null) throw const HttpException('Redirect without a location');
          uri = uri.resolve(loc);
          continue;
        }
        break;
      }
      final r = res!;
      if (r.statusCode == 416 && size != null && have >= size) {
        await r.drain<void>();
        return;
      }
      if (r.statusCode != 200 && r.statusCode != 206) {
        await r.drain<void>();
        throw HttpException('Download failed (HTTP ${r.statusCode})', uri: uri);
      }
      final append = r.statusCode == 206;
      if (!append) have = 0;
      final sink = part.openWrite(mode: append ? FileMode.append : FileMode.write);
      var received = have;
      var lastReport = 0;
      try {
        await for (final chunk in r.timeout(const Duration(seconds: 30))) {
          if (cancelled?.call() == true) throw DownloadCancelled();
          sink.add(chunk);
          received += chunk.length;
          if (received - lastReport > 256 * 1024) {
            lastReport = received;
            onProgress?.call(received);
          }
        }
      } finally {
        await sink.close();
      }
      onProgress?.call(received);
    } finally {
      client.close(force: true);
    }
  }

  static Future<String> _sha256(File f) async {
    final digest = await sha256.bind(f.openRead()).first;
    return digest.toString();
  }

  static String _friendly(Object e) {
    final s = e.toString();
    if (e is SocketException || s.contains('Failed host lookup')) {
      return 'No internet connection. Connect and tap Download again; it resumes where it stopped.';
    }
    if (e is TimeoutException) return 'The connection stalled. Tap Download again to resume.';
    if (e is FileSystemException && s.contains('No space')) return 'Your phone is out of storage space.';
    return s.replaceFirst(RegExp(r'^(Exception|StateError|HttpException): ?'), '');
  }
}
