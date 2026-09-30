import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive_io.dart';
import 'package:crypto/crypto.dart';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/models/archive_extractor.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_installer.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/models/resumable_downloader.dart';

/// A tiny file server with the behaviours real hosts have: Range support,
/// redirects, gated access, dropped connections.
class TestServer {
  TestServer._(this._server);

  final HttpServer _server;
  final Map<String, Uint8List> files = {};
  final List<HttpRequest> log = [];
  final List<Map<String, String?>> requestHeaders = [];
  bool honourRange = true;
  bool paced = false; // send in small delayed chunks, like a real network
  int? dropAfterBytes; // simulate a connection dying mid-download
  String? requiredToken;
  String? redirectTo; // path -> full url
  Map<String, String> firstResponseHeaders = {};

  static Future<TestServer> start() async {
    final s = TestServer._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));
    unawaited(s._serve());
    return s;
  }

  int get port => _server.port;
  Uri url(String path) => Uri.parse('http://127.0.0.1:$port$path');
  Future<void> close() => _server.close(force: true);

  Future<void> _serve() async {
    await for (final req in _server) {
      // Each request is isolated: a client hanging up must not kill the server.
      unawaited(_handle(req).catchError((Object _) {}));
    }
  }

  Future<void> _handle(HttpRequest req) async {
    {
      requestHeaders.add({
        'range': req.headers.value('range'),
        'authorization': req.headers.value('authorization'),
        'path': req.uri.path,
      });
      final path = req.uri.path;
      final res = req.response;

      if (redirectTo != null && path == '/gated') {
        firstResponseHeaders.forEach(res.headers.set);
        res.statusCode = 302;
        res.headers.set('location', redirectTo!);
        await res.close();
        return;
      }
      if (requiredToken != null && req.headers.value('authorization') != 'Bearer $requiredToken') {
        res.statusCode = 401;
        await res.close();
        return;
      }
      final data = files[path];
      if (data == null) {
        res.statusCode = 404;
        await res.close();
        return;
      }

      var start = 0;
      final range = req.headers.value('range');
      if (honourRange && range != null) {
        start = int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)!.group(1)!);
        if (start >= data.length) {
          res.statusCode = 416;
          await res.close();
          return;
        }
        res.statusCode = 206;
        res.headers.set('content-range', 'bytes $start-${data.length - 1}/${data.length}');
      }
      final body = Uint8List.sublistView(data, start);
      res.contentLength = body.length;
      if (dropAfterBytes != null) {
        final cut = math.min(dropAfterBytes!, body.length);
        dropAfterBytes = null;
        final socket = await res.detachSocket();
        socket.add(Uint8List.sublistView(body, 0, cut));
        await socket.flush();
        socket.destroy();
        return;
      }
      if (paced) {
        for (var i = 0; i < body.length; i += 32768) {
          res.add(Uint8List.sublistView(body, i, math.min(i + 32768, body.length)));
          await res.flush();
          await Future<void>.delayed(const Duration(milliseconds: 2));
        }
        await res.close();
        return;
      }
      res.add(body);
      await res.close();
    }
  }
}

Uint8List bytes(int n, {int seed = 1}) {
  final r = math.Random(seed);
  return Uint8List.fromList(List.generate(n, (_) => r.nextInt(256)));
}

String sha(Uint8List b) => sha256.convert(b).toString();

Uint8List tarBz2(Map<String, Uint8List> entries) {
  final archive = Archive();
  entries.forEach((name, data) => archive.add(ArchiveFile(name, data.length, data)));
  final tar = TarEncoder().encodeBytes(archive);
  return Uint8List.fromList(BZip2Encoder().encodeBytes(tar));
}

void main() {
  late Directory tmp;
  late TestServer server;

  setUp(() async {
    tmp = Directory.systemTemp.createTempSync('vox_models_');
    server = await TestServer.start();
  });
  tearDown(() async {
    await server.close();
    tmp.deleteSync(recursive: true);
  });

  group('ResumableDownloader', () {
    test('downloads, reports progress, and verifies size and checksum', () async {
      final data = bytes(300000);
      server.files['/f.bin'] = data;
      final progress = <int>[];
      final file = await ResumableDownloader().download(
        url: server.url('/f.bin'),
        destination: File(p.join(tmp.path, 'f.bin')),
        expectedSha256: sha(data),
        expectedSize: data.length,
        onProgress: (r, t) => progress.add(r),
      );
      expect(file.readAsBytesSync(), data);
      expect(progress.last, data.length);
      expect(File('${file.path}.part').existsSync(), isFalse);
    });

    test('resumes a partial file using a Range request', () async {
      final data = bytes(100000, seed: 2);
      server.files['/f.bin'] = data;
      final dest = File(p.join(tmp.path, 'f.bin'));
      File('${dest.path}.part').writeAsBytesSync(data.sublist(0, 40000));
      await ResumableDownloader().download(url: server.url('/f.bin'), destination: dest, expectedSha256: sha(data));
      expect(dest.readAsBytesSync(), data);
      expect(server.requestHeaders.single['range'], 'bytes=40000-');
    });

    test('restarts cleanly when the server ignores Range', () async {
      final data = bytes(50000, seed: 3);
      server.files['/f.bin'] = data;
      server.honourRange = false;
      final dest = File(p.join(tmp.path, 'f.bin'));
      File('${dest.path}.part').writeAsBytesSync(data.sublist(0, 10000));
      await ResumableDownloader().download(url: server.url('/f.bin'), destination: dest, expectedSha256: sha(data));
      expect(dest.readAsBytesSync(), data);
    });

    test('an interrupted download keeps its progress and can be finished', () async {
      final data = bytes(200000, seed: 4);
      server.files['/f.bin'] = data;
      server.dropAfterBytes = 60000;
      final dest = File(p.join(tmp.path, 'f.bin'));
      final dl = ResumableDownloader();
      await expectLater(
        dl.download(url: server.url('/f.bin'), destination: dest, expectedSize: data.length),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.network)),
      );
      final partial = File('${dest.path}.part');
      expect(partial.existsSync(), isTrue);
      expect(partial.lengthSync(), greaterThan(0));
      expect(partial.lengthSync(), lessThan(data.length));
      await dl.download(url: server.url('/f.bin'), destination: dest, expectedSha256: sha(data));
      expect(dest.readAsBytesSync(), data);
    });

    test('a corrupted download is rejected and discarded', () async {
      final data = bytes(20000, seed: 5);
      server.files['/f.bin'] = data;
      final dest = File(p.join(tmp.path, 'f.bin'));
      await expectLater(
        ResumableDownloader().download(url: server.url('/f.bin'), destination: dest, expectedSha256: sha(bytes(10))),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.checksum)),
      );
      expect(dest.existsSync(), isFalse);
      expect(File('${dest.path}.part').existsSync(), isFalse);
    });

    test('wrong size is rejected', () async {
      server.files['/f.bin'] = bytes(1000);
      await expectLater(
        ResumableDownloader().download(url: server.url('/f.bin'), destination: File(p.join(tmp.path, 'f')), expectedSize: 999),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.sizeMismatch)),
      );
    });

    test('gated files: token required (401) then works with the token', () async {
      final data = bytes(5000, seed: 6);
      server.files['/f.bin'] = data;
      server.requiredToken = 'hf_secret';
      final dest = File(p.join(tmp.path, 'f.bin'));
      await expectLater(
        ResumableDownloader().download(url: server.url('/f.bin'), destination: dest),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.needsAuth)),
      );
      await ResumableDownloader().download(
        url: server.url('/f.bin'),
        destination: dest,
        headers: {'Authorization': 'Bearer hf_secret'},
      );
      expect(dest.readAsBytesSync(), data);
    });

    test('missing files report not found', () async {
      await expectLater(
        ResumableDownloader().download(url: server.url('/nope'), destination: File(p.join(tmp.path, 'x'))),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.notFound)),
      );
    });

    test('redirects: token goes to the origin only, never to the CDN; linked etag verifies the file', () async {
      final cdn = await TestServer.start();
      addTearDown(cdn.close);
      final data = bytes(30000, seed: 7);
      cdn.files['/blob'] = data;
      server.redirectTo = cdn.url('/blob').toString();
      server.firstResponseHeaders = {'x-linked-etag': '"${sha(data)}"', 'x-linked-size': '${data.length}'};

      final dest = File(p.join(tmp.path, 'g.bin'));
      await ResumableDownloader().download(
        url: server.url('/gated'),
        destination: dest,
        headers: {'Authorization': 'Bearer hf_secret'},
      );
      expect(dest.readAsBytesSync(), data);
      expect(server.requestHeaders.first['authorization'], 'Bearer hf_secret');
      expect(cdn.requestHeaders.single['authorization'], isNull);
    });

    test('a wrong linked etag from the origin fails verification', () async {
      final cdn = await TestServer.start();
      addTearDown(cdn.close);
      cdn.files['/blob'] = bytes(1000, seed: 8);
      server.redirectTo = cdn.url('/blob').toString();
      server.firstResponseHeaders = {'x-linked-etag': '"${sha(bytes(3))}"'};
      await expectLater(
        ResumableDownloader().download(url: server.url('/gated'), destination: File(p.join(tmp.path, 'g.bin'))),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.checksum)),
      );
    });

    test('cancelling stops the download but keeps the partial file for resuming', () async {
      final data = bytes(3000000, seed: 9);
      server.files['/big'] = data;
      server.paced = true;
      final token = CancelToken();
      final dest = File(p.join(tmp.path, 'big'));
      final future = ResumableDownloader().download(
        url: server.url('/big'),
        destination: dest,
        cancelToken: token,
        onProgress: (r, t) {
          if (r > 0 && !token.isCancelled) token.cancel();
        },
      );
      await expectLater(future, throwsA(isA<DownloadCancelled>()));
      expect(dest.existsSync(), isFalse);
      expect(File('${dest.path}.part').lengthSync(), greaterThan(0), reason: 'progress is kept for resuming');
      server.paced = false;
      await ResumableDownloader().download(url: server.url('/big'), destination: dest, expectedSha256: sha(data));
      expect(dest.readAsBytesSync(), data);
    });

    test('a complete .part file (416) is finalised without re-downloading', () async {
      final data = bytes(5000, seed: 10);
      server.files['/f.bin'] = data;
      final dest = File(p.join(tmp.path, 'f.bin'));
      File('${dest.path}.part').writeAsBytesSync(data);
      await ResumableDownloader().download(url: server.url('/f.bin'), destination: dest, expectedSize: data.length, expectedSha256: sha(data));
      expect(dest.readAsBytesSync(), data);
    });
  });

  group('ArchiveExtractor', () {
    test('extracts only wanted files, flattening directories', () async {
      final archive = File(p.join(tmp.path, 'a.tar.bz2'))
        ..writeAsBytesSync(tarBz2({
          'model-dir/encoder.onnx': bytes(5000, seed: 11),
          'model-dir/tokens.txt': Uint8List.fromList(utf8.encode('a 0\nb 1\n')),
          'model-dir/test_wavs/0.wav': bytes(2000, seed: 12),
        }));
      final out = Directory(p.join(tmp.path, 'out'));
      final files = await const ArchiveExtractor().extractTarBz2(archive, out, wanted: {'encoder.onnx', 'tokens.txt'});
      expect(files.map((f) => p.basename(f.path)).toSet(), {'encoder.onnx', 'tokens.txt'});
      expect(File(p.join(out.path, 'tokens.txt')).readAsStringSync(), 'a 0\nb 1\n');
      expect(File(p.join(out.path, '0.wav')).existsSync(), isFalse);
      expect(File(p.join(out.path, '.extract.tar')).existsSync(), isFalse);
    });

    test('missing expected files raise a clear error', () async {
      final archive = File(p.join(tmp.path, 'a.tar.bz2'))..writeAsBytesSync(tarBz2({'d/only.txt': bytes(10)}));
      await expectLater(
        const ArchiveExtractor().extractTarBz2(archive, Directory(p.join(tmp.path, 'o')), wanted: {'encoder.onnx'}),
        throwsA(isA<StateError>()),
      );
    });

    test('path traversal in entry names cannot escape the destination', () async {
      final archive = File(p.join(tmp.path, 'evil.tar.bz2'))
        ..writeAsBytesSync(tarBz2({'../../evil.txt': Uint8List.fromList(utf8.encode('x'))}));
      final out = Directory(p.join(tmp.path, 'safe', 'out'));
      await const ArchiveExtractor().extractTarBz2(archive, out, wanted: {'evil.txt'});
      expect(File(p.join(out.path, 'evil.txt')).existsSync(), isTrue);
      expect(File(p.join(tmp.path, 'evil.txt')).existsSync(), isFalse);
    });
  });

  group('ModelInstaller + ModelStore', () {
    ModelAsset assetFor(Uint8List archive, {Set<String>? extract, bool archived = true}) => ModelAsset(
          id: 'test-model',
          kind: ModelKind.speechToText,
          title: 't',
          description: 'd',
          approxDownloadBytes: archive.length,
          files: [
            RemoteFile(
              url: server.url('/model.tar.bz2').toString(),
              fileName: 'model.tar.bz2',
              sha256: sha(archive),
              sizeBytes: archive.length,
              extractFromArchive: extract ?? {'encoder.onnx', 'tokens.txt'},
            ),
          ],
        );

    test('installs an archive model end to end and marks it installed', () async {
      final archive = tarBz2({'m/encoder.onnx': bytes(3000, seed: 13), 'm/tokens.txt': bytes(50, seed: 14)});
      server.files['/model.tar.bz2'] = archive;
      final asset = assetFor(archive);
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      expect(store.isInstalled(asset), isFalse);

      final phases = <InstallPhase>{};
      await ModelInstaller(store: store, retryDelay: Duration.zero).install(asset, onProgress: (pr) => phases.add(pr.phase));

      expect(store.isInstalled(asset), isTrue);
      expect(store.file(asset, 'encoder.onnx').lengthSync(), 3000);
      expect(File(p.join(store.dir(asset).path, 'model.tar.bz2')).existsSync(), isFalse, reason: 'archive is deleted after unpacking');
      expect(phases, containsAll([InstallPhase.downloading, InstallPhase.unpacking, InstallPhase.finishing]));
    });

    test('is a no-op when already installed; detects a damaged install', () async {
      final archive = tarBz2({'m/encoder.onnx': bytes(300, seed: 15), 'm/tokens.txt': bytes(20, seed: 16)});
      server.files['/model.tar.bz2'] = archive;
      final asset = assetFor(archive);
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      final installer = ModelInstaller(store: store, retryDelay: Duration.zero);
      await installer.install(asset);
      final requests = server.requestHeaders.length;
      await installer.install(asset);
      expect(server.requestHeaders.length, requests);

      store.file(asset, 'encoder.onnx').writeAsBytesSync([1, 2, 3]); // truncate
      expect(store.isInstalled(asset), isFalse);
    });

    test('retries a dropped connection automatically and resumes', () async {
      final archive = tarBz2({'m/encoder.onnx': bytes(60000, seed: 17), 'm/tokens.txt': bytes(20, seed: 18)});
      server.files['/model.tar.bz2'] = archive;
      server.dropAfterBytes = 5000;
      final asset = assetFor(archive);
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      await ModelInstaller(store: store, retryDelay: Duration.zero).install(asset);
      expect(store.isInstalled(asset), isTrue);
      expect(server.requestHeaders.any((h) => h['range'] != null), isTrue);
    });

    test('sends the token for gated models and surfaces access errors', () async {
      final data = bytes(4000, seed: 19);
      server.files['/gemma.bin'] = data;
      server.requiredToken = 'hf_x';
      final asset = ModelAsset(
        id: 'gated',
        kind: ModelKind.languageModel,
        title: 'g',
        description: 'g',
        approxDownloadBytes: data.length,
        requiresToken: true,
        files: [RemoteFile(url: server.url('/gemma.bin').toString(), fileName: 'gemma.bin')],
      );
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      final installer = ModelInstaller(store: store, retryDelay: Duration.zero);
      await expectLater(
        installer.install(asset),
        throwsA(isA<DownloadException>().having((e) => e.failure, 'failure', DownloadFailure.needsAuth)),
      );
      expect(store.isInstalled(asset), isFalse);
      await installer.install(asset, token: 'hf_x');
      expect(store.isInstalled(asset), isTrue);
    });
  });

  group('catalog', () {
    test('essential models are the speech stack and the assistant is optional', () {
      expect(ModelCatalog.all.where((m) => m.essential).map((m) => m.kind).toSet(), {
        ModelKind.speechToText,
        ModelKind.voiceActivity,
        ModelKind.speakerVoiceprint,
      });
      expect(ModelCatalog.gemma4E4b.requiresToken, isFalse);
      expect(ModelCatalog.gemma4E4b.llmType, 'gemma4');
      expect(ModelCatalog.gemma4E4b.supportsTools, isTrue);
      expect(ModelCatalog.parakeet.installedFileNames,
          {'encoder.int8.onnx', 'encoder.int8.weights', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'});
      // Every fixed download is pinned to an exact size and checksum.
      for (final m in ModelCatalog.all) {
        for (final f in m.files) {
          expect(f.sha256, hasLength(64), reason: f.url);
          expect(f.sizeBytes, isNotNull, reason: f.url);
        }
      }
      expect(ModelCatalog.custom(url: 'https://x.io/a/b/my.litertlm').files.single.fileName, 'my.litertlm');
      expect(ModelCatalog.custom(url: 'https://x.io/').files.single.fileName, 'custom.litertlm');
      expect(ModelCatalog.all.map((m) => m.id).toSet().length, ModelCatalog.all.length);
      expect(ModelCatalog.all.every((m) => m.files.every((f) => Uri.parse(f.url).scheme == 'https')), isTrue);
    });
  });
}
