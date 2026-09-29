import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dio/dio.dart';

enum DownloadFailure { needsAuth, notFound, network, diskFull, checksum, sizeMismatch, http }

class DownloadException implements Exception {
  const DownloadException(this.failure, this.message);

  final DownloadFailure failure;
  final String message;

  @override
  String toString() => message;
}

class DownloadCancelled implements Exception {
  const DownloadCancelled();
}

/// Downloads a large file with resume support and integrity checks.
///
/// * Resumes from `<file>.part` using HTTP Range requests.
/// * Follows redirects manually, and never sends the `Authorization` header
///   to a different origin than the one the user's token was meant for
///   (Hugging Face redirects gated files to a CDN).
/// * Verifies size and SHA-256 (from the caller, or from Hugging Face's
///   `x-linked-etag` header) before the file is made visible.
class ResumableDownloader {
  ResumableDownloader({Dio? dio, this.maxRedirects = 5})
      : _dio = dio ?? Dio(BaseOptions(connectTimeout: const Duration(seconds: 30), receiveTimeout: const Duration(seconds: 60)));

  final Dio _dio;
  final int maxRedirects;

  Future<File> download({
    required Uri url,
    required File destination,
    Map<String, String> headers = const {},
    String? expectedSha256,
    int? expectedSize,
    void Function(int received, int total)? onProgress,
    CancelToken? cancelToken,
  }) async {
    await destination.parent.create(recursive: true);
    final part = File('${destination.path}.part');
    if (destination.existsSync() && expectedSize != null && destination.lengthSync() == expectedSize) {
      return destination;
    }

    var attemptedRestart = false;
    while (true) {
      final existing = part.existsSync() ? part.lengthSync() : 0;
      final result = await _open(url, headers, existing, cancelToken);
      final meta = result.meta;
      final size = expectedSize ?? meta.linkedSize;
      final sha = expectedSha256 ?? meta.linkedSha256;

      try {
        if (result.status == 416) {
          // Nothing left to fetch, or our partial file is unusable.
          if (size != null && existing == size) return await _finish(part, destination, size, sha);
          if (attemptedRestart) throw const DownloadException(DownloadFailure.http, 'Server rejected resume request');
          attemptedRestart = true;
          if (part.existsSync()) part.deleteSync();
          continue;
        }

        final resumed = result.status == 206;
        if (!resumed && existing > 0 && part.existsSync()) part.deleteSync(); // server ignored Range
        final start = resumed ? existing : 0;
        final total = start + (result.contentLength ?? -1);
        final declaredTotal = result.contentLength == null ? (size ?? 0) : total;

        final sink = part.openWrite(mode: resumed ? FileMode.append : FileMode.write);
        var received = start;
        var lastReport = DateTime.fromMillisecondsSinceEpoch(0);
        try {
          await for (final chunk in result.body) {
            if (cancelToken?.isCancelled ?? false) throw const DownloadCancelled();
            sink.add(chunk);
            received += chunk.length;
            final now = DateTime.now();
            if (now.difference(lastReport).inMilliseconds >= 200) {
              lastReport = now;
              onProgress?.call(received, declaredTotal);
            }
          }
          await sink.flush();
        } finally {
          await sink.close();
        }
        onProgress?.call(received, declaredTotal);

        if (result.contentLength != null && received != total) {
          throw const DownloadException(DownloadFailure.network, 'Connection closed before the download finished');
        }
        return await _finish(part, destination, size, sha);
      } on FileSystemException catch (e) {
        if (e.osError?.errorCode == 28) {
          throw const DownloadException(DownloadFailure.diskFull, 'Not enough free storage on the phone');
        }
        rethrow;
      } on IOException catch (e) {
        // Socket/HTTP errors while the body is streaming (connection dropped).
        throw DownloadException(DownloadFailure.network, 'Connection lost: ${e.runtimeType}');
      } on DioException catch (e) {
        throw _mapDio(e);
      }
    }
  }

  Future<File> _finish(File part, File destination, int? size, String? sha) async {
    if (size != null && part.lengthSync() != size) {
      final actual = part.lengthSync();
      part.deleteSync();
      throw DownloadException(DownloadFailure.sizeMismatch, 'Downloaded file has the wrong size ($actual of $size bytes)');
    }
    if (sha != null) {
      final actual = (await sha256.bind(part.openRead()).first).toString();
      if (actual != sha.toLowerCase()) {
        part.deleteSync();
        throw const DownloadException(DownloadFailure.checksum, 'Download is corrupted (checksum mismatch). Please retry.');
      }
    }
    if (destination.existsSync()) destination.deleteSync();
    return part.rename(destination.path);
  }

  Future<_Opened> _open(Uri url, Map<String, String> headers, int existing, CancelToken? cancelToken) async {
    var current = url;
    final origin = _origin(url);
    _Meta meta = const _Meta();
    for (var hop = 0; hop <= maxRedirects; hop++) {
      final sameOrigin = _origin(current) == origin;
      final requestHeaders = <String, String>{
        for (final e in headers.entries)
          if (sameOrigin || e.key.toLowerCase() != 'authorization') e.key: e.value,
        if (existing > 0) 'Range': 'bytes=$existing-',
      };
      final Response<ResponseBody> response;
      try {
        response = await _dio.getUri<ResponseBody>(
          current,
          cancelToken: cancelToken,
          options: Options(
            headers: requestHeaders,
            responseType: ResponseType.stream,
            followRedirects: false,
            validateStatus: (_) => true,
          ),
        );
      } on DioException catch (e) {
        throw _mapDio(e);
      }
      final status = response.statusCode ?? 0;
      if (hop == 0) meta = _Meta.from(response.headers);
      if (status >= 300 && status < 400) {
        final location = response.headers.value('location');
        await response.data?.stream.drain<void>();
        if (location == null) throw const DownloadException(DownloadFailure.http, 'Redirect without a location');
        current = current.resolve(location);
        continue;
      }
      if (status == 401 || status == 403) {
        await response.data?.stream.drain<void>();
        throw const DownloadException(
          DownloadFailure.needsAuth,
          'Access denied. Accept the model licence on Hugging Face and check your access token.',
        );
      }
      if (status == 404) {
        await response.data?.stream.drain<void>();
        throw const DownloadException(DownloadFailure.notFound, 'The model file was not found at its download address.');
      }
      if (status == 416) {
        await response.data?.stream.drain<void>();
        return _Opened(416, const Stream.empty(), null, meta);
      }
      if (status != 200 && status != 206) {
        await response.data?.stream.drain<void>();
        throw DownloadException(DownloadFailure.http, 'Server returned HTTP $status');
      }
      final length = int.tryParse(response.headers.value(Headers.contentLengthHeader) ?? '');
      final body = response.data!.stream.map<List<int>>((c) => c);
      return _Opened(status, body, length, meta);
    }
    throw const DownloadException(DownloadFailure.http, 'Too many redirects');
  }

  static String _origin(Uri u) => '${u.scheme}://${u.host}:${u.port}';

  DownloadException _mapDio(DioException e) {
    if (CancelToken.isCancel(e)) throw const DownloadCancelled();
    if (e.error is FileSystemException) {
      return const DownloadException(DownloadFailure.diskFull, 'Not enough free storage on the phone');
    }
    return DownloadException(DownloadFailure.network, 'Network problem: ${e.type.name}');
  }
}

class _Opened {
  const _Opened(this.status, this.body, this.contentLength, this.meta);

  final int status;
  final Stream<List<int>> body;
  final int? contentLength;
  final _Meta meta;
}

/// Integrity hints Hugging Face puts on its first (redirect) response.
class _Meta {
  const _Meta({this.linkedSha256, this.linkedSize});

  factory _Meta.from(Headers h) {
    final etag = h.value('x-linked-etag')?.replaceAll('"', '').toLowerCase();
    final sha = etag != null && RegExp(r'^[0-9a-f]{64}$').hasMatch(etag) ? etag : null;
    return _Meta(linkedSha256: sha, linkedSize: int.tryParse(h.value('x-linked-size') ?? ''));
  }

  final String? linkedSha256;
  final int? linkedSize;
}
