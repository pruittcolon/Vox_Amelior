import 'dart:io';

import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/models/archive_extractor.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/models/resumable_downloader.dart';

enum InstallPhase { downloading, unpacking, finishing }

class InstallProgress {
  const InstallProgress(this.phase, this.receivedBytes, this.totalBytes);

  final InstallPhase phase;
  final int receivedBytes;
  final int totalBytes;

  /// 0..1, or null when not measurable (unpacking).
  double? get fraction => phase == InstallPhase.downloading && totalBytes > 0 ? receivedBytes / totalBytes : null;
}

/// Downloads, verifies and unpacks a [ModelAsset] into the [ModelStore].
class ModelInstaller {
  ModelInstaller({
    required this.store,
    ResumableDownloader? downloader,
    this.extractor = const ArchiveExtractor(),
    this.retries = 3,
    this.retryDelay = const Duration(seconds: 3),
  }) : downloader = downloader ?? ResumableDownloader();

  final ModelStore store;
  final ResumableDownloader downloader;
  final ArchiveExtractor extractor;

  /// Automatic retries for dropped connections (progress is kept between tries).
  final int retries;
  final Duration retryDelay;

  /// Installs [asset] unless it is already complete.
  ///
  /// [token] is sent as a Bearer token (Hugging Face). Throws
  /// [DownloadException] or [DownloadCancelled].
  Future<void> install(
    ModelAsset asset, {
    String? token,
    void Function(InstallProgress progress)? onProgress,
    CancelToken? cancelToken,
  }) async {
    if (store.isInstalled(asset)) return;
    final dir = store.dir(asset)..createSync(recursive: true);
    final marker = File(p.join(dir.path, '.installed'));
    if (marker.existsSync()) marker.deleteSync(); // stale/invalid marker

    final headers = <String, String>{
      if (token != null && token.trim().isNotEmpty) 'Authorization': 'Bearer ${token.trim()}',
    };

    final totalBytes = asset.approxDownloadBytes;
    var completedBytes = 0;
    for (final file in asset.files) {
      final target = File(p.join(dir.path, file.fileName));
      final base = completedBytes;
      await _withRetries(() => downloader.download(
            url: Uri.parse(file.url),
            destination: target,
            headers: headers,
            expectedSha256: file.sha256,
            expectedSize: file.sizeBytes,
            cancelToken: cancelToken,
            onProgress: (received, total) =>
                onProgress?.call(InstallProgress(InstallPhase.downloading, base + received, totalBytes)),
          ));
      completedBytes += file.sizeBytes ?? target.lengthSync();

      if (file.isArchive) {
        onProgress?.call(InstallProgress(InstallPhase.unpacking, completedBytes, totalBytes));
        await extractor.extractTarBz2(target, dir, wanted: file.extractFromArchive);
        target.deleteSync(); // reclaim ~half a GB
      }
    }

    onProgress?.call(InstallProgress(InstallPhase.finishing, totalBytes, totalBytes));
    store.markInstalled(asset);
  }

  Future<void> _withRetries(Future<File> Function() action) async {
    for (var attempt = 0;; attempt++) {
      try {
        await action();
        return;
      } on DownloadException catch (e) {
        final transient = e.failure == DownloadFailure.network;
        if (!transient || attempt >= retries) rethrow;
        Log.w('models', 'download interrupted, retrying (${attempt + 1}/$retries)');
        await Future<void>.delayed(retryDelay);
      }
    }
  }
}
