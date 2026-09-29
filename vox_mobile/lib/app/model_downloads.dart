import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_installer.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/models/resumable_downloader.dart';

enum DownloadStatus { notInstalled, downloading, unpacking, installed, failed }

class DownloadState {
  const DownloadState(this.status, {this.progress, this.received = 0, this.total = 0, this.error, this.needsToken = false});

  final DownloadStatus status;
  final double? progress;
  final int received;
  final int total;
  final String? error;

  /// The failure was an access problem the user can fix with a token.
  final bool needsToken;

  bool get isBusy => status == DownloadStatus.downloading || status == DownloadStatus.unpacking;
}

/// Tracks model downloads for the UI. Downloads survive screen changes and
/// resume where they stopped if the app is closed.
class ModelDownloads extends ChangeNotifier {
  ModelDownloads({required this.store, required this.installer, required this.tokenProvider, this.onInstalled});

  final ModelStore store;
  final ModelInstaller installer;
  final Future<String?> Function() tokenProvider;

  /// Called after a model finishes installing.
  final Future<void> Function(ModelAsset asset)? onInstalled;

  final Map<String, DownloadState> _states = {};
  final Map<String, CancelToken> _cancels = {};

  DownloadState stateOf(ModelAsset asset) {
    final s = _states[asset.id];
    if (s != null && (s.isBusy || s.status == DownloadStatus.failed)) return s;
    if (store.isInstalled(asset)) return const DownloadState(DownloadStatus.installed);
    final partial = store.partialBytes(asset);
    return DownloadState(DownloadStatus.notInstalled, received: partial, total: asset.approxDownloadBytes);
  }

  bool isInstalled(ModelAsset asset) => store.isInstalled(asset);

  bool get speechReady =>
      ModelCatalog.all.where((m) => m.essential).every(store.isInstalled);

  Future<void> download(ModelAsset asset) async {
    if (stateOf(asset).isBusy) return;
    final cancel = CancelToken();
    _cancels[asset.id] = cancel;
    _set(asset, DownloadState(DownloadStatus.downloading, progress: 0, total: asset.approxDownloadBytes));
    try {
      final token = asset.requiresToken ? await tokenProvider() : null;
      await installer.install(
        asset,
        token: token,
        cancelToken: cancel,
        onProgress: (p) => _set(
          asset,
          DownloadState(
            p.phase == InstallPhase.downloading ? DownloadStatus.downloading : DownloadStatus.unpacking,
            progress: p.fraction,
            received: p.receivedBytes,
            total: p.totalBytes,
          ),
        ),
      );
      _states.remove(asset.id);
      await onInstalled?.call(asset);
      notifyListeners();
    } on DownloadCancelled {
      _states.remove(asset.id);
      notifyListeners();
    } on DownloadException catch (e) {
      _set(asset, DownloadState(DownloadStatus.failed, error: e.message, needsToken: e.failure == DownloadFailure.needsAuth));
    } on Object catch (e, st) {
      Log.e('models', 'install failed for ${asset.id}', e, st);
      _set(asset, DownloadState(DownloadStatus.failed, error: 'Could not install: $e'));
    } finally {
      _cancels.remove(asset.id);
    }
  }

  /// Downloads every model needed for listening, one after another.
  Future<void> downloadSpeechModels() async {
    for (final asset in ModelCatalog.all.where((m) => m.essential)) {
      if (!store.isInstalled(asset)) await download(asset);
      if (stateOf(asset).status == DownloadStatus.failed) return;
    }
  }

  void cancel(ModelAsset asset) => _cancels[asset.id]?.cancel();

  void remove(ModelAsset asset) {
    cancel(asset);
    store.remove(asset);
    _states.remove(asset.id);
    notifyListeners();
  }

  void _set(ModelAsset asset, DownloadState state) {
    _states[asset.id] = state;
    notifyListeners();
  }
}
