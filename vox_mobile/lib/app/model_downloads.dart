import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_installer.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/models/resumable_downloader.dart';

enum DownloadStatus { notInstalled, queued, downloading, unpacking, installed, failed }

class DownloadState {
  const DownloadState(this.status, {this.progress, this.received = 0, this.total = 0, this.error, this.needsToken = false});

  final DownloadStatus status;
  final double? progress;
  final int received;
  final int total;
  final String? error;

  /// The failure was an access problem the user can fix with a token.
  final bool needsToken;

  bool get isBusy =>
      status == DownloadStatus.downloading || status == DownloadStatus.unpacking || status == DownloadStatus.queued;
}

/// Tracks model downloads for the whole app (not tied to any screen).
///
/// Downloads run one at a time, continue while the app is in the background
/// ([onBusyChanged] keeps a foreground service alive), survive restarts
/// (wanted downloads are remembered and resumed), and pick up exactly where
/// they stopped.
class ModelDownloads extends ChangeNotifier {
  ModelDownloads({
    required this.store,
    required this.installer,
    required this.tokenProvider,
    required this.resolve,
    this.onInstalled,
    this.onBusyChanged,
    this.onProgressText,
  });

  final ModelStore store;
  final ModelInstaller installer;
  final Future<String?> Function() tokenProvider;

  /// Looks up an asset by id (includes the user's custom model).
  final ModelAsset? Function(String id) resolve;
  final Future<void> Function(ModelAsset asset)? onInstalled;
  final Future<void> Function(bool busy)? onBusyChanged;
  final void Function(String text)? onProgressText;

  static const _wantedKey = 'wanted_downloads';

  final Map<String, DownloadState> _states = {};
  final List<String> _queue = [];
  CancelToken? _cancel;
  String? _active;
  bool _busy = false;

  DownloadState stateOf(ModelAsset asset) {
    final s = _states[asset.id];
    if (s != null && (s.isBusy || s.status == DownloadStatus.failed)) return s;
    if (store.isInstalled(asset)) return const DownloadState(DownloadStatus.installed);
    return DownloadState(DownloadStatus.notInstalled, received: store.partialBytes(asset), total: asset.approxDownloadBytes);
  }

  bool get isBusy => _busy;

  /// The download in progress, for app-wide progress banners.
  ({ModelAsset asset, DownloadState state})? get current {
    final id = _active;
    if (id == null) return null;
    final asset = resolve(id);
    return asset == null ? null : (asset: asset, state: stateOf(asset));
  }

  bool get speechReady => ModelCatalog.speech.every(store.isInstalled);

  /// Queues [asset] for download (no-op if installed or already queued).
  Future<void> download(ModelAsset asset) async {
    if (store.isInstalled(asset) || _queue.contains(asset.id) || _active == asset.id) return;
    _queue.add(asset.id);
    _states[asset.id] = DownloadState(DownloadStatus.queued, total: asset.approxDownloadBytes);
    await _remember();
    notifyListeners();
    unawaited(_pump());
  }

  Future<void> downloadSpeechModels() async {
    for (final a in ModelCatalog.speech) {
      await download(a);
    }
  }

  /// Continues downloads that were interrupted when the app last closed.
  Future<void> resumeInterrupted() async {
    final prefs = await SharedPreferences.getInstance();
    for (final id in prefs.getStringList(_wantedKey) ?? const <String>[]) {
      final asset = resolve(id);
      if (asset != null) await download(asset);
    }
  }

  /// Pauses [asset]'s download (progress is kept) or removes it from the queue.
  void cancel(ModelAsset asset) {
    if (_active == asset.id) {
      _cancel?.cancel();
    } else if (_queue.remove(asset.id)) {
      _states.remove(asset.id);
      unawaited(_remember());
      notifyListeners();
    }
  }

  void remove(ModelAsset asset) {
    cancel(asset);
    store.remove(asset);
    _states.remove(asset.id);
    notifyListeners();
  }

  Future<void> _pump() async {
    if (_active != null) return;
    if (_queue.isEmpty) {
      await _setBusy(false);
      return;
    }
    await _setBusy(true);
    final id = _queue.removeAt(0);
    final asset = resolve(id);
    if (asset == null) {
      unawaited(_pump());
      return;
    }
    _active = id;
    _cancel = CancelToken();
    _set(asset, DownloadState(DownloadStatus.downloading, progress: 0, total: asset.approxDownloadBytes));
    try {
      // Only ever send the token to Hugging Face, never to other hosts.
      final onHf = asset.files.every((f) => Uri.tryParse(f.url)?.host.endsWith('huggingface.co') ?? false);
      final token = asset.requiresToken || onHf ? await tokenProvider() : null;
      var lastText = DateTime.fromMillisecondsSinceEpoch(0);
      await installer.install(
        asset,
        token: token,
        cancelToken: _cancel,
        onProgress: (p) {
          final unpacking = p.phase != InstallPhase.downloading;
          _set(asset, DownloadState(
            unpacking ? DownloadStatus.unpacking : DownloadStatus.downloading,
            progress: p.fraction,
            received: p.receivedBytes,
            total: p.totalBytes,
          ));
          final now = DateTime.now();
          if (now.difference(lastText).inSeconds >= 5) {
            lastText = now;
            final pct = p.fraction == null ? '' : ' ${(p.fraction! * 100).toStringAsFixed(0)}%';
            onProgressText?.call('${asset.title}$pct');
          }
        },
      );
      _states.remove(id);
      await onInstalled?.call(asset);
    } on DownloadCancelled {
      _states.remove(id);
    } on DownloadException catch (e) {
      _set(asset, DownloadState(DownloadStatus.failed, error: e.message, needsToken: e.failure == DownloadFailure.needsAuth));
    } on Object catch (e, st) {
      Log.e('models', 'install failed for $id', e, st);
      _set(asset, DownloadState(DownloadStatus.failed, error: 'Could not install: $e'));
    } finally {
      _active = null;
      _cancel = null;
      await _remember();
      notifyListeners();
      unawaited(_pump());
    }
  }

  Future<void> _setBusy(bool busy) async {
    if (_busy == busy) return;
    _busy = busy;
    notifyListeners();
    await onBusyChanged?.call(busy);
  }

  Future<void> _remember() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_wantedKey, [?_active, ..._queue]);
  }

  void _set(ModelAsset asset, DownloadState state) {
    _states[asset.id] = state;
    notifyListeners();
  }
}
