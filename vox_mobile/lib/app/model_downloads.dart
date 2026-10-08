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
  const DownloadState(
    this.status, {
    this.progress,
    this.received = 0,
    this.total = 0,
    this.error,
    this.needsToken = false,
    this.stage,
    this.step = 0,
    this.steps = 0,
    this.remaining,
  });

  final DownloadStatus status;

  /// 0..1 for the current step ([status] downloading: the download itself).
  final double? progress;
  final int received;
  final int total;
  final String? error;

  /// After the download ([DownloadStatus.unpacking]): what is happening now,
  /// e.g. "Checking the download", with [step] of [steps] (1-based).
  final String? stage;
  final int step;
  final int steps;

  /// Rough time left in the current step, once it can be estimated.
  final Duration? remaining;

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
    this.onRemoved,
    this.onBusyChanged,
    this.onProgressText,
  });

  final ModelStore store;
  final ModelInstaller installer;
  final Future<String?> Function() tokenProvider;

  /// Looks up an asset by id (includes the user's custom model).
  final ModelAsset? Function(String id) resolve;
  final Future<void> Function(ModelAsset asset)? onInstalled;

  /// Called after [asset]'s files were deleted (e.g. to stop using it).
  final void Function(ModelAsset asset)? onRemoved;
  final Future<void> Function(bool busy)? onBusyChanged;
  final void Function(String text)? onProgressText;

  static const _wantedKey = 'wanted_downloads';

  final Map<String, DownloadState> _states = {};
  final List<String> _queue = [];

  /// Downloads being cancelled whose files are deleted once they have stopped.
  final Set<String> _discard = {};
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

  bool get speechReady => ModelCatalog.speechReady(store.isInstalled);

  /// Queues [asset] for download (no-op if installed or already queued).
  Future<void> download(ModelAsset asset) async {
    if (store.isInstalled(asset) || _queue.contains(asset.id)) return;
    if (_active == asset.id) {
      // Asked for again while a delete was stopping it: keep the files and start again after.
      if (!_discard.remove(asset.id)) return;
    }
    _queue.add(asset.id);
    _states[asset.id] = DownloadState(DownloadStatus.queued, total: asset.approxDownloadBytes);
    await _remember();
    notifyListeners();
    unawaited(_pump());
  }

  Future<void> downloadSpeechModels() async {
    for (final a in [...ModelCatalog.speech, ...ModelCatalog.speechExtras]) {
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

  /// Deletes [asset] (installed or partly downloaded). A download in
  /// progress is stopped first and its files deleted once it has let go of them.
  void remove(ModelAsset asset) {
    if (_active == asset.id) {
      _discard.add(asset.id);
      _cancel?.cancel();
      return;
    }
    cancel(asset);
    _delete(asset);
  }

  void _delete(ModelAsset asset) {
    store.remove(asset);
    _states.remove(asset.id);
    notifyListeners();
    onRemoved?.call(asset);
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
      final archived = asset.files.any((f) => f.isArchive);
      final steps = archived ? 4 : 2;
      InstallPhase? phase;
      var phaseStart = DateTime.now();
      await installer.install(
        asset,
        token: token,
        cancelToken: _cancel,
        onProgress: (p) {
          final now = DateTime.now();
          if (p.phase != phase) {
            phase = p.phase;
            phaseStart = now;
          }
          final (step, stage) = _describe(p);
          final unpacking = p.phase != InstallPhase.downloading;
          final state = DownloadState(
            unpacking ? DownloadStatus.unpacking : DownloadStatus.downloading,
            progress: p.fraction,
            received: p.receivedBytes,
            total: p.totalBytes,
            stage: stage,
            step: step,
            steps: steps,
            remaining: _remaining(p.fraction, now.difference(phaseStart)),
          );
          _set(asset, state);
          if (now.difference(lastText).inSeconds >= 5 || (unpacking && lastText.isBefore(phaseStart))) {
            lastText = now;
            final pct = p.fraction == null ? '' : ' ${(p.fraction! * 100).toStringAsFixed(0)}%';
            final of = step > 0 ? ' (step $step of $steps)' : '';
            onProgressText?.call(unpacking ? '${asset.title}: $stage$pct$of' : '${asset.title}$pct');
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
      if (_discard.remove(id)) _delete(asset);
      await _remember();
      notifyListeners();
      unawaited(_pump());
    }
  }

  /// Step number and a short description of what [p] is doing.
  static (int, String) _describe(InstallProgress p) => switch (p.phase) {
        InstallPhase.downloading => (1, 'Downloading'),
        InstallPhase.verifying => (2, 'Checking the download for errors'),
        InstallPhase.unpacking => (3, 'Unpacking the model'),
        InstallPhase.copying => (4, p.fileName == null ? 'Copying model files' : 'Copying ${p.fileName}'),
        InstallPhase.finishing => (0, 'Saving'),
      };

  /// Time left in a step from how long the part done so far took.
  static Duration? _remaining(double? fraction, Duration elapsed) {
    if (fraction == null || fraction < 0.02 || fraction >= 1 || elapsed.inSeconds < 3) return null;
    return elapsed * ((1 - fraction) / fraction);
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
