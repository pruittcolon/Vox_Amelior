/// Renders a whole document to an audio file (M4A for podcasts, or WAV).
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import '../app_state.dart';
import '../render/audio.dart';
import '../render/renderer.dart';
import '../render/wav.dart';
import 'native.dart';

enum ExportState { idle, rendering, encoding, done, failed, cancelled }

class Exporter extends ChangeNotifier {
  Exporter(this.app, this.native);

  final AppState app;
  final NativeApi native;

  ExportState state = ExportState.idle;
  double progress = 0;
  String? output;
  String? error;
  String? title;
  Duration length = Duration.zero;
  final _watch = Stopwatch();
  bool _cancel = false;

  bool get busy => state == ExportState.rendering || state == ExportState.encoding;

  /// Time left, from the pace so far.
  Duration? get eta {
    if (progress < 0.03 || !busy) return null;
    final elapsed = _watch.elapsedMilliseconds;
    return Duration(milliseconds: (elapsed / progress * (1 - progress)).round());
  }

  void cancel() => _cancel = true;

  void reset() {
    if (busy) return;
    state = ExportState.idle;
    progress = 0;
    output = null;
    error = null;
    notifyListeners();
  }

  Future<String?> export(Session s, {String format = 'm4a'}) async {
    if (busy) return null;
    _cancel = false;
    state = ExportState.rendering;
    progress = 0;
    output = null;
    error = null;
    title = s.doc.title;
    _watch
      ..reset()
      ..start();
    notifyListeners();
    await native.keepScreenOn(true);
    final name = safeFileName(s.doc.title);
    final wavPath = p.join(app.cacheDir.path, 'export-$name.wav');
    WavWriter? wav;
    try {
      await _notify(s, 0);
      wav = await WavWriter.create(wavPath, outputRate);
      final r = Renderer(plan: s.plan, synth: app.synth, voiceFor: s.voiceFor, theme: s.theme);
      final total = s.plan.utterances.length;
      var lastShown = -1;
      while (true) {
        if (_cancel) throw const _Cancelled();
        final chunk = await r.next();
        if (chunk == null) break;
        await wav.add(chunk.samples);
        if (chunk.utterance != null && total > 0) {
          progress = (chunk.utterance! + 1) / total * (format == 'm4a' ? 0.92 : 1.0);
          final pct = (progress * 100).floor();
          if (pct != lastShown) {
            lastShown = pct;
            notifyListeners();
            if (pct % 5 == 0) await _notify(s, pct);
          }
        }
      }
      await wav.close();
      length = Duration(milliseconds: (wav.seconds * 1000).round());
      wav = null;
      await app.exportsDir.create(recursive: true);
      String out;
      if (format == 'wav') {
        out = p.join(app.exportsDir.path, '$name.wav');
        await File(wavPath).rename(out);
      } else {
        state = ExportState.encoding;
        notifyListeners();
        out = p.join(app.exportsDir.path, '$name.m4a');
        final sub = native.encodeProgress.listen((x) {
          progress = 0.92 + 0.08 * x;
          notifyListeners();
        });
        try {
          await native.encodeM4a(wavPath, out);
        } finally {
          await sub.cancel();
          final tmp = File(wavPath);
          if (tmp.existsSync()) await tmp.delete();
        }
      }
      output = out;
      progress = 1;
      state = ExportState.done;
      return out;
    } on _Cancelled {
      state = ExportState.cancelled;
      return null;
    } catch (e) {
      error = e.toString();
      state = ExportState.failed;
      return null;
    } finally {
      _watch.stop();
      if (wav != null) {
        try {
          await wav.close();
        } catch (_) {}
      }
      if (state != ExportState.done) {
        final tmp = File(wavPath);
        if (tmp.existsSync()) await tmp.delete();
      }
      await native.keepScreenOn(false);
      try {
        await native.serviceStop();
      } catch (_) {}
      notifyListeners();
    }
  }

  Future<void> _notify(Session s, int pct) async {
    try {
      await native.serviceShow(kind: 'export', title: 'Exporting ${s.doc.title}', text: '$pct% done', playing: true);
    } catch (_) {}
  }
}

class _Cancelled implements Exception {
  const _Cancelled();
}

String safeFileName(String title) {
  final s = title.replaceAll(RegExp(r'[^\w\- ]+'), '').trim().replaceAll(RegExp(r'\s+'), ' ');
  final cut = s.length > 60 ? s.substring(0, 60).trim() : s;
  return cut.isEmpty ? 'narration' : cut;
}
