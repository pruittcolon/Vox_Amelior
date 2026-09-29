import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/idle_timeout.dart';

enum ProbeStatus { testing, passed, failed }

/// Progress of the phone test, one size at a time.
class ProbeStep {
  const ProbeStep(this.size, this.status, [this.detail = '']);

  final int size;
  final ProbeStatus status;
  final String detail;

  Map<String, Object?> toJson() => {'size': size, 'status': status.name, 'detail': detail};

  static ProbeStep? fromJson(Object? j) {
    if (j is! Map || j['size'] is! int) return null;
    return ProbeStep(
      j['size']! as int,
      ProbeStatus.values.asNameMap()[j['status']] ?? ProbeStatus.failed,
      '${j['detail'] ?? ''}',
    );
  }
}

/// What the phone test found.
class ProbeResult {
  const ProbeResult({required this.best, this.failedAt, required this.note});

  /// Largest context that worked (0 if none did).
  final int best;
  final int? failedAt;
  final String note;

  /// Recommended transcript per review part: half of [best].
  int get recommendedChunk => best ~/ 2;

  Map<String, Object?> toJson() => {'best': best, 'failedAt': failedAt, 'note': note};

  static ProbeResult? fromJson(Object? j) {
    if (j is! Map || j['best'] is! int) return null;
    return ProbeResult(best: j['best']! as int, failedAt: j['failedAt'] as int?, note: '${j['note'] ?? ''}');
  }
}

/// Finds the largest context window the assistant really handles on this
/// phone: for each size it fills the window with filler text that starts
/// with a code word, and checks the model can still repeat the code word.
/// That catches crashes, errors *and* silently dropped text.
///
/// A size can crash the whole app (native memory errors). Before each size
/// a marker file records what is being tried; if the app dies, the next
/// start reads the marker ([resultAfterCrash]) and counts that size as failed.
class ContextProbe {
  ContextProbe({
    required this.llm,
    required this.markerFile,
    this.sizes = ContextBudget.testSizes,
    this.stepTimeout = const Duration(minutes: 8),
    int? processId,
  }) : processId = processId ?? pid;

  final LlmEngine llm;
  final File markerFile;
  final List<int> sizes;
  final Duration stepTimeout;
  final int processId;

  static const int _replyTokens = 24;

  Future<ProbeResult> run({void Function(ProbeStep step)? onStep}) async {
    final passed = <int>[];
    int? failedAt;
    var why = '';
    try {
      for (final size in sizes) {
        _writeMarker(size, passed);
        onStep?.call(ProbeStep(size, ProbeStatus.testing));
        try {
          final ok = await _try(size).timeout(stepTimeout * 2);
          if (ok) {
            passed.add(size);
            onStep?.call(ProbeStep(size, ProbeStatus.passed));
            continue;
          }
          why = 'Gemma could not read it all';
        } on TimeoutException {
          why = 'took too long';
        } on Object catch (e) {
          why = friendlyLlmError(e);
        }
        failedAt = size;
        onStep?.call(ProbeStep(size, ProbeStatus.failed, why));
        break;
      }
    } finally {
      _deleteMarker();
      // Next use reloads with whatever the user settles on.
      await llm.unload();
    }
    return _result(passed, failedAt, why);
  }

  /// After a crash during a test: the sizes that passed before it.
  static ProbeResult? resultAfterCrash(File markerFile, {int? processId}) {
    if (!markerFile.existsSync()) return null;
    try {
      final j = jsonDecode(markerFile.readAsStringSync());
      if (j is! Map) return null;
      if (j['pid'] == (processId ?? pid)) return null; // still running in this app
      final passed = (j['passed'] as List<Object?>? ?? const []).whereType<int>().toList();
      final testing = j['testing'] as int?;
      markerFile.deleteSync();
      return _result(passed, testing, 'the app closed while testing it');
    } on Object {
      markerFile.deleteSync();
      return null;
    }
  }

  static ProbeResult _result(List<int> passed, int? failedAt, String why) {
    final best = passed.isEmpty ? 0 : passed.last;
    final note = best == 0
        ? 'Even ${_fmt(failedAt ?? 0)} tokens failed ($why).'
        : failedAt == null
            ? 'Works up to ${_fmt(best)} tokens (the largest size tested).'
            : 'Works up to ${_fmt(best)} tokens. ${_fmt(failedAt)} failed: $why.';
    return ProbeResult(best: best, failedAt: failedAt, note: note);
  }

  static String _fmt(int n) => n.toString().replaceAllMapped(RegExp(r'\B(?=(\d{3})+$)'), (_) => ',');

  Future<bool> _try(int size) async {
    final session = await llm.openSession(
      system: 'You are a careful reader. Reply with the code word only.',
      maxReplyTokens: _replyTokens,
      contextTokens: size,
    );
    try {
      final rng = math.Random();
      const words = ['MAPLE', 'RIVER', 'COMET', 'TULIP', 'ORBIT', 'CEDAR', 'EMBER', 'LOTUS'];
      final code = '${words[rng.nextInt(words.length)]}-${1000 + rng.nextInt(9000)}';
      final header = 'Remember this code word: $code\nBelow is filler text. Ignore it.\n';
      const footer = '\nWhat is the code word from the very first line? Reply with the code word only.';
      // Room for the chat template, the system line and the reply.
      final target = size - _replyTokens - 96;
      final block = StringBuffer();
      for (var i = 1000; i < 1020; i++) {
        block.writeln(_filler(i));
      }
      final perLine = ((await session.countTokens(block.toString())) ?? (block.length / 4).ceil()) / 20;
      final fixed = (await session.countTokens(header + footer)) ?? ((header.length + footer.length) / 4).ceil();
      var lines = math.max(1, ((target - fixed) / perLine).floor());
      String build(int n) {
        final b = StringBuffer(header);
        for (var i = 1; i <= n; i++) {
          b.writeln(_filler(i));
        }
        return (b..write(footer)).toString();
      }

      var prompt = build(lines);
      // Measure the real prompt and trim until it fits.
      for (var round = 0; round < 4; round++) {
        final tokens = await session.countTokens(prompt);
        if (tokens == null || tokens <= target) break;
        lines = math.max(1, lines - ((tokens - target) / perLine).ceil() - 1);
        prompt = build(lines);
      }
      final reply = StringBuffer();
      await for (final e in withIdleTimeout(session.send(prompt), stepTimeout)) {
        if (e is LlmText) reply.write(e.text);
      }
      String norm(String s) => s.toUpperCase().replaceAll(RegExp('[^A-Z0-9]'), '');
      return norm(reply.toString()).contains(norm(code));
    } finally {
      await session.close();
    }
  }

  static String _filler(int i) => 'Note $i: the kettle clicked off and the afternoon stayed quiet.';

  void _writeMarker(int size, List<int> passed) {
    try {
      markerFile.parent.createSync(recursive: true);
      markerFile.writeAsStringSync(jsonEncode({'pid': processId, 'testing': size, 'passed': passed}), flush: true);
    } on Object {
      // Best effort: without the marker a crash is simply not remembered.
    }
  }

  void _deleteMarker() {
    try {
      if (markerFile.existsSync()) markerFile.deleteSync();
    } on Object {
      // ignore
    }
  }
}
