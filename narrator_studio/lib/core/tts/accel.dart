/// Measures what the phone's AI chip (Hexagon NPU) and GPU can do for the
/// lifelike voices, next to the CPU: each piece of VibeVoice on each
/// processor, with its time and whether the numbers agree. The report is
/// plain text, saved to Downloads so it can be shared.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import 'ort.dart';

/// QNN options for the Hexagon NPU: float16 arithmetic, full speed.
const npuOptions = {
  'backend_path': 'libQnnHtp.so',
  'htp_performance_mode': 'burst',
  'enable_htp_fp16_precision': '1',
  'htp_graph_finalization_optimization_mode': '3',
};

const gpuOptions = {'backend_path': 'libQnnGpu.so'};

/// For `compute`: the report for the VibeVoice pack in [dir].
String accelReportFor(String dir) => accelReport(dir);

/// Runs every test; never throws (failures are part of the report).
String accelReport(String vibevoiceDir, {int threads = 6}) {
  final out = StringBuffer();
  void line(String s) => out.writeln(s);
  line('Narrator Studio: AI chip test, ${DateTime.now().toIso8601String()}');
  line('cores: ${Platform.numberOfProcessors}, threads: $threads');
  try {
    line('speech runtime: ONNX Runtime ${Ort.instance.version} ${Ort.instance.providers}');
  } catch (e) {
    line('speech runtime: $e');
  }
  Ort? qnn;
  try {
    qnn = Ort.qnn;
    line('AI-chip runtime: ONNX Runtime ${qnn.version} ${qnn.providers}');
  } catch (e) {
    line('AI-chip runtime: not available ($e)');
  }
  final env = Platform.environment['ADSP_LIBRARY_PATH'];
  line('ADSP_LIBRARY_PATH: ${env ?? '(unset)'}');
  line('');

  final rng = math.Random(7);
  Float32List rand(int n, [double scale = 1]) =>
      Float32List.fromList([for (var i = 0; i < n; i++) (rng.nextDouble() * 2 - 1) * scale]);

  // ---- The diffusion head (static: batch 2), 5 runs per speech frame.
  final head = p.join(vibevoiceDir, 'head.onnx');
  final noisy = rand(2 * 64), t = Float32List.fromList([799, 799]), cond = rand(2 * 896, 3);
  Float32List? reference;
  final runs = <String, (Ort?, String?, Map<String, String>, bool)>{
    'CPU (speech runtime)': (Ort.instance, null, const {}, false),
    if (qnn != null) 'CPU (AI-chip runtime)': (qnn, null, const {}, false),
    if (qnn != null) 'NPU only (fp16)': (qnn, 'QNN', npuOptions, true),
    if (qnn != null) 'NPU + CPU fallback': (qnn, 'QNN', npuOptions, false),
    if (qnn != null) 'GPU only': (qnn, 'QNN', gpuOptions, true),
    if (qnn != null) 'GPU + CPU fallback': (qnn, 'QNN', gpuOptions, false),
  };
  line('Diffusion head (42M parameters; 5 runs per speech frame):');
  for (final e in runs.entries) {
    final (ort, provider, options, strict) = e.value;
    final watch = Stopwatch()..start();
    OrtSession? s;
    try {
      s = ort!.open(head, threads: threads, dims: const {'b': 2}, provider: provider, options: options, strict: strict);
      final openMs = watch.elapsedMilliseconds;
      final ins = [s.tensor(noisy, [2, 64]), s.tensor(t, [2]), s.tensor(cond, [2, 896])];
      Float32List? v;
      for (var i = 0; i < 3; i++) {
        final o = s.run(ins);
        v = Float32List.fromList(o.first.data);
        for (final x in o) {
          x.free();
        }
      }
      watch.reset();
      const n = 20;
      for (var i = 0; i < n; i++) {
        for (final x in s.run(ins)) {
          x.free();
        }
      }
      final ms = watch.elapsedMicroseconds / 1000 / n;
      for (final x in ins) {
        x.free();
      }
      reference ??= v;
      var err = 0.0, mag = 1e-9;
      for (var i = 0; i < v!.length; i++) {
        err = math.max(err, (v[i] - reference![i]).abs());
        mag = math.max(mag, reference[i].abs());
      }
      line('  ${e.key}: ${ms.toStringAsFixed(2)} ms per run (opened in $openMs ms), '
          'differs from CPU by ${(100 * err / mag).toStringAsFixed(2)}%');
    } catch (err) {
      line('  ${e.key}: FAILED: ${err.toString().replaceAll('\n', ' ')}');
    } finally {
      s?.free();
    }
  }

  // ---- One batched TTS LM step (the voice and its guidance branch).
  line('');
  line('TTS language model, one speech step (20 layers, batch 2, 400 cached positions):');
  final lm = p.join(vibevoiceDir, 'tts_lm.onnx');
  for (final e in {'CPU (speech runtime)': Ort.instance, if (qnn != null) 'CPU (AI-chip runtime)': qnn}.entries) {
    OrtSession? s;
    try {
      final watch = Stopwatch()..start();
      s = e.value.open(lm, threads: threads);
      final openMs = watch.elapsedMilliseconds;
      const past = 400;
      final x = s.tensor(rand(2 * 896, 0.1), [2, 1, 896]);
      final pos = s.tensor(Float32List.fromList([400, 300]), [2, 1]);
      final keep = s.tensor(Float32List.fromList(List.filled(2 * (past + 1), 1.0)), [2, past + 1]);
      final kv = [for (var i = 0; i < 40; i++) s.tensor(rand(2 * 2 * past * 64, 0.5), [2, 2, past, 64])];
      for (final o in s.run([x, pos, keep, ...kv])) {
        o.free();
      }
      watch.reset();
      const n = 5;
      for (var i = 0; i < n; i++) {
        for (final o in s.run([x, pos, keep, ...kv])) {
          o.free();
        }
      }
      line('  ${e.key}: ${(watch.elapsedMicroseconds / 1000 / n).toStringAsFixed(1)} ms per step (opened in $openMs ms)');
      for (final t in [x, pos, keep, ...kv]) {
        t.free();
      }
    } catch (err) {
      line('  ${e.key}: FAILED: ${err.toString().replaceAll('\n', ' ')}');
    } finally {
      s?.free();
    }
  }
  line('');
  line('(A speech frame is 0.133 s of audio: 5 head runs + 1 language-model step.)');
  return out.toString();
}
