/// Runs the sherpa-onnx voices in a background isolate so the screen never
/// stutters while a sentence is being generated.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as so;

import '../render/audio.dart';
import 'synth.dart';
import 'voices.dart';

/// Where each engine's files live and how hard to work.
class EngineSetup {
  const EngineSetup({
    required this.kokoroDir,
    required this.supertonicDir,
    required this.pocketDir,
    required this.kokoroVoicesFile,
    this.threads = 4,
  });

  final String kokoroDir;
  final String supertonicDir;
  final String pocketDir;

  /// voices.bin, or the app's copy with voice mixes written into it.
  final String kokoroVoicesFile;
  final int threads;
}

class SherpaSynth implements Synth {
  SherpaSynth(this.setup);

  EngineSetup Function() setup;
  Isolate? _isolate;
  SendPort? _send;
  ReceivePort? _rx;
  Future<void>? _starting;
  final _pending = <int, Completer<Map<Object?, Object?>>>{};
  var _next = 0;

  Future<void> _ensure() {
    if (_send != null) return Future.value();
    return _starting ??= () async {
      final rx = ReceivePort();
      _rx = rx;
      final ready = Completer<SendPort>();
      rx.listen((msg) {
        if (msg is SendPort) {
          ready.complete(msg);
        } else if (msg is Map) {
          _pending.remove(msg['id'])?.complete(msg);
        }
      });
      _isolate = await Isolate.spawn(_workerMain, rx.sendPort, debugName: 'speech');
      _send = await ready.future;
    }();
  }

  Future<Map<Object?, Object?>> _call(Map<String, Object?> msg) async {
    await _ensure();
    final id = _next++;
    final c = Completer<Map<Object?, Object?>>();
    _pending[id] = c;
    final s = setup();
    _send!.send({
      ...msg,
      'id': id,
      'kokoroDir': s.kokoroDir,
      'supertonicDir': s.supertonicDir,
      'pocketDir': s.pocketDir,
      'kokoroVoices': s.kokoroVoicesFile,
      'threads': s.threads,
    });
    final r = await c.future;
    if (r['ok'] != true) throw SynthError(r['error']?.toString() ?? 'Speech failed');
    return r;
  }

  @override
  Future<Float32List> speak(Voice voice, String text, {double speed = 1.0}) async {
    final r = await _call({
      'op': 'speak',
      'engine': voice.engine.name,
      'sid': voice.sid,
      // British voices get British pronunciation from espeak-ng, whose British
      // English voice is called "en" (gmw/en); "en-gb" is not a voice name.
      'lang': voice.engine == Engine.kokoro && voice.id.startsWith('kokoro:b') ? 'en' : '',
      'ref': voice.refPath,
      'speed': speed,
      'text': text,
    });
    final data = (r['pcm']! as TransferableTypedData).materialize();
    return data.asFloat32List();
  }

  @override
  Future<void> unload() async {
    if (_send == null) return;
    await _call({'op': 'unload'});
  }

  @override
  Future<void> reloadKokoroVoices() async {
    if (_send == null) return;
    await _call({'op': 'unloadKokoro'});
  }

  Future<void> dispose() async {
    _isolate?.kill(priority: Isolate.immediate);
    _rx?.close();
    _isolate = null;
    _send = null;
    _starting = null;
    for (final c in _pending.values) {
      c.completeError(SynthError('Speech engine stopped'));
    }
    _pending.clear();
  }
}

class SynthError implements Exception {
  SynthError(this.message);
  final String message;
  @override
  String toString() => message;
}

// ---------------------------------------------------------------------------
// Worker isolate.

void _workerMain(SendPort reply) {
  final rx = ReceivePort();
  reply.send(rx.sendPort);
  so.initBindings();
  final w = _Worker();
  rx.listen((raw) {
    final msg = (raw as Map).cast<String, Object?>();
    final id = msg['id'];
    try {
      switch (msg['op']) {
        case 'speak':
          final samples = w.speak(msg);
          reply.send({
            'id': id,
            'ok': true,
            'pcm': TransferableTypedData.fromList([samples.buffer.asUint8List(samples.offsetInBytes, samples.lengthInBytes)]),
          });
        case 'unload':
          w.unloadAll();
          reply.send({'id': id, 'ok': true});
        case 'unloadKokoro':
          w.unloadKokoro();
          reply.send({'id': id, 'ok': true});
        default:
          reply.send({'id': id, 'ok': false, 'error': 'unknown op ${msg['op']}'});
      }
    } catch (e) {
      reply.send({'id': id, 'ok': false, 'error': e.toString()});
    }
  });
}

class _Worker {
  so.OfflineTts? kokoro;
  String? kokoroKey;
  so.OfflineTts? supertonic;
  String? supertonicKey;
  so.OfflineTts? pocket;
  String? pocketKey;
  final refs = <String, (Float32List, int)>{};

  void unloadKokoro() {
    kokoro?.free();
    kokoro = null;
    kokoroKey = null;
  }

  void unloadAll() {
    unloadKokoro();
    supertonic?.free();
    supertonic = null;
    supertonicKey = null;
    pocket?.free();
    pocket = null;
    pocketKey = null;
    refs.clear();
  }

  static String _need(String dir, String name) {
    final path = p.join(dir, name);
    if (FileSystemEntity.typeSync(path) == FileSystemEntityType.notFound) {
      // sherpa-onnx exits the whole app on a missing file; fail politely instead.
      throw StateError('Voice files missing ($name). Re-download the voice pack.');
    }
    return path;
  }

  Float32List speak(Map<String, Object?> m) {
    final engine = m['engine']! as String;
    final text = m['text']! as String;
    final speed = (m['speed']! as num).toDouble();
    final sid = (m['sid']! as num).toInt();
    final threads = (m['threads']! as num).toInt();
    so.GeneratedAudio audio;
    switch (engine) {
      case 'kokoro':
        final dir = m['kokoroDir']! as String;
        final voicesFile = m['kokoroVoices']! as String;
        final key = '$dir|$voicesFile|$threads';
        if (kokoroKey != key) {
          unloadKokoro();
          if (!File(voicesFile).existsSync()) throw StateError('Voice table missing. Re-download Kokoro.');
          kokoro = so.OfflineTts(so.OfflineTtsConfig(
            model: so.OfflineTtsModelConfig(
              kokoro: so.OfflineTtsKokoroModelConfig(
                // The fast fp32 model, or the compact int8 one.
                model: File(p.join(dir, 'model.onnx')).existsSync()
                    ? p.join(dir, 'model.onnx')
                    : _need(dir, 'model.int8.onnx'),
                voices: voicesFile,
                tokens: _need(dir, 'tokens.txt'),
                dataDir: _need(dir, 'espeak-ng-data'),
                lexicon: _need(dir, 'lexicon-us-en.txt'),
              ),
              numThreads: threads,
              debug: false,
            ),
            maxNumSenetences: 1,
          ));
          kokoroKey = key;
        }
        final lang = (m['lang'] as String?) ?? '';
        audio = kokoro!.generateWithConfig(
          text: text,
          config: so.OfflineTtsGenerationConfig(sid: sid, speed: speed, extra: {if (lang.isNotEmpty) 'lang': lang}),
        );
        if (audio.samples.isEmpty && lang.isNotEmpty) {
          // Never lose a sentence over an accent: fall back to the dictionary.
          audio = kokoro!.generateWithConfig(
            text: text,
            config: so.OfflineTtsGenerationConfig(sid: sid, speed: speed),
          );
        }
      case 'supertonic':
        final dir = m['supertonicDir']! as String;
        final key = '$dir|$threads';
        if (supertonicKey != key) {
          supertonic?.free();
          supertonic = so.OfflineTts(so.OfflineTtsConfig(
            model: so.OfflineTtsModelConfig(
              supertonic: so.OfflineTtsSupertonicModelConfig(
                durationPredictor: _need(dir, 'duration_predictor.int8.onnx'),
                textEncoder: _need(dir, 'text_encoder.int8.onnx'),
                vectorEstimator: _need(dir, 'vector_estimator.int8.onnx'),
                vocoder: _need(dir, 'vocoder.int8.onnx'),
                ttsJson: _need(dir, 'tts.json'),
                unicodeIndexer: _need(dir, 'unicode_indexer.bin'),
                voiceStyle: _need(dir, 'voice.bin'),
              ),
              numThreads: threads,
              debug: false,
            ),
          ));
          supertonicKey = key;
        }
        audio = supertonic!.generateWithConfig(
          text: text,
          config: so.OfflineTtsGenerationConfig(sid: sid, speed: speed, numSteps: 8, extra: const {'lang': 'en'}),
        );
      case 'pocket':
        final dir = m['pocketDir']! as String;
        final key = '$dir|$threads';
        if (pocketKey != key) {
          pocket?.free();
          pocket = so.OfflineTts(so.OfflineTtsConfig(
            model: so.OfflineTtsModelConfig(
              pocket: so.OfflineTtsPocketModelConfig(
                lmFlow: _need(dir, 'lm_flow.int8.onnx'),
                lmMain: _need(dir, 'lm_main.int8.onnx'),
                encoder: _need(dir, 'encoder.onnx'),
                decoder: _need(dir, 'decoder.int8.onnx'),
                textConditioner: _need(dir, 'text_conditioner.onnx'),
                vocabJson: _need(dir, 'vocab.json'),
                tokenScoresJson: _need(dir, 'token_scores.json'),
              ),
              numThreads: threads,
              debug: false,
            ),
          ));
          pocketKey = key;
        }
        final refPath = m['ref'] as String?;
        if (refPath == null || !File(refPath).existsSync()) {
          throw StateError('This voice\'s sample clip is missing. Download it again.');
        }
        final ref = refs.putIfAbsent(refPath, () {
          final w = so.readWave(refPath);
          return (w.samples, w.sampleRate);
        });
        audio = pocket!.generateWithConfig(
          text: text,
          config: so.OfflineTtsGenerationConfig(
            speed: speed,
            referenceAudio: ref.$1,
            referenceSampleRate: ref.$2,
            numSteps: 5,
            extra: const {'max_reference_audio_len': 12, 'seed': 7},
          ),
        );
      default:
        throw StateError('Unknown engine $engine');
    }
    var samples = audio.samples;
    if (samples.isEmpty) throw StateError('The voice produced no audio for: "$text"');
    if (audio.sampleRate != outputRate) samples = resample(samples, audio.sampleRate, outputRate);
    return samples;
  }
}
