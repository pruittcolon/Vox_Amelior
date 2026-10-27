import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import '../log.dart';
import 'parakeet_model.dart';
import 'speech.dart';

const _rate = 16000;

/// 16-bit little-endian PCM to floats in -1..1.
Float32List pcmToFloat(Uint8List bytes) {
  final n = bytes.length ~/ 2;
  final data = ByteData.sublistView(bytes);
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

/// Loudness on the scale Android's recognizer reports (about -2 quiet to
/// 10 loud), so the app's voice meter works the same with either engine.
double loudness(Float32List s) {
  if (s.isEmpty) return -2;
  var sum = 0.0;
  for (final v in s) {
    sum += v * v;
  }
  final rms = math.sqrt(sum / s.length);
  final db = 20 * math.log(rms + 1e-6) / math.ln10;
  return ((db + 60) / 5 - 2).clamp(-2.0, 10.0);
}

/// Turns sound into text.
abstract class Transcriber {
  String transcribe(Float32List samples);
}

/// Says when someone is talking and cuts out each thing they said.
abstract class SpeechDetector {
  void accept(Float32List samples);
  bool get talking;

  /// Finished stretches of speech (each ended by a pause).
  List<Float32List> take();
  void flush();
  void reset();
}

/// The live loop: words while you speak (the stretch so far, decoded again
/// whenever the last decode is done and there is enough new sound), the
/// final text when you pause.
class UtteranceLoop {
  UtteranceLoop(this.asr, this.vad, {this.partialEvery = 0.45});

  final Transcriber asr;
  final SpeechDetector vad;

  /// Seconds of new sound before the words so far are decoded again.
  final double partialEvery;

  final List<Float32List> _speech = [];
  final List<Float32List> _recent = []; // the moment before speech starts
  int _speechLen = 0;
  int _sinceDecode = 0;
  int _decodeCost = 0; // samples of sound a decode took, so they never pile up
  String _lastPartial = '';
  bool _talking = false;

  /// One chunk of sound in; events out.
  List<Map<String, Object>> onSamples(Float32List s, {Stopwatch? clock}) {
    final out = <Map<String, Object>>[
      {'type': 'level', 'value': loudness(s)},
    ];
    vad.accept(s);
    final talking = vad.talking;
    if (talking && !_talking) {
      out.add({'type': 'begin'});
      _speech
        ..clear()
        ..addAll(_recent);
      _speechLen = _recent.fold(0, (n, c) => n + c.length);
      _sinceDecode = _speechLen;
    }
    _talking = talking;
    _recent.add(s);
    if (_recent.length > 3) _recent.removeAt(0); // about 0.3 s
    if (talking) {
      _speech.add(s);
      _speechLen += s.length;
      _sinceDecode += s.length;
      final wait = math.max((partialEvery * _rate).round(), _decodeCost * 3 ~/ 2);
      if (_sinceDecode >= wait && _speechLen < 25 * _rate) {
        _sinceDecode = 0;
        final sw = clock ?? (Stopwatch()..start());
        final text = asr.transcribe(_joined()).trim();
        _decodeCost = (sw.elapsedMicroseconds * _rate / 1e6).round();
        if (text.isNotEmpty && text != _lastPartial) {
          _lastPartial = text;
          out.add({'type': 'partial', 'text': text});
        }
      }
    }
    out.addAll(_finals());
    return out;
  }

  /// End what is being said now (TouchBar let go, "send" button).
  List<Map<String, Object>> finish() {
    vad.flush();
    final out = _finals(finishing: true);
    if (out.isEmpty && _speechLen > _rate ~/ 4) {
      final text = asr.transcribe(_joined()).trim();
      if (text.isNotEmpty) out.add({'type': 'final', 'text': text, 'finishing': true});
    }
    if (out.isEmpty) out.add({'type': 'end', 'finishing': true});
    _clear();
    return out;
  }

  void reset() {
    vad.reset();
    _recent.clear();
    _clear();
  }

  List<Map<String, Object>> _finals({bool finishing = false}) {
    final out = <Map<String, Object>>[];
    for (final seg in vad.take()) {
      final text = asr.transcribe(seg).trim();
      _clear();
      if (text.isNotEmpty) out.add({'type': 'final', 'text': text, 'finishing': finishing});
    }
    return out;
  }

  void _clear() {
    _speech.clear();
    _speechLen = 0;
    _sinceDecode = 0;
    _lastPartial = '';
    _talking = false;
  }

  Float32List _joined() {
    final all = Float32List(_speechLen);
    var at = 0;
    for (final c in _speech) {
      all.setAll(at, c);
      at += c.length;
    }
    return all;
  }
}

// ------------------------------------------------------------ sherpa-onnx

class _SherpaAsr implements Transcriber {
  _SherpaAsr(String dir, int threads)
      : _rec = sherpa.OfflineRecognizer(sherpa.OfflineRecognizerConfig(
          model: sherpa.OfflineModelConfig(
            transducer: sherpa.OfflineTransducerModelConfig(
              encoder: '$dir/${ParakeetFiles.encoder}',
              decoder: '$dir/${ParakeetFiles.decoder}',
              joiner: '$dir/${ParakeetFiles.joiner}',
            ),
            tokens: '$dir/${ParakeetFiles.tokens}',
            numThreads: threads,
            debug: false,
            modelType: 'nemo_transducer',
          ),
        ));

  final sherpa.OfflineRecognizer _rec;

  @override
  String transcribe(Float32List samples) {
    final stream = _rec.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: _rate);
      _rec.decode(stream);
      return _rec.getResult(stream).text;
    } finally {
      stream.free();
    }
  }
}

class _SherpaVad implements SpeechDetector {
  _SherpaVad(this.model, double minSilence) : _vad = _make(model, minSilence);

  final String model;
  sherpa.VoiceActivityDetector _vad;

  static sherpa.VoiceActivityDetector _make(String model, double minSilence) => sherpa.VoiceActivityDetector(
        config: sherpa.VadModelConfig(
          sileroVad: sherpa.SileroVadModelConfig(
            model: model,
            threshold: 0.5,
            minSilenceDuration: minSilence,
            minSpeechDuration: 0.25,
            maxSpeechDuration: 20,
          ),
          sampleRate: _rate,
          numThreads: 1,
          debug: false,
        ),
        bufferSizeInSeconds: 60,
      );

  void setSilence(double s) {
    _vad.free();
    _vad = _make(model, s);
  }

  @override
  void accept(Float32List samples) => _vad.acceptWaveform(samples);

  @override
  bool get talking => _vad.isDetected();

  @override
  List<Float32List> take() {
    final out = <Float32List>[];
    while (!_vad.isEmpty()) {
      out.add(_vad.front().samples);
      _vad.pop();
    }
    return out;
  }

  @override
  void flush() => _vad.flush();

  @override
  void reset() => _vad.reset();
}

/// Runs in its own isolate, so decoding never stalls the app.
@pragma('vm:entry-point')
void _worker(List<Object?> args) {
  final out = args[0] as SendPort;
  final dir = args[1] as String;
  final threads = args[2] as int;
  final inbox = ReceivePort();
  out.send(inbox.sendPort);
  late final _SherpaVad vad;
  late final UtteranceLoop loop;
  var silence = 0.8;
  try {
    final sw = Stopwatch()..start();
    sherpa.initBindings();
    final asr = _SherpaAsr(dir, threads);
    vad = _SherpaVad('$dir/${ParakeetFiles.vad}', silence);
    loop = UtteranceLoop(asr, vad);
    out.send({'type': 'loaded', 'ms': sw.elapsedMilliseconds});
  } catch (e) {
    out.send({'type': 'error', 'message': 'Parakeet did not load: $e', 'fatal': true});
    return;
  }
  inbox.listen((m) {
    try {
      if (m is TransferableTypedData) {
        for (final e in loop.onSamples(pcmToFloat(m.materialize().asUint8List()))) {
          out.send(e);
        }
      } else if (m is Map) {
        switch (m['cmd']) {
          case 'reset':
            final s = (m['silence'] as num?)?.toDouble() ?? silence;
            if (s != silence) {
              silence = s;
              vad.setSilence(s);
            }
            loop.reset();
          case 'finish':
            for (final e in loop.finish()) {
              out.send(e);
            }
          case 'quit':
            inbox.close();
        }
      }
    } catch (e) {
      out.send({'type': 'error', 'message': '$e', 'fatal': false});
    }
  });
}

/// Parakeet (NVIDIA NeMo) on the phone, through sherpa-onnx: the phone's or
/// the glasses' microphone, words while you speak, the final text at each
/// pause. Same events as Android's recognizer.
class ParakeetSpeech extends ChangeNotifier implements SpeechEngine {
  ParakeetSpeech(this.model, {this.threads = 4});

  final SpeechModel model;
  final int threads;

  static const _pcm = MethodChannel('suite/pcm');
  static final Stream<dynamic> _pcmEvents = const EventChannel('suite/pcmEvents').receiveBroadcastStream();

  final _events = StreamController<SpeechEvent>.broadcast();
  StreamSubscription<dynamic>? _pcmSub;
  Isolate? _isolate;
  ReceivePort? _port;
  SendPort? _toWorker;
  Completer<void>? _loading;
  bool _continuous = true;
  bool _running = false;

  /// 'off', 'loading', 'ready' or an error.
  String _status = 'off';
  String get status => _status;
  set status(String v) {
    if (v == _status) return;
    _status = v;
    notifyListeners();
  }

  @override
  Stream<SpeechEvent> get events => _events.stream;

  @override
  Future<bool> available() async => model.ready;

  /// Load the model now (takes a few seconds), so the first word is fast.
  Future<void> warm() => _ensureWorker();

  Future<void> _ensureWorker() async {
    final pending = _loading;
    if (pending != null) return pending.future;
    final dir = model.dir;
    if (!model.ready || dir == null) throw 'Download Parakeet first';
    final loading = _loading = Completer<void>();
    status = 'loading';
    final port = _port = ReceivePort();
    port.listen((m) {
      if (m is SendPort) {
        _toWorker = m;
      } else if (m is Map) {
        _fromWorker(m, loading);
      }
    });
    try {
      _isolate = await Isolate.spawn(_worker, [port.sendPort, dir, threads]);
    } catch (e) {
      _shutdown();
      status = 'Parakeet did not start: $e';
      rethrow;
    }
    return loading.future;
  }

  void _fromWorker(Map m, Completer<void> loading) {
    switch (m['type']) {
      case 'loaded':
        status = 'ready';
        EventLog.add('speech', 'parakeetLoaded', {'ms': m['ms']});
        if (!loading.isCompleted) loading.complete();
      case 'error':
        final fatal = m['fatal'] == true;
        if (fatal) {
          status = '${m['message']}';
          if (!loading.isCompleted) loading.completeError(status);
          _shutdown();
        }
        _events.add(SpeechEvent('error', message: '${m['message']}', fatal: fatal));
      case 'level':
        if (_running) _events.add(SpeechEvent('level', level: (m['value'] as num).toDouble()));
      case 'begin':
        if (_running) _events.add(const SpeechEvent('begin'));
      case 'partial':
        if (_running) _events.add(SpeechEvent('partial', text: '${m['text']}'));
      case 'final':
        // After stop() only what finish() asked for still counts.
        if (!_running && m['finishing'] != true) return;
        _events.add(SpeechEvent('final', text: '${m['text']}', finishing: m['finishing'] == true));
        _events.add(SpeechEvent('end', finishing: m['finishing'] == true));
        if (!_continuous) unawaited(stop());
      case 'end':
        _events.add(SpeechEvent('end', finishing: m['finishing'] == true));
    }
  }

  @override
  Future<void> start({
    bool continuous = true,
    String language = '',
    bool preferOffline = true,
    bool quiet = true,
    String engine = 'auto',
    String audioSource = 'phone',
    int silenceMs = 1500,
  }) async {
    _continuous = continuous;
    try {
      await _ensureWorker();
    } catch (e) {
      _events.add(SpeechEvent('error', message: '$e', fatal: true));
      return;
    }
    // A shorter pause than Android needs: Parakeet is fast, and this is
    // what makes a TouchBar question go the moment you stop.
    _toWorker?.send({'cmd': 'reset', 'silence': (silenceMs * 0.55 / 1000).clamp(0.4, 1.2)});
    _pcmSub ??= _pcmEvents.listen((d) {
      if (_running && d is Uint8List) _toWorker?.send(TransferableTypedData.fromList([d]));
    });
    _running = true;
    final src = await _pcm.invokeMethod<String>('start', {'source': audioSource});
    if (src != 'phone' && src != 'glasses') {
      _running = false;
      _events.add(SpeechEvent('error',
          message: src == 'no-permission' ? 'Allow the microphone' : 'The microphone did not start', fatal: true));
      return;
    }
    _events.add(SpeechEvent('source', glassesMic: src == 'glasses'));
    _events.add(const SpeechEvent('state', listening: true));
    _events.add(const SpeechEvent('ready'));
  }

  @override
  Future<void> stop() async {
    if (!_running) return;
    _running = false;
    try {
      await _pcm.invokeMethod('stop');
    } catch (_) {}
    _toWorker?.send({'cmd': 'reset'});
    _events.add(const SpeechEvent('state', listening: false));
  }

  @override
  Future<void> finish() async => _toWorker?.send({'cmd': 'finish'});

  /// Unload the model (frees its memory), e.g. when switching back to Android.
  Future<void> unload() async {
    await stop();
    _shutdown();
  }

  void _shutdown() {
    _toWorker?.send({'cmd': 'quit'});
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _port?.close();
    _port = null;
    _toWorker = null;
    _loading = null;
    if (status == 'ready' || status == 'loading') status = 'off';
  }
}

/// The recognizer in use: Android's, or Parakeet once it is downloaded and
/// picked. Events come from whichever one was started last.
class SwitchableSpeech implements SpeechEngine {
  SwitchableSpeech(this.android, [this.parakeet]) {
    android.events.listen((e) {
      if (identical(_current, android)) _out.add(e);
    });
    parakeet?.events.listen((e) {
      if (identical(_current, parakeet)) _out.add(e);
    });
  }

  final SpeechEngine android;
  final SpeechEngine? parakeet;
  final _out = StreamController<SpeechEvent>.broadcast();
  late SpeechEngine _current = android;

  bool useParakeet = false;

  SpeechEngine get _wanted => useParakeet && parakeet != null ? parakeet! : android;

  /// The engine handling speech right now.
  SpeechEngine get current => _current;

  @override
  Stream<SpeechEvent> get events => _out.stream;

  @override
  Future<bool> available() => _wanted.available();

  @override
  Future<void> start({
    bool continuous = true,
    String language = '',
    bool preferOffline = true,
    bool quiet = true,
    String engine = 'auto',
    String audioSource = 'phone',
    int silenceMs = 1500,
  }) async {
    final next = _wanted;
    if (!identical(next, _current)) {
      try {
        await _current.stop();
      } catch (_) {}
      _current = next;
    }
    await next.start(
        continuous: continuous,
        language: language,
        preferOffline: preferOffline,
        quiet: quiet,
        engine: engine,
        audioSource: audioSource,
        silenceMs: silenceMs);
  }

  @override
  Future<void> stop() => _current.stop();

  @override
  Future<void> finish() => _current.finish();
}
