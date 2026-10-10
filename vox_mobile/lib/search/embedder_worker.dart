import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/native/embedding_gemma.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';

/// Loads the model inside the worker. Sent to the worker isolate, so it must
/// be a top-level or static function.
typedef EmbedderLoader = TextEmbedder Function(String modelPath, String tokenizerPath);

TextEmbedder loadEmbeddingGemma(String modelPath, String tokenizerPath) =>
    EmbeddingGemmaOnnx(modelPath: modelPath, tokenizerPath: tokenizerPath);

/// Runs EmbeddingGemma in a long-lived worker isolate, so embedding never
/// blocks the UI. The model (about 400 MB in memory) is loaded on first use
/// and released after [idle] without work, or on [close].
///
/// The model's memory is native (ONNX Runtime), so it is only freed when the
/// worker disposes it: closing asks the worker to dispose and waits for it to
/// end, rather than killing it (which would leak the whole model).
class IsolateEmbedder implements AsyncEmbedder {
  IsolateEmbedder({
    required this.modelPath,
    required this.tokenizerPath,
    this.load = loadEmbeddingGemma,
    this.idle = const Duration(minutes: 2),
    this.closeTimeout = const Duration(seconds: 30),
  });

  final String modelPath;
  final String tokenizerPath;
  final EmbedderLoader load;
  final Duration idle;

  /// How long [close] waits for the worker to finish what it is doing and
  /// release the model before ending it the hard way.
  final Duration closeTimeout;

  _Worker? _current;
  Timer? _idleTimer;

  @override
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task}) async {
    if (texts.isEmpty) return const [];
    _idleTimer?.cancel();
    if (_current?.dead ?? false) _current = null; // it failed: start afresh
    final worker = _current ??= _Worker.start(load, modelPath, tokenizerPath);
    try {
      return await worker.embed(texts, task);
    } finally {
      if (identical(worker, _current)) {
        if (worker.dead) {
          _current = null;
        } else if (worker.idle) {
          _idleTimer?.cancel();
          _idleTimer = Timer(idle, () => unawaited(close()));
        }
      }
    }
  }

  @override
  Future<void> close() async {
    _idleTimer?.cancel();
    final worker = _current;
    _current = null;
    await worker?.shutdown(closeTimeout);
  }
}

/// One worker isolate and the model loaded in it.
class _Worker {
  _Worker._() {
    // Callers wait on this; with none waiting, a failed start is not an
    // unhandled error.
    _ready.future.ignore();
  }

  final ReceivePort _inbox = ReceivePort();
  final Completer<SendPort> _ready = Completer();
  final Completer<void> _ended = Completer();
  final Map<int, Completer<List<Float32List>>> _pending = {};
  Isolate? _isolate;
  SendPort? _port;
  int _nextId = 0;
  bool _closing = false;

  /// Failed or closed: it takes no more work.
  bool dead = false;

  bool get idle => _pending.isEmpty;

  static _Worker start(EmbedderLoader load, String modelPath, String tokenizerPath) {
    final w = _Worker._();
    w._inbox.listen(w._receive);
    unawaited(w._spawn(load, modelPath, tokenizerPath));
    return w;
  }

  Future<void> _spawn(EmbedderLoader load, String modelPath, String tokenizerPath) async {
    try {
      _isolate = await Isolate.spawn(
        _main,
        (_inbox.sendPort, load, modelPath, tokenizerPath),
        onError: _inbox.sendPort,
        onExit: _inbox.sendPort, // sends null when the isolate has ended
        debugName: 'embedder',
      );
    } on Object catch (e) {
      _fail(StateError('Search by meaning could not start: $e'));
      _end();
    }
  }

  Future<List<Float32List>> embed(List<String> texts, EmbedTask task) async {
    final port = await _ready.future;
    if (dead) throw StateError('Embedder closed');
    final id = _nextId++;
    final done = Completer<List<Float32List>>();
    _pending[id] = done;
    port.send((id, texts, task.index));
    return done.future;
  }

  void _receive(Object? message) {
    switch (message) {
      case final SendPort port:
        _port = port;
        // Closed while the model was loading: release it straight away.
        if (_closing) {
          port.send(null);
        } else if (!_ready.isCompleted) {
          _ready.complete(port);
        }
      case (final int id, final List<Float32List> vectors):
        _pending.remove(id)?.complete(vectors);
      case (final int id, final String error):
        _pending.remove(id)?.completeError(StateError(error));
      case final String fatal: // the model could not load
        _fail(StateError(fatal));
      case [final Object error, _]: // an uncaught error in the worker
        _fail(StateError('$error'));
      case null: // the isolate has ended
        _fail(StateError('Search by meaning stopped'));
        _end();
    }
  }

  void _fail(Object error) {
    dead = true;
    if (!_ready.isCompleted) _ready.completeError(error);
    for (final c in _pending.values) {
      c.completeError(error);
    }
    _pending.clear();
  }

  void _end() {
    _inbox.close();
    if (!_ended.isCompleted) _ended.complete();
  }

  /// Asks the worker to release the model and waits until it has ended.
  Future<void> shutdown(Duration timeout) async {
    if (!_closing) {
      _closing = true;
      _fail(StateError('Embedder closed'));
      // Still loading: the worker is told as soon as it is ready (above).
      _port?.send(null);
    }
    await _ended.future.timeout(timeout, onTimeout: () {
      // Stuck in a run that never returns: the model's memory is lost, but
      // the worker must not live on.
      _isolate?.kill(priority: Isolate.immediate);
      _end();
    });
  }

  static void _main((SendPort, EmbedderLoader, String, String) args) {
    final (reply, load, modelPath, tokenizerPath) = args;
    final TextEmbedder embedder;
    try {
      embedder = load(modelPath, tokenizerPath);
    } on Object catch (e) {
      reply.send('Search by meaning could not start: $e');
      return; // nothing left open: the isolate ends
    }
    final inbox = ReceivePort();
    reply.send(inbox.sendPort);
    inbox.listen((Object? message) {
      if (message case (final int id, final List<String> texts, final int task)) {
        try {
          reply.send((id, embedder.embed(texts, task: EmbedTask.values[task])));
        } on Object catch (e) {
          reply.send((id, '$e'));
        }
      } else {
        // Asked to stop: free the model's native memory, then let the
        // isolate end (its last port closes).
        embedder.dispose();
        inbox.close();
      }
    });
  }
}
