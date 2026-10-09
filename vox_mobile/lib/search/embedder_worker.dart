import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/native/embedding_gemma.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';

/// Runs EmbeddingGemma in a long-lived worker isolate, so embedding never
/// blocks the UI. The model (about 400 MB in memory) is loaded on first use
/// and the worker exits after [idle] without work, freeing it again.
class IsolateEmbedder implements AsyncEmbedder {
  IsolateEmbedder({required this.modelPath, required this.tokenizerPath, this.idle = const Duration(minutes: 2)});

  final String modelPath;
  final String tokenizerPath;
  final Duration idle;

  Isolate? _isolate;
  ReceivePort? _inbox;
  SendPort? _worker;
  Future<SendPort>? _starting;
  Timer? _idleTimer;
  int _nextId = 0;
  final Map<int, Completer<List<Float32List>>> _pending = {};

  @override
  Future<List<Float32List>> embed(List<String> texts, {required EmbedTask task}) async {
    if (texts.isEmpty) return const [];
    _idleTimer?.cancel();
    final worker = await (_starting ??= _start());
    final id = _nextId++;
    final done = Completer<List<Float32List>>();
    _pending[id] = done;
    worker.send((id, texts, task.index));
    try {
      return await done.future;
    } finally {
      if (_pending.isEmpty) _idleTimer = Timer(idle, () => unawaited(close()));
    }
  }

  Future<SendPort> _start() async {
    final inbox = ReceivePort();
    final ready = Completer<SendPort>();
    _inbox = inbox;
    inbox.listen((Object? message) {
      switch (message) {
        case SendPort port:
          ready.complete(port);
        case (int id, List<Float32List> vectors):
          _pending.remove(id)?.complete(vectors);
        case (int id, String error):
          _pending.remove(id)?.completeError(StateError(error));
        case String fatal: // the model could not load
          final error = StateError(fatal);
          if (!ready.isCompleted) ready.completeError(error);
          _failAll(error);
          unawaited(close());
        case [final Object error, _]: // an uncaught error in the worker
          _failAll(StateError('$error'));
          unawaited(close());
      }
    });
    try {
      _isolate = await Isolate.spawn(
        _main,
        (inbox.sendPort, modelPath, tokenizerPath),
        onError: inbox.sendPort,
        debugName: 'embedder',
      );
      return _worker = await ready.future;
    } on Object {
      unawaited(close());
      rethrow;
    }
  }

  void _failAll(Object error) {
    for (final c in _pending.values) {
      if (!c.isCompleted) c.completeError(error);
    }
    _pending.clear();
  }

  @override
  Future<void> close() async {
    _idleTimer?.cancel();
    _worker?.send(null);
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _inbox?.close();
    _failAll(StateError('Embedder closed'));
    _isolate = null;
    _inbox = null;
    _worker = null;
    _starting = null;
  }

  static void _main((SendPort, String, String) args) {
    final (reply, model, tokenizer) = args;
    final TextEmbedder embedder;
    try {
      embedder = EmbeddingGemmaOnnx(modelPath: model, tokenizerPath: tokenizer);
    } on Object catch (e) {
      reply.send('Search by meaning could not start: $e');
      return;
    }
    final inbox = ReceivePort();
    reply.send(inbox.sendPort);
    inbox.listen((Object? message) {
      if (message case (int id, List<String> texts, int task)) {
        try {
          reply.send((id, embedder.embed(texts, task: EmbedTask.values[task])));
        } on Object catch (e) {
          reply.send((id, '$e'));
        }
      } else {
        embedder.dispose();
        inbox.close();
      }
    });
  }
}
