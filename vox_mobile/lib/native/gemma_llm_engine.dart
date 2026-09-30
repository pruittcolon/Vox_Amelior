import 'dart:async';
import 'dart:io';

import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/log.dart';

/// Which model file to run and how.
class LlmModelSpec {
  const LlmModelSpec({
    required this.path,
    required this.modelType,
    required this.supportsTools,
    this.contextTokens = ContextBudget.defaultContext,
  });

  final String path;

  /// flutter_gemma model family name, e.g. 'gemma4', 'gemmaIt', 'qwen3'.
  final String modelType;
  final bool supportsTools;

  /// Context window (input + output) the model is loaded with.
  final int contextTokens;

  ModelType get type => ModelType.values.asNameMap()[modelType] ?? ModelType.general;
}

/// Gemma 4 (or another LiteRT-LM model) running on the phone.
///
/// The model needs several GB of RAM, so it is loaded on demand and released
/// after [idleUnloadAfter] without use. Sessions run one at a time.
class GemmaLlmEngine implements LlmEngine {
  GemmaLlmEngine({
    required this.spec,
    this.defaultReplyTokens = 700,
    this.idleUnloadAfter = const Duration(minutes: 3),
  });

  /// Read each time the model is loaded, so settings changes apply.
  final LlmModelSpec? Function() spec;
  final int defaultReplyTokens;
  final Duration idleUnloadAfter;

  InferenceModel? _model;
  LlmModelSpec? _loaded;
  int _loadedContext = 0;
  Timer? _idle;
  Future<void> _lock = Future<void>.value();

  /// Set after a failure on the GPU: run on the CPU until the model changes.
  bool _cpuOnly = false;

  /// Context size requested explicitly (phone test), until [unload].
  int? _pinnedContext;

  /// Chats currently open. The model is never reloaded or freed under one.
  int _active = 0;
  bool _unloadWhenIdle = false;

  static bool _initialized = false;

  /// Registers the LiteRT-LM engine. Call once per isolate before use.
  static Future<void> initializePlugin() async {
    if (_initialized) return;
    await FlutterGemma.initialize(inferenceEngines: const [LiteRtLmEngine()]);
    _initialized = true;
  }

  @override
  bool get isLoaded => _model != null;

  @override
  bool get supportsTools => (_loaded ?? spec())?.supportsTools ?? false;

  /// Where the model runs now ('GPU' or 'CPU'), for the settings screen.
  String get backendLabel => _cpuOnly ? 'CPU' : 'GPU';

  @override
  Future<void> ensureLoaded({int? contextTokens}) async {
    _idle?.cancel();
    final want = spec();
    if (want == null || !File(want.path).existsSync()) {
      throw const LlmUnavailable('The assistant model is not downloaded yet. Get it under More → Models.');
    }
    // An explicit size (the phone test) sticks until the model is unloaded,
    // so opening a chat does not quietly reload it at the default size.
    if (contextTokens != null) _pinnedContext = contextTokens;
    final context = _pinnedContext ?? want.contextTokens;
    if (_model != null && _loaded?.path == want.path && _loadedContext == context) return;
    // A chat is running: keep the current model; the change applies to the
    // next chat (openSession reloads once the running one has closed).
    if (_model != null && _active > 0) return;
    if (_loaded != null && _loaded!.path != want.path) _cpuOnly = false;
    final pin = _pinnedContext;
    await _free();
    _pinnedContext = pin;
    await initializePlugin();
    try {
      await FlutterGemma.installModel(modelType: want.type, fileType: ModelFileType.litertlm).fromFile(want.path).install();
    } on Object catch (e) {
      throw LlmUnavailable('Could not register the model file ($e).');
    }
    if (!_cpuOnly) {
      try {
        _model = await FlutterGemma.getActiveModel(maxTokens: context, preferredBackend: PreferredBackend.gpu);
      } on Object catch (e) {
        // Some phones lack a usable GPU delegate; the CPU path always works.
        Log.w('gemma', 'GPU load failed, retrying on CPU', e.runtimeType);
        _cpuOnly = true;
      }
    }
    if (_model == null) {
      try {
        _model = await FlutterGemma.getActiveModel(maxTokens: context, preferredBackend: PreferredBackend.cpu);
      } on Object catch (e2) {
        throw LlmUnavailable('Could not start the assistant on this phone (${e2.runtimeType}). It may need more free memory.');
      }
    }
    _loaded = want;
    _loadedContext = context;
  }

  @override
  Future<bool> recover() async {
    if (_cpuOnly) return false;
    Log.w('gemma', 'generation failed on GPU; switching to CPU');
    _cpuOnly = true;
    await unload();
    return true;
  }

  @override
  Future<LlmSession> openSession({
    required String system,
    List<ToolSpec> tools = const [],
    int? maxReplyTokens,
    int? contextTokens,
  }) async {
    // One session at a time: wait for the previous one to close.
    final previous = _lock;
    final release = Completer<void>();
    _lock = release.future;
    try {
      await previous.timeout(const Duration(minutes: 5));
    } on TimeoutException {
      release.complete();
      throw const LlmUnavailable('Gemma is still busy with an earlier request. Try again in a moment.');
    }
    try {
      await ensureLoaded(contextTokens: contextTokens);
      final useTools = tools.isNotEmpty && supportsTools;
      final chat = await _model!.createChat(
        systemInstruction: system,
        temperature: 0.4,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: maxReplyTokens ?? defaultReplyTokens,
        modelType: _loaded!.type,
        supportsFunctionCalls: useTools,
        tools: [
          if (useTools)
            for (final t in tools) Tool(name: t.name, description: t.description, parameters: t.parameters),
        ],
      );
      _active++;
      return _GemmaSession(chat, () {
        _active--;
        if (_active == 0 && _unloadWhenIdle) {
          unawaited(unload());
        } else {
          _scheduleUnload();
        }
        if (!release.isCompleted) release.complete();
      });
    } on Object {
      release.complete();
      rethrow;
    }
  }

  @override
  Future<void> unload() async {
    _idle?.cancel();
    if (_active > 0) {
      _unloadWhenIdle = true;
      return;
    }
    _unloadWhenIdle = false;
    _pinnedContext = null;
    await _free();
  }

  Future<void> _free() async {
    final model = _model;
    _model = null;
    _loaded = null;
    _loadedContext = 0;
    await model?.close();
  }

  void _scheduleUnload() {
    _idle?.cancel();
    _idle = Timer(idleUnloadAfter, () => unawaited(unload()));
  }
}

class _GemmaSession implements LlmSession {
  _GemmaSession(this._chat, this._onClose);

  final InferenceChat _chat;
  final void Function() _onClose;
  bool _closed = false;

  @override
  Stream<LlmEvent> send(String text) async* {
    await _chat.addQueryChunk(Message.text(text: text, isUser: true));
    yield* _generate();
  }

  @override
  Stream<LlmEvent> sendToolResult(String name, Map<String, Object?> result) async* {
    await _chat.addQueryChunk(Message.toolResponse(toolName: name, response: result));
    yield* _generate();
  }

  @override
  Future<int?> countTokens(String text) async {
    try {
      return await _chat.session.sizeInTokens(text);
    } on Object {
      return null;
    }
  }

  Stream<LlmEvent> _generate() async* {
    await for (final r in _chat.generateChatResponseAsync()) {
      if (r is TextResponse) {
        yield LlmText(r.token);
      } else if (r is FunctionCallResponse) {
        yield LlmToolCall(r.name, Map<String, Object?>.from(r.args));
      } else if (r is ParallelFunctionCallResponse) {
        for (final c in r.calls) {
          yield LlmToolCall(c.name, Map<String, Object?>.from(c.args));
        }
      }
    }
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    try {
      await _chat.close();
    } on Object catch (e) {
      Log.w('gemma', 'closing a chat failed', e);
    } finally {
      _onClose();
    }
  }
}
