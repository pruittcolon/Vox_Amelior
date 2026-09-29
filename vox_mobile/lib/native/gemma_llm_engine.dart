import 'dart:async';
import 'dart:io';

import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/log.dart';

/// Which model file to run and how.
class LlmModelSpec {
  const LlmModelSpec({required this.path, required this.modelType, required this.supportsTools});

  final String path;

  /// flutter_gemma model family name, e.g. 'gemma4', 'gemmaIt', 'qwen3'.
  final String modelType;
  final bool supportsTools;

  ModelType get type => ModelType.values.asNameMap()[modelType] ?? ModelType.general;

  Map<String, Object?> toJson() => {'path': path, 'modelType': modelType, 'supportsTools': supportsTools};

  static LlmModelSpec? fromJson(Object? j) {
    if (j is! Map) return null;
    final path = j['path'];
    if (path is! String) return null;
    return LlmModelSpec(
      path: path,
      modelType: j['modelType'] as String? ?? 'gemma4',
      supportsTools: j['supportsTools'] as bool? ?? true,
    );
  }
}

/// Gemma 4 (or another LiteRT-LM model) running on the phone.
///
/// The model needs several GB of RAM, so it is loaded on demand and released
/// after [idleUnloadAfter] without use. Sessions run one at a time.
class GemmaLlmEngine implements LlmEngine {
  GemmaLlmEngine({
    required this.spec,
    this.contextTokens = 4096,
    this.maxReplyTokens = 700,
    this.idleUnloadAfter = const Duration(minutes: 3),
  });

  /// Read each time the model is loaded, so settings changes apply.
  final LlmModelSpec? Function() spec;
  final int contextTokens;
  final int maxReplyTokens;
  final Duration idleUnloadAfter;

  InferenceModel? _model;
  LlmModelSpec? _loaded;
  Timer? _idle;
  Future<void> _lock = Future<void>.value();

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

  @override
  Future<void> ensureLoaded() async {
    _idle?.cancel();
    final want = spec();
    if (want == null || !File(want.path).existsSync()) {
      throw const LlmUnavailable('The assistant model is not downloaded yet. Get it under More → Models.');
    }
    if (_model != null && _loaded?.path == want.path) return;
    await unload();
    await initializePlugin();
    try {
      await FlutterGemma.installModel(modelType: want.type, fileType: ModelFileType.litertlm).fromFile(want.path).install();
    } on Object catch (e) {
      throw LlmUnavailable('Could not register the model file ($e).');
    }
    try {
      _model = await FlutterGemma.getActiveModel(maxTokens: contextTokens, preferredBackend: PreferredBackend.gpu);
    } on Object catch (e) {
      // Some phones lack a usable GPU delegate; the CPU path always works.
      Log.w('gemma', 'GPU load failed, retrying on CPU', e.runtimeType);
      try {
        _model = await FlutterGemma.getActiveModel(maxTokens: contextTokens, preferredBackend: PreferredBackend.cpu);
      } on Object catch (e2) {
        throw LlmUnavailable('Could not start the assistant on this phone (${e2.runtimeType}). It may need more free memory.');
      }
    }
    _loaded = want;
  }

  @override
  Future<LlmSession> openSession({required String system, List<ToolSpec> tools = const []}) async {
    // One session at a time: wait for the previous one to close.
    final previous = _lock;
    final release = Completer<void>();
    _lock = release.future;
    await previous;
    try {
      await ensureLoaded();
      final useTools = tools.isNotEmpty && supportsTools;
      final chat = await _model!.createChat(
        systemInstruction: system,
        temperature: 0.4,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: maxReplyTokens,
        modelType: _loaded!.type,
        supportsFunctionCalls: useTools,
        tools: [
          if (useTools)
            for (final t in tools) Tool(name: t.name, description: t.description, parameters: t.parameters),
        ],
      );
      return _GemmaSession(chat, () {
        _scheduleUnload();
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
    final model = _model;
    _model = null;
    _loaded = null;
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
    } finally {
      _onClose();
    }
  }
}
