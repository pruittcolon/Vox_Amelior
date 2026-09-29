import 'dart:async';

import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/log.dart';

/// Gemma 3n running on the phone through LiteRT-LM.
///
/// The model needs several GB of RAM, so it is loaded on demand and released
/// after [idleUnloadAfter] without use. One reply is generated at a time.
class GemmaLlmEngine implements LlmEngine {
  GemmaLlmEngine({
    this.contextTokens = 4096,
    this.maxReplyTokens = 600,
    this.idleUnloadAfter = const Duration(minutes: 3),
  });

  final int contextTokens;
  final int maxReplyTokens;
  final Duration idleUnloadAfter;

  InferenceModel? _model;
  Timer? _idle;
  Future<void> _queue = Future<void>.value();

  @override
  bool get isLoaded => _model != null;

  @override
  Future<void> ensureLoaded() async {
    _idle?.cancel();
    if (_model != null) return;
    if (!FlutterGemma.hasActiveModel()) {
      throw const LlmUnavailable('Gemma is not installed yet. Download it in Setup.');
    }
    try {
      _model = await FlutterGemma.getActiveModel(maxTokens: contextTokens, preferredBackend: PreferredBackend.gpu);
    } on Object catch (e) {
      // Some phones lack a usable GPU delegate; the CPU path always works.
      Log.w('gemma', 'GPU load failed, retrying on CPU', e.runtimeType);
      try {
        _model = await FlutterGemma.getActiveModel(maxTokens: contextTokens, preferredBackend: PreferredBackend.cpu);
      } on Object catch (e2) {
        throw LlmUnavailable('Could not start Gemma on this phone (${e2.runtimeType}). It may need more free memory.');
      }
    }
  }

  @override
  Stream<String> generate({required String system, required String prompt}) {
    final controller = StreamController<String>();
    final previous = _queue;
    final done = Completer<void>();
    _queue = done.future;

    unawaited(() async {
      try {
        await previous;
        await ensureLoaded();
        final chat = await _model!.createChat(
          systemInstruction: system,
          temperature: 0.3,
          maxOutputTokens: maxReplyTokens,
        );
        try {
          await chat.addQueryChunk(Message.text(text: prompt, isUser: true));
          await for (final response in chat.generateChatResponseAsync()) {
            if (response is TextResponse) controller.add(response.token);
          }
        } finally {
          await chat.close();
        }
      } on Object catch (e, st) {
        controller.addError(e, st);
      } finally {
        _scheduleUnload();
        done.complete();
        await controller.close();
      }
    }());
    return controller.stream;
  }

  @override
  Future<void> unload() async {
    _idle?.cancel();
    final model = _model;
    _model = null;
    await model?.close();
  }

  void _scheduleUnload() {
    _idle?.cancel();
    _idle = Timer(idleUnloadAfter, () => unawaited(unload()));
  }
}
