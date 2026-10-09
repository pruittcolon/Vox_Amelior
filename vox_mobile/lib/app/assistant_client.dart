import 'dart:async';

import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';
import 'package:vox_amelior_mobile/service/service_controller.dart';

/// Asks the assistant from the app.
///
/// While Vox is listening, questions go to the always-on service so they
/// share its work queue with transcription (the model is loaded once, and
/// heavy jobs never overlap). Otherwise the app runs the model itself.
class AssistantClient {
  AssistantClient({required this.service, required this.requests, required this.transcripts, required this.local});

  final ServiceController service;
  final AssistantRequestRepository requests;
  final TranscriptRepository transcripts;
  final AssistantService local;

  /// Longest silence from the service before giving up (model loading and a
  /// transcription backlog can take a while on a phone).
  static const Duration idleTimeout = Duration(minutes: 3);

  Stream<AnswerEvent> ask(String question) {
    final q = question.trim();
    if (q.isEmpty) return const Stream.empty();
    if (service.isListening && service.state != ListenState.error) return _remote(q);
    return _localSaved(q);
  }

  /// Answers in the app, and keeps the finished answer in the saved list
  /// (the service keeps the ones it answers). Stopped answers are not kept.
  /// The question is only stored once answered, so a starting service never
  /// mistakes it for one still waiting.
  Stream<AnswerEvent> _localSaved(String question) {
    final answer = StringBuffer();
    var sources = const <int>[];
    var failed = false;
    return local.ask(question).transform(StreamTransformer<AnswerEvent, AnswerEvent>.fromHandlers(
      handleError: (e, st, sink) {
        failed = true;
        sink.addError(e, st);
      },
      handleData: (e, sink) {
        if (e.token != null) answer.write(e.token);
        if (e.sources != null) sources = [for (final s in e.sources!) s.id];
        sink.add(e);
      },
      handleDone: (sink) {
        final text = answer.toString().trim();
        if (!failed && text.isNotEmpty) requests.answer(requests.add(question, source: RequestSource.app), text, sources: sources);
        sink.close();
      },
    ));
  }

  Stream<AnswerEvent> _remote(String question) {
    final id = requests.add(question, source: RequestSource.app);
    final out = StreamController<AnswerEvent>();
    late StreamSubscription<Map<Object?, Object?>> sub;
    Timer? idle;

    var finished = false;
    Future<void> finish([Object? error]) async {
      if (finished) return;
      finished = true;
      idle?.cancel();
      await sub.cancel();
      if (error != null) out.addError(error);
      await out.close();
    }

    // Leaving the answer (Stop button, new chat) stops the service too.
    out.onCancel = () {
      if (!finished) service.cancelAsk(id);
      finished = true;
      idle?.cancel();
      return sub.cancel();
    };

    void resetIdle() {
      idle?.cancel();
      idle = Timer(idleTimeout, () {
        service.cancelAsk(id);
        unawaited(finish(const LlmUnavailable('Gemma took too long. Try again, or ask about a shorter period.')));
      });
    }

    sub = service.events.listen((e) {
      if (e['type'] != ServiceEvents.answer || e['id'] != id) return;
      resetIdle();
      switch (e['kind']) {
        case 'sources':
          final ids = (e['ids'] as List<Object?>? ?? const []).whereType<int>().toList();
          out.add(AnswerEvent.sources(transcripts.segmentsByIds(ids)));
        case 'token':
          out.add(AnswerEvent.token('${e['t']}'));
        case 'tool':
          out.add(AnswerEvent.tool('${e['name']}', const {}));
        case 'done':
          unawaited(finish());
        case 'error':
          unawaited(finish(LlmUnavailable('${e['message']}')));
      }
    });
    resetIdle();
    service.ask(id);
    return out.stream;
  }
}
