import 'dart:async';

import 'package:flutter_tts/flutter_tts.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/native/local_notifier.dart';
import 'package:vox_amelior_mobile/service/listening_runtime.dart';

/// Answers questions spoken to Vox ("Hey Vox, ...") while the app is alive:
/// shows the answer as a notification and reads it aloud.
class AssistantInbox {
  AssistantInbox(this._services);

  final AppServices _services;
  final FlutterTts _tts = FlutterTts();
  StreamSubscription<Map<Object?, Object?>>? _sub;
  bool _working = false;

  void start() {
    _sub = _services.listening.events.listen((event) {
      if (event['type'] != ServiceEvents.request) return;
      final id = event['id'];
      if (id is int) _services.listening.acknowledgeRequest(id);
      unawaited(processPending());
    });
    unawaited(processPending());
  }

  Future<void> processPending() async {
    if (_working) return;
    _working = true;
    try {
      for (final request in _services.requests.takePending()) {
        await _answer(request);
      }
    } finally {
      _working = false;
    }
  }

  Future<void> _answer(AssistantRequest request) async {
    try {
      final answer = await _services.assistant.answer(request.text);
      final text = answer.text.isEmpty ? 'Sorry, I have no answer for that.' : answer.text;
      _services.requests.answer(request.id, text);
      await LocalNotifier.instance.show(request.text, text);
      if (_services.settings.value.speakReplies) await _tts.speak(text);
    } on LlmUnavailable catch (e) {
      _services.requests.fail(request.id, e.message);
      await LocalNotifier.instance.show('Vox cannot answer yet', e.message);
    } on Object catch (e, st) {
      Log.e('inbox', 'answering failed', e, st);
      _services.requests.fail(request.id, 'Something went wrong while answering.');
    }
    _services.dataVersion.value++;
  }

  Future<void> dispose() async {
    await _sub?.cancel();
    await _tts.stop();
  }
}
