/// "Try" buttons: speaks a short line in a voice so you can hear it.
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../core/app_state.dart';
import '../core/audio/native.dart';
import '../core/audio/player.dart';
import '../core/render/audio.dart';
import '../core/tts/voices.dart';

const previewLine = 'Hello! I can read your books, notes and podcast scripts. Shall we begin?';

class VoicePreview extends ChangeNotifier {
  VoicePreview(this.app, this.native, this.player);

  final AppState app;
  final NativeApi native;
  final Player player;
  final _cache = <String, Float32List>{};

  /// Voice id being generated or played.
  String? busy;
  String? error;
  Timer? _done;

  Future<void> play(Voice voice, {double speed = 1.0, String text = previewLine}) async {
    if (busy == voice.id) return stop();
    // Previews share the speaker; stop reading (your place is saved).
    if (player.active) await player.stop();
    _done?.cancel();
    busy = voice.id;
    error = null;
    notifyListeners();
    try {
      final key = '${voice.id}|$speed|$text';
      var pcm = _cache[key];
      if (pcm == null) {
        final raw = await app.synth.speak(voice, text, speed: speed);
        pcm = normalizeSpeech(trimSilence(raw, outputRate), outputRate);
        fadeEdges(pcm, outputRate);
        _cache[key] = pcm;
      }
      if (busy != voice.id) return;
      await native.playerOpen(outputRate);
      await native.playerWrite(toInt16Bytes(pcm));
      final ms = (pcm.length / outputRate * 1000).round() + 300;
      _done = Timer(Duration(milliseconds: ms), () {
        if (busy == voice.id) {
          busy = null;
          notifyListeners();
        }
      });
    } catch (e) {
      error = e.toString();
      busy = null;
      notifyListeners();
    }
  }

  Future<void> stop() async {
    _done?.cancel();
    busy = null;
    notifyListeners();
    try {
      await native.playerStop();
    } catch (_) {}
  }

  void forget(String voiceId) => _cache.removeWhere((k, _) => k.startsWith('$voiceId|'));
}
