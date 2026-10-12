import 'package:flutter/services.dart';

/// One event from the phone's speech recognizer.
class SpeechEvent {
  final String type; // ready, begin, level, partial, final, end, error, state, engine
  final String text;
  final double level;
  final bool finishing;
  final bool fatal;
  final bool listening;
  final String message;

  /// For 'source' events: audio is coming from the glasses' microphone.
  final bool glassesMic;

  const SpeechEvent(
    this.type, {
    this.text = '',
    this.level = 0,
    this.finishing = false,
    this.fatal = false,
    this.listening = false,
    this.message = '',
    this.glassesMic = false,
  });

  factory SpeechEvent.fromMap(Map map) => SpeechEvent(
        '${map['type']}',
        text: map['text'] as String? ?? '',
        level: (map['value'] as num?)?.toDouble() ?? 0,
        finishing: map['finishing'] as bool? ?? false,
        fatal: map['fatal'] as bool? ?? false,
        listening: map['listening'] as bool? ?? false,
        message: map['message'] as String? ?? '',
        glassesMic: map['glasses'] as bool? ?? false,
      );

  @override
  String toString() => 'SpeechEvent($type, "$text")';
}

abstract class SpeechEngine {
  Stream<SpeechEvent> get events;
  Future<bool> available();

  /// Start listening. Continuous mode restarts after every utterance.
  /// [audioSource] 'glasses' streams the G1 microphone (Android 13+; other
  /// phones fall back to their own microphone).
  Future<void> start({
    bool continuous = true,
    String language = '',
    bool preferOffline = true,
    bool quiet = true,
    String engine = 'auto',
    String audioSource = 'phone',
  });

  Future<void> stop();

  /// End the current utterance now and deliver its final text.
  Future<void> finish();
}

/// Android's built-in SpeechRecognizer through the app's own channel.
class PlatformSpeech implements SpeechEngine {
  static const _methods = MethodChannel('suite/speech');
  static final Stream<SpeechEvent> _events = const EventChannel('suite/speechEvents')
      .receiveBroadcastStream()
      .map((e) => SpeechEvent.fromMap(e as Map));

  @override
  Stream<SpeechEvent> get events => _events;

  @override
  Future<bool> available() async {
    try {
      return await _methods.invokeMethod<bool>('available') ?? false;
    } on MissingPluginException {
      return false;
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
  }) =>
      _methods.invokeMethod('start', {
        'continuous': continuous,
        'language': language,
        'preferOffline': preferOffline,
        'quiet': quiet,
        'engine': engine,
        'audioSource': audioSource,
      });

  @override
  Future<void> stop() => _methods.invokeMethod('stop');

  @override
  Future<void> finish() => _methods.invokeMethod('finish');
}
