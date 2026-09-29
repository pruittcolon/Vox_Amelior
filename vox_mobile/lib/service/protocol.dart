/// Messages between the app and the always-on service (JSON-friendly maps).
abstract final class ServiceEvents {
  /// {type, state, reason, backlog, activity, today, errors}
  static const status = 'status';

  /// {type, id} — a new utterance was saved.
  static const segment = 'segment';

  /// {type, id, kind: sources|token|tool|done|error, ...}
  static const answer = 'answer';

  /// {type, message, fatal}
  static const error = 'error';
}

abstract final class ServiceCommands {
  static const pause = 'pause';
  static const resume = 'resume';
  static const reload = 'reload';

  /// {cmd, id} — answer assistant request [id] (already stored by the app).
  static const ask = 'ask';

  /// Stop transcribing for a moment (audio keeps queueing), e.g. while the
  /// app records voice samples. {cmd, on: bool}
  static const holdTranscription = 'hold';

  static const retryStart = 'retry';
}

/// What the service is doing, for the UI.
enum ListenState { starting, listening, paused, autoPaused, error }
