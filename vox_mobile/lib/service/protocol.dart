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

  /// Phone context test: {type, kind: step|done|error, size, status, detail | best, failedAt, note | message}
  static const probe = 'probe';

  /// A review made progress: {type, id}
  static const review = 'review';

  /// Microphone loudness while a screen asks for it: {type, level, peak, clipping}
  /// (level and peak are 0–1 on a dB scale).
  static const level = 'level';
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

  /// {cmd, id} — stop answering request [id].
  static const cancelAsk = 'cancelAsk';

  /// Run the phone context test with the service's model.
  static const probe = 'probe';

  /// Reviews were added or resumed: start working on them.
  static const reviewKick = 'reviewKick';

  /// {cmd, on: bool} — send [ServiceEvents.level] about 5 times a second.
  static const levelMeter = 'meter';
}

/// What the service is doing, for the UI.
enum ListenState { starting, listening, paused, autoPaused, error }
