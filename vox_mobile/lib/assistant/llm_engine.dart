import 'dart:async';

/// A function the model may call (Gemma 4 native tool calling).
///
/// Every tool must declare at least one parameter: LiteRT-LM fails every
/// request ("Failed to start streaming (code: 13)") when a tool has none.
class ToolSpec {
  const ToolSpec({required this.name, required this.description, required this.parameters});

  final String name;
  final String description;

  /// JSON schema of the arguments: `{'type': 'object', 'properties': {...}}`.
  final Map<String, Object?> parameters;
}

/// Something the model produced.
sealed class LlmEvent {
  const LlmEvent();
}

class LlmText extends LlmEvent {
  const LlmText(this.text);
  final String text;
}

class LlmToolCall extends LlmEvent {
  const LlmToolCall(this.name, this.args);
  final String name;
  final Map<String, Object?> args;
}

/// One conversation with the model.
abstract interface class LlmSession {
  /// Sends the user's message and streams the reply.
  Stream<LlmEvent> send(String text);

  /// Returns a tool's result to the model and streams its continuation.
  Stream<LlmEvent> sendToolResult(String name, Map<String, Object?> result);

  /// Exact size of [text] in the model's tokens, or null if unknown.
  Future<int?> countTokens(String text);

  Future<void> close();
}

/// Local language model (Gemma 4 via LiteRT-LM). Faked in tests.
abstract interface class LlmEngine {
  bool get isLoaded;

  /// Whether the loaded model understands tool calls.
  bool get supportsTools;

  /// Loads the model into memory if needed, with a context window of
  /// [contextTokens] (default: the configured size). Throws [LlmUnavailable]
  /// when the model has not been downloaded or cannot be started.
  Future<void> ensureLoaded({int? contextTokens});

  /// Opens a chat. [contextTokens] loads the model with that context window
  /// first (used by the phone test); otherwise the configured size is used.
  Future<LlmSession> openSession({
    required String system,
    List<ToolSpec> tools = const [],
    int? maxReplyTokens,
    int? contextTokens,
  });

  /// After a failure: switch to a safer way of running (e.g. CPU instead of
  /// GPU). Returns false when there is nothing left to try.
  Future<bool> recover();

  /// Frees the model's memory (several GB). While a chat is running this
  /// happens as soon as it ends.
  Future<void> unload();
}

class LlmUnavailable implements Exception {
  const LlmUnavailable(this.message);
  final String message;
  @override
  String toString() => message;
}

/// A plain-language explanation of an assistant failure.
String friendlyLlmError(Object e) {
  if (e is LlmUnavailable) return e.message;
  if (e is TimeoutException) {
    return 'Gemma took too long to answer. Try again, or ask about a shorter period.';
  }
  final raw = '$e';
  if (raw.contains('code: 13') || raw.contains('Failed to start streaming')) {
    return 'Gemma could not start answering on this phone. Try again, or run Settings → Assistant → Test this phone.';
  }
  if (raw.contains('DYNAMIC_UPDATE_SLICE') || raw.toLowerCase().contains('too long') || raw.contains('exceed')) {
    return 'That was more than Gemma can read at once. Try a shorter period, or test the phone in Settings.';
  }
  return 'Gemma stopped unexpectedly. Try again. ($raw)';
}
