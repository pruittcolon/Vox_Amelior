/// A function the model may call (Gemma 4 native tool calling).
class ToolSpec {
  const ToolSpec({required this.name, required this.description, this.parameters = const {}});

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

  Future<void> close();
}

/// Local language model (Gemma 4 via LiteRT-LM). Faked in tests.
abstract interface class LlmEngine {
  bool get isLoaded;

  /// Whether the loaded model understands tool calls.
  bool get supportsTools;

  /// Loads the model into memory if needed. Throws [LlmUnavailable] when the
  /// model has not been downloaded or cannot be started.
  Future<void> ensureLoaded();

  Future<LlmSession> openSession({required String system, List<ToolSpec> tools = const []});

  /// Frees the model's memory (several GB).
  Future<void> unload();
}

class LlmUnavailable implements Exception {
  const LlmUnavailable(this.message);
  final String message;
  @override
  String toString() => message;
}
