/// Local language model (Gemma 3n via LiteRT-LM). Faked in tests.
abstract interface class LlmEngine {
  bool get isLoaded;

  /// Loads the model into memory if needed. Throws [LlmUnavailable] when the
  /// model has not been downloaded or cannot be started.
  Future<void> ensureLoaded();

  /// Streams the reply token by token.
  Stream<String> generate({required String system, required String prompt});

  /// Frees the model's memory (several GB).
  Future<void> unload();
}

class LlmUnavailable implements Exception {
  const LlmUnavailable(this.message);
  final String message;
  @override
  String toString() => message;
}
