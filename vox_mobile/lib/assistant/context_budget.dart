import 'dart:math' as math;

/// How the model's context window (everything it can see at once: our
/// instructions, the transcript, tool results and its own reply) is shared
/// out. Every size here derives from [contextTokens], which the phone test
/// measures (Settings → Assistant → Test this phone).
class ContextBudget {
  const ContextBudget(this.contextTokens, {this._chunkTokens = 0});

  /// Sizes the phone test tries, smallest first.
  static const List<int> testSizes = [2048, 4096, 8192, 16384, 32768];

  /// Used until the phone has been tested (reported to work on phones).
  static const int defaultContext = 4096;

  /// Conservative characters per token for planning without the model's
  /// tokenizer. Gemma averages ~4.9 for English; names, times and numbers
  /// are denser, so assume fewer to never overfill.
  static const double charsPerToken = 3.2;

  final int contextTokens;
  final int _chunkTokens;

  /// Longest reply the model may write.
  int get replyTokens => (contextTokens * 0.15).round().clamp(256, 1024);

  /// Recommended transcript per "go through everything" part: half the context.
  int get recommendedChunkTokens => contextTokens ~/ 2;

  /// Largest part that still leaves room for instructions, notes and the reply.
  int get maxChunkTokens => math.max(512, contextTokens - replyTokens - 700);

  /// Transcript per part: the user's choice, or the recommendation.
  int get chunkTokens => (_chunkTokens > 0 ? _chunkTokens : recommendedChunkTokens).clamp(512, maxChunkTokens);

  /// "Found so far" notes carried from one part to the next.
  int get carryTokens => math.min(300, contextTokens ~/ 12);

  /// Excerpts handed over with a quick question. Tools need room too.
  int excerptTokens({required bool agent}) => agent ? (contextTokens * 0.3).round() : contextTokens ~/ 2;

  /// Largest single tool result.
  int get toolResultTokens => (contextTokens * 0.12).round();

  /// Rough share taken by tool descriptions when agent mode is on.
  static const int toolSpecTokens = 700;

  /// Rough size of the fixed instructions.
  static const int systemTokens = 350;

  static int estimateTokens(String text) => (text.length / charsPerToken).ceil();

  static int charsFor(int tokens) => (tokens * charsPerToken).floor();

  /// About how many transcript lines fit in [tokens].
  static int linesFor(int tokens) => tokens ~/ 22;
}
