import 'dart:async';
import 'package:flutter/foundation.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Result of a mode handler execution
class ModeResult {
  /// Whether the mode should continue listening for more commands
  final bool continueListening;

  /// Text to display on glasses (if any)
  final String? displayText;

  /// Whether to use WhisperServer instead of Deepgram for next listen
  final bool useWhisperServer;

  /// Whether the handler handled the transcript
  final bool handled;

  const ModeResult({
    this.continueListening = false,
    this.displayText,
    this.useWhisperServer = false,
    this.handled = true,
  });

  /// Result indicating the handler didn't handle this transcript
  static const ModeResult notHandled = ModeResult(handled: false);

  /// Result indicating session should end
  static const ModeResult endSession = ModeResult(
    continueListening: false,
    handled: true,
  );
}

/// Abstract interface for mode handlers.
///
/// Each mode handler encapsulates the logic for one interaction mode
/// (e.g., Roku remote, Alexa commands, memory queries, etc.)
abstract class ModeHandler {
  /// Unique identifier for this mode
  String get modeName;

  /// Whether this handler is currently active
  bool get isActive;

  /// Check if this handler can handle the given transcript to enter the mode.
  /// This is called when no mode is active to determine if this mode should be entered.
  bool canEnterMode(String transcript);

  /// Check if this handler should handle the transcript while already active.
  /// Returns true if this is a valid command for this mode.
  bool canHandleInMode(String transcript);

  /// Enter this mode (activate it)
  Future<ModeResult> enterMode(String transcript);

  /// Handle a command while in this mode
  Future<ModeResult> handleCommand(String transcript);

  /// Check if the transcript is a termination command for this mode
  bool isTerminateCommand(String transcript);

  /// Reset/deactivate this mode
  void reset();
}

/// Mixin providing common terminate detection
mixin TerminateDetection {
  /// Common termination patterns
  bool matchesTerminate(String text) {
    final t = text.toLowerCase();
    final re = RegExp(r"\bterminate(?:s|d|ing|ion)?\b");
    return re.hasMatch(t) || t.contains('end') || t.contains('stop') || t.contains('exit');
  }
}

/// Mixin providing common word matching utilities
mixin WordMatching {
  bool matchesWord(String text, String word) {
    return RegExp("\\b$word\\b", caseSensitive: false).hasMatch(text);
  }

  bool containsAny(String text, List<String> words) {
    final lower = text.toLowerCase();
    return words.any((word) => lower.contains(word.toLowerCase()));
  }
}
