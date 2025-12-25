import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for menu-based interactions.
class MenuModeHandler extends ModeHandler with TerminateDetection, WordMatching {
  bool _isActive = false;

  /// Menu items configuration
  static const Map<String, String> menuItems = {
    'one': 'This is menu item one.',
    '1': 'This is menu item one.',
    'two': 'This is menu item two.',
    '2': 'This is menu item two.',
  };

  MenuModeHandler();

  @override
  String get modeName => 'menu';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    return matchesWord(transcript, 'menu');
  }

  @override
  bool canHandleInMode(String transcript) {
    return _isActive;
  }

  @override
  bool isTerminateCommand(String transcript) {
    return matchesTerminate(transcript);
  }

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _log("📋 MenuModeHandler: Entering Menu Mode.");
    _isActive = true;

    return const ModeResult(
      continueListening: true,
      displayText: "Menu Item 1\nMenu Item 2",
    );
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // Menu mode automatically exits after selection
    _isActive = false;

    // Check for termination
    if (isTerminateCommand(lower)) {
      _log("📋 MenuModeHandler: User terminated menu mode.");
      return ModeResult.endSession;
    }

    // Find matching menu item
    for (final entry in menuItems.entries) {
      if (lower.contains(entry.key)) {
        _log("📋 MenuModeHandler: Selected '${entry.key}'");
        return ModeResult(
          continueListening: false,
          displayText: entry.value,
        );
      }
    }

    // No match found
    _log("📋 MenuModeHandler: Unrecognized selection: '$transcript'");
    return const ModeResult(
      continueListening: false,
      displayText: "Sorry, I didn't recognize that option.",
    );
  }

  @override
  void reset() {
    _log("📋 MenuModeHandler: Resetting mode.");
    _isActive = false;
  }
}
