import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/alexa_service.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for Alexa/Voicemonkey smart home control via voice commands.
class AlexaModeHandler extends ModeHandler with TerminateDetection, WordMatching {
  final AlexaService _alexaService;

  bool _isActive = false;
  Timer? _autoExitTimer;

  /// Auto-exit timeout in seconds
  static const int autoExitSeconds = 30;

  AlexaModeHandler({AlexaService? alexaService})
      : _alexaService = alexaService ?? AlexaService.instance;

  @override
  String get modeName => 'alexa';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    return matchesWord(transcript, 'alexa');
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
    _log("🏠 AlexaModeHandler: Entering Alexa Remote Mode.");
    _isActive = true;
    _startAutoExitTimer();

    final commands = _alexaService.getAvailableCommands();

    return ModeResult(
      continueListening: true,
      displayText: "Alexa:\n$commands\nSay 'Terminate' to exit",
    );
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // Check for termination
    if (isTerminateCommand(lower)) {
      _log("🏠 AlexaModeHandler: User terminated Alexa mode.");
      reset();
      return ModeResult.endSession;
    }

    // Reset auto-exit timer on each command
    _startAutoExitTimer();

    // Find and trigger the matching device
    final deviceName = _alexaService.findDeviceForCommand(lower);

    if (deviceName != null) {
      final success = await _alexaService.triggerRoutine(deviceName, transcript: lower);
      final displayText = success ? "Sent: $deviceName" : "Error: $deviceName";

      return ModeResult(
        continueListening: true,
        displayText: displayText,
      );
    } else {
      _log("🏠 AlexaModeHandler: Unrecognized command: '$transcript'");
      return const ModeResult(
        continueListening: true,
        displayText: "Command not recognized",
      );
    }
  }

  void _startAutoExitTimer() {
    _autoExitTimer?.cancel();
    _autoExitTimer = Timer(Duration(seconds: autoExitSeconds), () {
      _log("🏠 AlexaModeHandler: Auto-exit after $autoExitSeconds seconds.");
      reset();
    });
  }

  @override
  void reset() {
    _log("🏠 AlexaModeHandler: Resetting mode.");
    _isActive = false;
    _autoExitTimer?.cancel();
    _autoExitTimer = null;
  }
}
