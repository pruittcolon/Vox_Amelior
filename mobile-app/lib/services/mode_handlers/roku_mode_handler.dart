import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/roku.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/evenai.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for Roku TV remote control via voice commands.
/// Uses animated text display with dot animation for visual feedback.
class RokuModeHandler extends ModeHandler with TerminateDetection, WordMatching {
  final RokuRemote _rokuRemote;

  bool _isActive = false;
  Timer? _autoExitTimer;
  Timer? _feedbackTimer;
  Timer? _animationTimer;
  int _dotCount = 1;

  /// Auto-exit timeout - 2 minutes
  static const int autoExitSeconds = 120;

  /// Command menu text - single line format
  static const String commandLine = 
      "ON | OFF | UP | DOWN | LEFT | RIGHT | SELECT | HOME | BACK | VOL UP | VOL DOWN";

  RokuModeHandler({RokuRemote? rokuRemote})
      : _rokuRemote = rokuRemote ?? RokuRemote();

  @override
  String get modeName => 'roku';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    // Deepgram may transcribe "roku" as various spellings
    final lower = transcript.toLowerCase();
    final matched = lower.contains('roku') || 
                    lower.contains('rocco') || 
                    lower.contains('roko') ||
                    lower.contains('ruku');
    _log("🎮 RokuModeHandler: canEnterMode('$transcript') = $matched");
    return matched;
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
    _log("🎮 RokuModeHandler: Entering Roku Remote Mode.");
    _isActive = true;
    _startAutoExitTimer();

    // Start the waiting animation
    await _startWaitingAnimation();

    return const ModeResult(
      continueListening: true,
      displayText: null, // We handle display ourselves
    );
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // Check for termination
    if (isTerminateCommand(lower)) {
      _log("🎮 RokuModeHandler: User terminated Roku mode.");
      reset();
      return ModeResult.endSession;
    }

    // Reset auto-exit timer on each command
    _startAutoExitTimer();

    // Stop waiting animation while processing
    _stopAnimation();

    // Parse and execute command
    String? commandName;

    if (lower.contains("volume up") || lower.contains("vol up") ||
        (lower.contains("volume") && lower.contains("up"))) {
      await _rokuRemote.volumeUp();
      await _rokuRemote.volumeUp();
      await _rokuRemote.volumeUp();
      commandName = "VOL UP";
    } else if (lower.contains("volume down") || lower.contains("vol down") ||
        (lower.contains("volume") && lower.contains("down"))) {
      await _rokuRemote.volumeDown();
      await _rokuRemote.volumeDown();
      await _rokuRemote.volumeDown();
      commandName = "VOL DOWN";
    } else if (lower.contains("on")) {
      await _rokuRemote.powerOn();
      commandName = "ON";
    } else if (lower.contains("off")) {
      await _rokuRemote.powerOff();
      commandName = "OFF";
    } else if (lower.contains("up")) {
      await _rokuRemote.up();
      commandName = "UP";
    } else if (lower.contains("down")) {
      await _rokuRemote.down();
      commandName = "DOWN";
    } else if (lower.contains("left")) {
      await _rokuRemote.left();
      commandName = "LEFT";
    } else if (lower.contains("right")) {
      await _rokuRemote.right();
      commandName = "RIGHT";
    } else if (lower.contains("select") || lower.contains("okay") || lower.contains("ok")) {
      await _rokuRemote.select();
      commandName = "SELECT";
    } else if (lower.contains("home")) {
      await _rokuRemote.home();
      commandName = "HOME";
    } else if (lower.contains("back")) {
      await _rokuRemote.back();
      commandName = "BACK";
    }

    if (commandName != null) {
      _log("🎮 RokuModeHandler: Sent '$commandName' command.");
      await _showFeedbackAnimation(commandName);
    } else {
      _log("🎮 RokuModeHandler: Unrecognized command: '$transcript'");
      await _showFeedbackAnimation("'$transcript' not recognized");
    }

    return const ModeResult(
      continueListening: true,
      displayText: null, // We handle display ourselves
    );
  }

  /// Get animated dots string based on count
  String _getDots() {
    switch (_dotCount) {
      case 1: return ".";
      case 2: return "..";
      case 3: return "...";
      default: return ".";
    }
  }

  /// Build the display text with current animation state
  String _buildWaitingDisplay() {
    return "$commandLine\n\nWaiting for Command${_getDots()}";
  }

  /// Build the feedback display text
  String _buildFeedbackDisplay(String command) {
    return "$commandLine\n\n$command heard${_getDots()}";
  }

  /// Start the waiting animation (cycles dots)
  Future<void> _startWaitingAnimation() async {
    _stopAnimation();
    _dotCount = 1;

    // Send initial display
    final text = _buildWaitingDisplay();
    EvenAI.updateDynamicText(text);
    await TextService.get.startSendText(text);

    // Start animation timer - cycle every 500ms
    _animationTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      if (!_isActive) {
        timer.cancel();
        return;
      }
      
      _dotCount = (_dotCount % 3) + 1; // Cycle 1 -> 2 -> 3 -> 1
      final text = _buildWaitingDisplay();
      EvenAI.updateDynamicText(text);
      await TextService.get.startSendText(text);
    });
  }

  /// Show feedback animation for 1 second, then return to waiting
  Future<void> _showFeedbackAnimation(String command) async {
    _stopAnimation();
    _dotCount = 1;

    // Show initial feedback
    final text = _buildFeedbackDisplay(command);
    EvenAI.updateDynamicText(text);
    await TextService.get.startSendText(text);

    // Animate for 1 second (2 cycles)
    int cycles = 0;
    _animationTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      cycles++;
      if (cycles >= 2 || !_isActive) {
        timer.cancel();
        // Return to waiting animation after feedback
        if (_isActive) {
          await _startWaitingAnimation();
        }
        return;
      }
      
      _dotCount = (_dotCount % 3) + 1;
      final text = _buildFeedbackDisplay(command);
      EvenAI.updateDynamicText(text);
      await TextService.get.startSendText(text);
    });
  }

  void _stopAnimation() {
    _animationTimer?.cancel();
    _animationTimer = null;
  }

  void _startAutoExitTimer() {
    _autoExitTimer?.cancel();
    _autoExitTimer = Timer(Duration(seconds: autoExitSeconds), () {
      _log("🎮 RokuModeHandler: Auto-exit after $autoExitSeconds seconds (2 min).");
      reset();
    });
  }

  @override
  void reset() {
    _log("🎮 RokuModeHandler: Resetting mode.");
    _isActive = false;
    _autoExitTimer?.cancel();
    _autoExitTimer = null;
    _feedbackTimer?.cancel();
    _feedbackTimer = null;
    _stopAnimation();
  }
}
