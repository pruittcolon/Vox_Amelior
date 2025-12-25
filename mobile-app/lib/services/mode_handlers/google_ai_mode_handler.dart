import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/google_assistant_service.dart';
import 'package:demo_ai_even/services/command_parser.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for Google AI-like voice commands.
///
/// No wake word required - detects command patterns directly:
/// - Timers: "set timer for 5 minutes"
/// - Alarms: "set alarm for 7 AM", "wake me up at 6"
/// - Calls: "call mom"
/// - Messages: "text john hello"
/// - Search: "search for weather", "google restaurants"
/// - Navigation: "navigate to downtown", "directions to home"
///
/// Falls back to launching native Google Assistant for unrecognized commands.
class GoogleAIModeHandler extends ModeHandler
    with TerminateDetection, WordMatching {
  final GoogleAssistantService _gaService;

  bool _isActive = false;
  Timer? _autoExitTimer;

  /// Auto-exit timeout in seconds (no activity)
  static const int autoExitSeconds = 30;

  /// Command pattern prefixes that activate this mode (no wake word needed)
  static final List<RegExp> _commandPatterns = [
    RegExp(r'^(?:set\s+)?(?:a\s+)?timer\s+', caseSensitive: false),
    RegExp(r'^(?:set\s+)?(?:an?\s+)?alarm\s+', caseSensitive: false),
    RegExp(r'^wake\s+(?:me\s+)?up\s+', caseSensitive: false),
    RegExp(r'^(?:call|phone|dial)\s+', caseSensitive: false),
    RegExp(r'^(?:text|message|sms)\s+', caseSensitive: false),
    RegExp(r'^(?:search|google|look\s+up|find)\s+', caseSensitive: false),
    RegExp(r'^(?:navigate|directions|take\s+me|go\s+to|drive\s+to)\s+', caseSensitive: false),
    RegExp(r'^remind(?:er)?\s+', caseSensitive: false),
  ];

  GoogleAIModeHandler({GoogleAssistantService? gaService})
      : _gaService = gaService ?? GoogleAssistantService.instance;

  @override
  String get modeName => 'google_ai';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    // Strip trailing punctuation that speech-to-text might add
    var lower = transcript.toLowerCase().trim();
    lower = lower.replaceAll(RegExp(r'[.?!,]+$'), '').trim();
    // Check if transcript matches any command pattern
    return _commandPatterns.any((pattern) => pattern.hasMatch(lower));
  }

  @override
  bool canHandleInMode(String transcript) => _isActive;

  @override
  bool isTerminateCommand(String transcript) => matchesTerminate(transcript);

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _log("🤖 GoogleAIModeHandler: Processing command: '$transcript'");
    _isActive = true;
    _startAutoExitTimer();

    // Process the command directly (no trigger phrase to remove)
    return await handleCommand(transcript);
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // Check for termination
    if (isTerminateCommand(lower)) {
      _log("🤖 GoogleAIModeHandler: User terminated Google AI mode.");
      reset();
      return ModeResult.endSession;
    }

    // Reset auto-exit timer on activity
    _startAutoExitTimer();

    // Parse the command
    final parsed = CommandParser.parse(lower);

    if (parsed != null) {
      return await _executeCommand(parsed);
    }

    // No recognized command - launch native Google Assistant
    _log("🤖 GoogleAIModeHandler: Unrecognized command, launching Google Assistant");
    final success = await _gaService.launchAssistant();

    reset(); // Exit mode after launching Assistant

    return ModeResult(
      continueListening: false,
      displayText:
          success ? "🤖 Launching Google Assistant..." : "❌ Failed to launch Assistant",
    );
  }

  /// Execute a parsed command using the GoogleAssistantService.
  Future<ModeResult> _executeCommand(ParsedCommand cmd) async {
    bool success = false;
    String displayText = "";

    switch (cmd.type) {
      case CommandType.timer:
        final seconds = cmd.params['seconds'] as int;
        success = await _gaService.setTimer(seconds: seconds);
        displayText =
            success ? "⏱️ Timer: ${_formatDuration(seconds)}" : "❌ Failed to set timer";
        break;

      case CommandType.alarm:
        final hour = cmd.params['hour'] as int;
        final minute = cmd.params['minute'] as int;
        success = await _gaService.setAlarm(hour: hour, minute: minute);
        displayText =
            success ? "⏰ Alarm: ${_formatTime(hour, minute)}" : "❌ Failed to set alarm";
        break;

      case CommandType.call:
        final target = cmd.params['target'] as String;
        success = await _gaService.makeCall(target);
        displayText = success ? "📞 Calling: $target" : "❌ Failed to call";
        break;

      case CommandType.message:
        final recipient = cmd.params['recipient'] as String;
        final content = cmd.params['content'] as String?;
        success =
            await _gaService.sendMessage(number: recipient, message: content);
        displayText = success ? "💬 Message: $recipient" : "❌ Failed to message";
        break;

      case CommandType.search:
        final query = cmd.params['query'] as String;
        success = await _gaService.webSearch(query);
        displayText = success ? "🔍 Search: $query" : "❌ Failed to search";
        break;

      case CommandType.navigation:
        final dest = cmd.params['destination'] as String;
        success = await _gaService.openMaps(dest);
        displayText = success ? "🗺️ Navigate: $dest" : "❌ Failed to navigate";
        break;

      case CommandType.reminder:
        // Fall back to Google Assistant for reminders (requires Tasks API)
        success = await _gaService.launchAssistant();
        displayText = "🤖 Opening Assistant for reminder...";
        reset();
        return ModeResult(continueListening: false, displayText: displayText);
    }

    // Stay in mode after executing command
    return ModeResult(
      continueListening: true,
      displayText: displayText,
    );
  }

  /// Format a duration in seconds to a human-readable string.
  String _formatDuration(int seconds) {
    if (seconds >= 3600) {
      final hours = seconds ~/ 3600;
      final mins = (seconds % 3600) ~/ 60;
      return mins > 0
          ? "$hours hr $mins min"
          : "$hours hour${hours > 1 ? 's' : ''}";
    } else if (seconds >= 60) {
      final mins = seconds ~/ 60;
      final secs = seconds % 60;
      return secs > 0
          ? "$mins min $secs sec"
          : "$mins minute${mins > 1 ? 's' : ''}";
    }
    return "$seconds second${seconds > 1 ? 's' : ''}";
  }

  /// Format a time to a human-readable string (12-hour format).
  String _formatTime(int hour, int minute) {
    final period = hour >= 12 ? "PM" : "AM";
    final displayHour = hour > 12 ? hour - 12 : (hour == 0 ? 12 : hour);
    return "$displayHour:${minute.toString().padLeft(2, '0')} $period";
  }

  /// Start or reset the auto-exit timer.
  void _startAutoExitTimer() {
    _autoExitTimer?.cancel();
    _autoExitTimer = Timer(Duration(seconds: autoExitSeconds), () {
      _log("🤖 GoogleAIModeHandler: Auto-exit after $autoExitSeconds seconds of inactivity.");
      reset();
    });
  }

  @override
  void reset() {
    _log("🤖 GoogleAIModeHandler: Resetting mode.");
    _isActive = false;
    _autoExitTimer?.cancel();
    _autoExitTimer = null;
  }
}
