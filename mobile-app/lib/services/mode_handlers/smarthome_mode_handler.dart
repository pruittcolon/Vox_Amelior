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

/// Mode handler for direct smart home control via voice commands.
///
/// Unlike AlexaModeHandler which requires saying "Alexa" first,
/// this handler triggers immediately when it detects light/device commands.
///
/// Supported commands:
/// - "kitchen light off" / "kitchen lights off"
/// - "bedroom light off"
/// - "all lights off" / "lights off"
/// - "tv off"
class SmartHomeModeHandler extends ModeHandler with WordMatching {
  final AlexaService _alexaService;

  bool _isActive = false;

  /// Trigger phrases that activate this mode directly
  static const Map<String, String> triggerCommands = {
    'kitchen light off': 'kitchenlights',
    'kitchen lights off': 'kitchenlights',
    'turn off kitchen light': 'kitchenlights',
    'turn off the kitchen light': 'kitchenlights',
    'bedroom light off': 'bedroomlight',
    'bedroom lights off': 'bedroomlight',
    'turn off bedroom light': 'bedroomlight',
    'turn off the bedroom light': 'bedroomlight',
    'all lights off': 'alllights',
    'lights off': 'alllights',
    'turn off the lights': 'alllights',
    'turn off all lights': 'alllights',
    'tv off': 'tvoff',
    'turn off the tv': 'tvoff',
    'turn off tv': 'tvoff',
  };

  SmartHomeModeHandler({AlexaService? alexaService})
      : _alexaService = alexaService ?? AlexaService.instance;

  @override
  String get modeName => 'smarthome';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    final lower = transcript.toLowerCase();
    return triggerCommands.keys.any((phrase) => lower.contains(phrase));
  }

  @override
  bool canHandleInMode(String transcript) {
    // This handler processes commands immediately and exits
    return false;
  }

  @override
  bool isTerminateCommand(String transcript) {
    return false; // No termination needed - single command mode
  }

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _log("🏠 SmartHomeModeHandler: Processing command: '$transcript'");
    _isActive = true;

    final lower = transcript.toLowerCase();
    
    // Find matching command
    String? deviceName;
    for (final entry in triggerCommands.entries) {
      if (lower.contains(entry.key)) {
        deviceName = entry.value;
        _log("🏠 SmartHomeModeHandler: Matched '${entry.key}' -> '$deviceName'");
        break;
      }
    }

    if (deviceName != null) {
      final success = await _alexaService.triggerRoutine(deviceName, transcript: transcript);
      
      _isActive = false; // Immediately deactivate after processing
      
      final displayText = success 
          ? "✅ $deviceName" 
          : "❌ Failed: $deviceName";
      
      _log("🏠 SmartHomeModeHandler: Trigger result - success: $success");
      
      return ModeResult(
        continueListening: true,
        displayText: displayText,
        handled: true,
      );
    } else {
      _log("🏠 SmartHomeModeHandler: No matching command found");
      _isActive = false;
      return ModeResult.notHandled;
    }
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    // Should not be called since we exit immediately
    return ModeResult.notHandled;
  }

  @override
  void reset() {
    _log("🏠 SmartHomeModeHandler: Resetting mode.");
    _isActive = false;
  }
}
