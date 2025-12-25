import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:demo_ai_even/services/n8n_service.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for triggering Alexa routines via Voicemonkey API.
///
/// Requires VOICEMONKEY_TRIGGER_URL to be set in .env file.
/// Optionally integrates with n8n-service for centralized logging.
class AlexaService {
  static AlexaService? _instance;
  static AlexaService get instance => _instance ??= AlexaService._();

  late final Dio _dio;
  late final String? _voicemonkeyUrl;
  late final N8nService _n8nService;

  /// Map of recognized voice commands to Voicemonkey device names
  /// Includes common transcription error variants (chicken/kitchen, lite/light, etc.)
  static const Map<String, String> deviceMappings = {
    // All lights
    'all lights': 'alllights',
    'all lights off': 'alllights',
    'all lights on': 'alllights',
    'lights off': 'alllights',
    'lights on': 'alllights',
    
    // Bedroom light
    'bedroom light': 'bedroomlight',
    'bedroom light off': 'bedroomlight',
    'bedroom light on': 'bedroomlight',
    
    // TV
    'tv off': 'tvoff',
    
    // Kitchen light - off (toggle device)
    'kitchen light off': 'kitchenlights',
    'kitchen lights off': 'kitchenlights',
    'chicken light off': 'kitchenlights',  // transcription error
    'chicken lights off': 'kitchenlights', // transcription error
    
    // Kitchen light - on (toggle device)
    'kitchen light on': 'kitchenlights',
    'kitchen lights on': 'kitchenlights',
    'chicken light on': 'kitchenlights',   // transcription error
    'chicken lights on': 'kitchenlights',  // transcription error
    
    // Kitchen light - generic
    'kitchen lights': 'kitchenlights',
    'kitchen light': 'kitchenlights',
    'kitchenlights': 'kitchenlights',
    
    // Living room light - off
    'living room light off': 'livingroomlight',
    'living room lights off': 'livingroomlight',
    'leaving room light off': 'livingroomlight',  // transcription error
    
    // Living room light - on
    'living room light on': 'livingroomlight',
    'living room lights on': 'livingroomlight',
    'leaving room light on': 'livingroomlight',   // transcription error
    
    // Living room light - generic
    'living room light': 'livingroomlight',
    'living room lights': 'livingroomlight',
  };

  AlexaService._() {
    _voicemonkeyUrl = dotenv.env['VOICEMONKEY_TRIGGER_URL'];
    _n8nService = N8nService.instance;
    _dio = Dio(
      BaseOptions(
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 10),
      ),
    );

    if (_voicemonkeyUrl == null || _voicemonkeyUrl!.isEmpty) {
      _log("⚠️ AlexaService: VOICEMONKEY_TRIGGER_URL not set in .env file.");
    }
  }

  /// Check if the service is properly configured
  bool get isConfigured =>
      _voicemonkeyUrl != null && _voicemonkeyUrl!.isNotEmpty;

  /// Check if n8n service is also configured
  bool get isN8nConfigured => _n8nService.isConfigured;

  /// Find the device name for a given voice command transcript
  String? findDeviceForCommand(String transcript) {
    final lower = transcript.toLowerCase();
    for (final entry in deviceMappings.entries) {
      if (lower.contains(entry.key)) {
        return entry.value;
      }
    }
    return null;
  }

  /// Triggers an Alexa routine via Voicemonkey GET request.
  ///
  /// Also sends the transcript to n8n-service for centralized logging
  /// and additional workflow processing.
  ///
  /// Returns true if the routine was triggered successfully.
  Future<bool> triggerRoutine(String deviceName, {String? transcript}) async {
    if (!isConfigured) {
      _log("❌ AlexaService: Cannot trigger routine - URL not configured.");
      return false;
    }

    final String encodedDevice = Uri.encodeComponent(deviceName);
    final String fullUrl = "$_voicemonkeyUrl&device=$encodedDevice";

    // Also notify n8n-service for centralized logging (fire and forget)
    if (_n8nService.isConfigured && transcript != null) {
      _n8nService.processTranscript(
        text: transcript,
        speaker: 'user',
      ).then((result) {
        if (result.hasTriggeredCommands) {
          _log("📊 AlexaService: n8n logged ${result.voiceCommandsTriggered} commands");
        }
      }).catchError((e) {
        _log("⚠️ AlexaService: n8n logging failed: $e");
      });
    }

    try {
      _log("🏠 AlexaService: Triggering routine for '$deviceName'...");
      final response = await _dio.get(fullUrl);

      if (response.statusCode == 200) {
        _log("✅ AlexaService: Successfully triggered '$deviceName'.");
        return true;
      } else {
        if (kDebugMode) {
          debugPrint(
              "❌ AlexaService: Failed to trigger '$deviceName'. Status: ${response.statusCode}");
        }
        return false;
      }
    } catch (e) {
      _log("❌ AlexaService: Error triggering '$deviceName': $e");
      return false;
    }
  }

  /// Get the list of available voice commands for display
  String getAvailableCommands() {
    return "All Lights Off, Kitchen Light Off, Bedroom Light Off, TV Off";
  }
}
