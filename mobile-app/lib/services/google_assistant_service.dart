import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';

// Debug logging helper - tree-shaken in release builds
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for Google Assistant-like functionality using native Android intents.
///
/// Provides methods for:
/// - Launching Google Assistant
/// - Setting timers and alarms
/// - Making calls and sending messages
/// - Web search and navigation
///
/// Uses a platform channel to communicate with native Android code.
class GoogleAssistantService {
  static GoogleAssistantService? _instance;
  static GoogleAssistantService get instance =>
      _instance ??= GoogleAssistantService._();

  static const _channel =
      MethodChannel('com.example.demo_ai_even/google_assistant');

  GoogleAssistantService._();

  /// Launch Google Assistant in listening mode.
  ///
  /// Opens the native Google Assistant UI ready to accept voice input.
  /// Returns true if successful, false otherwise.
  Future<bool> launchAssistant() async {
    try {
      final result = await _channel.invokeMethod('launchAssistant');
      _log('🤖 GoogleAssistantService: Launched Assistant');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to launch Assistant: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error launching Assistant: $e');
      return false;
    }
  }

  /// Set a countdown timer.
  ///
  /// [seconds] - Duration of the timer in seconds
  /// [message] - Optional label for the timer (default: "Timer")
  ///
  /// Opens the device's Clock app with the timer preset.
  /// Returns true if successful, false otherwise.
  Future<bool> setTimer({required int seconds, String? message}) async {
    try {
      final result = await _channel.invokeMethod('setTimer', {
        'seconds': seconds,
        'message': message ?? 'Timer',
      });
      _log('⏱️ GoogleAssistantService: Set timer for $seconds seconds');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to set timer: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error setting timer: $e');
      return false;
    }
  }

  /// Set an alarm.
  ///
  /// [hour] - Hour in 24-hour format (0-23)
  /// [minute] - Minute (0-59)
  /// [message] - Optional label for the alarm (default: "Alarm")
  ///
  /// Opens the device's Clock app with the alarm preset.
  /// Returns true if successful, false otherwise.
  Future<bool> setAlarm({
    required int hour,
    required int minute,
    String? message,
  }) async {
    try {
      final result = await _channel.invokeMethod('setAlarm', {
        'hour': hour,
        'minute': minute,
        'message': message ?? 'Alarm',
      });
      _log('⏰ GoogleAssistantService: Set alarm for $hour:${minute.toString().padLeft(2, '0')}');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to set alarm: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error setting alarm: $e');
      return false;
    }
  }

  /// Open phone dialer with a phone number.
  ///
  /// [phoneNumber] - The phone number to dial (can include dashes, spaces, etc.)
  ///
  /// Opens the device's Phone app with the number ready to dial.
  /// Does NOT automatically start the call (user must confirm).
  /// Returns true if successful, false otherwise.
  Future<bool> makeCall(String phoneNumber) async {
    try {
      final result = await _channel.invokeMethod('makeCall', {
        'number': phoneNumber,
      });
      _log('📞 GoogleAssistantService: Opening dialer for $phoneNumber');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to open dialer: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error making call: $e');
      return false;
    }
  }

  /// Open SMS app with an optional recipient and message.
  ///
  /// [number] - Optional recipient phone number
  /// [message] - Optional pre-filled message text
  ///
  /// Opens the device's default messaging app.
  /// Returns true if successful, false otherwise.
  Future<bool> sendMessage({String? number, String? message}) async {
    try {
      final result = await _channel.invokeMethod('sendMessage', {
        'number': number ?? '',
        'message': message ?? '',
      });
      _log('💬 GoogleAssistantService: Opening messaging${number != null ? ' for $number' : ''}');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to open messaging: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error sending message: $e');
      return false;
    }
  }

  /// Perform a web search.
  ///
  /// [query] - The search query
  ///
  /// Opens the device's default browser or search app with the query.
  /// Returns true if successful, false otherwise.
  Future<bool> webSearch(String query) async {
    try {
      final result = await _channel.invokeMethod('webSearch', {
        'query': query,
      });
      _log('🔍 GoogleAssistantService: Searching for "$query"');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to search: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error searching: $e');
      return false;
    }
  }

  /// Open Google Maps with a destination.
  ///
  /// [destination] - The destination to navigate to (address, place name, etc.)
  ///
  /// Opens Google Maps (or default maps app) with the destination.
  /// Returns true if successful, false otherwise.
  Future<bool> openMaps(String destination) async {
    try {
      final result = await _channel.invokeMethod('openMaps', {
        'query': destination,
      });
      _log('🗺️ GoogleAssistantService: Opening maps for "$destination"');
      return result == true;
    } on PlatformException catch (e) {
      _log('❌ GoogleAssistantService: Failed to open maps: ${e.message}');
      return false;
    } catch (e) {
      _log('❌ GoogleAssistantService: Unexpected error opening maps: $e');
      return false;
    }
  }
}
