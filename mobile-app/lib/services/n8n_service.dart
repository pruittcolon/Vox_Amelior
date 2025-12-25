import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for integrating with the n8n Integration Service (port 8011).
///
/// Sends transcript segments to the n8n-service for:
/// - Voice command pattern matching
/// - Emotion tracking and alerts
/// - Centralized automation workflows
class N8nService {
  static N8nService? _instance;
  static N8nService get instance => _instance ??= N8nService._();

  late final Dio _dio;
  late final String? _n8nServiceUrl;

  N8nService._() {
    _n8nServiceUrl = dotenv.env['N8N_SERVICE_URL'];
    _dio = Dio(
      BaseOptions(
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 10),
        headers: {'Content-Type': 'application/json'},
      ),
    );

    if (_n8nServiceUrl == null || _n8nServiceUrl!.isEmpty) {
      _log("⚠️ N8nService: N8N_SERVICE_URL not set in .env file.");
    } else {
      _log("✅ N8nService: Configured with URL: $_n8nServiceUrl");
    }
  }

  /// Check if the service is properly configured
  bool get isConfigured => _n8nServiceUrl != null && _n8nServiceUrl!.isNotEmpty;

  /// Process a transcript segment through the n8n service.
  ///
  /// The n8n-service will:
  /// - Match voice commands and trigger Voice Monkey
  /// - Track emotions per speaker
  /// - Fire n8n webhooks for automation workflows
  Future<N8nProcessResult> processTranscript({
    required String text,
    String speaker = 'unknown',
    String? emotion,
    double? emotionConfidence,
    double startTime = 0.0,
    double endTime = 0.0,
    String? sessionId,
  }) async {
    if (!isConfigured) {
      _log("❌ N8nService: Cannot process - URL not configured.");
      return N8nProcessResult.notConfigured();
    }

    try {
      final response = await _dio.post(
        '$_n8nServiceUrl/process',
        data: {
          'segments': [
            {
              'text': text,
              'speaker': speaker,
              'emotion': emotion,
              'emotion_confidence': emotionConfidence,
              'start_time': startTime,
              'end_time': endTime,
            }
          ],
          'session_id': sessionId,
        },
      );

      if (response.statusCode == 200) {
        final data = response.data;
        if (kDebugMode) {
          debugPrint("✅ N8nService: Processed segment. "
              "Commands: ${data['voice_commands_triggered']}, "
              "Alerts: ${data['emotion_alerts_triggered']}");
        }

        return N8nProcessResult(
          success: true,
          voiceCommandsTriggered: data['voice_commands_triggered'] ?? 0,
          emotionAlertsTriggered: data['emotion_alerts_triggered'] ?? 0,
          details: List<Map<String, dynamic>>.from(data['details'] ?? []),
        );
      } else {
        _log("❌ N8nService: Failed with status ${response.statusCode}");
        return N8nProcessResult.error("Status ${response.statusCode}");
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        _log("⚠️ N8nService: Cannot connect to n8n-service at $_n8nServiceUrl");
        return N8nProcessResult.connectionError();
      }
      _log("❌ N8nService: Error: ${e.message}");
      return N8nProcessResult.error(e.message ?? 'Unknown error');
    } catch (e) {
      _log("❌ N8nService: Error: $e");
      return N8nProcessResult.error(e.toString());
    }
  }

  /// Check if the n8n-service is reachable
  Future<bool> healthCheck() async {
    if (!isConfigured) return false;

    try {
      final response = await _dio.get('$_n8nServiceUrl/health');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Get list of registered voice commands from n8n-service
  Future<List<String>> getAvailableCommands() async {
    if (!isConfigured) return [];

    try {
      final response = await _dio.get('$_n8nServiceUrl/commands');
      if (response.statusCode == 200) {
        final commands = List<Map<String, dynamic>>.from(response.data);
        return commands.map((c) => c['description'] as String).toList();
      }
    } catch (e) {
      _log("⚠️ N8nService: Could not fetch commands: $e");
    }
    return [];
  }
}

/// Result of processing a transcript through n8n-service
class N8nProcessResult {
  final bool success;
  final bool configured;
  final bool connectionError;
  final int voiceCommandsTriggered;
  final int emotionAlertsTriggered;
  final List<Map<String, dynamic>> details;
  final String? errorMessage;

  N8nProcessResult({
    required this.success,
    this.configured = true,
    this.connectionError = false,
    this.voiceCommandsTriggered = 0,
    this.emotionAlertsTriggered = 0,
    this.details = const [],
    this.errorMessage,
  });

  factory N8nProcessResult.notConfigured() => N8nProcessResult(
        success: false,
        configured: false,
      );

  factory N8nProcessResult.connectionError() => N8nProcessResult(
        success: false,
        connectionError: true,
      );

  factory N8nProcessResult.error(String message) => N8nProcessResult(
        success: false,
        errorMessage: message,
      );

  bool get hasTriggeredCommands => voiceCommandsTriggered > 0;
  bool get hasTriggeredAlerts => emotionAlertsTriggered > 0;
}
