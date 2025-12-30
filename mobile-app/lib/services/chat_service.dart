import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for handling AI chat requests via OpenAI API.
///
/// Requires OPENAI_API_KEY to be set in .env file.
class ChatService {
  static ChatService? _instance;
  static ChatService get instance => _instance ??= ChatService._();

  late final Dio _dio;
  late final String? _apiKey;

  /// Default model - latest GPT-4o (November 2024)
  static const String defaultModel = 'gpt-4o-2024-11-20';

  /// Advanced model for complex queries
  static const String advancedModel = 'gpt-4o-2024-11-20';

  /// System prompt for the AI assistant - optimized for smart glasses display
  static const String systemPrompt = 
      'You are a helpful assistant for smart glasses. '
      'Give SHORT and CLEAR responses. '
      'Be concise - limit answers to 2-3 short sentences when possible. '
      'Avoid long explanations unless specifically asked.';

  /// System prompt for advanced mode - more thorough
  static const String advancedSystemPrompt = 
      'You are an advanced AI assistant. Provide thorough, well-reasoned responses. '
      'Take your time to think through complex questions carefully.';

  ChatService._() {
    _apiKey = dotenv.env['OPENAI_API_KEY'];

    _dio = Dio(
      BaseOptions(
        baseUrl: 'https://api.openai.com/v1',
        headers: {
          'Authorization': 'Bearer $_apiKey',
          'Content-Type': 'application/json',
        },
        connectTimeout: const Duration(seconds: 15),
        receiveTimeout: const Duration(seconds: 45), // Longer for advanced
      ),
    );

    if (_apiKey == null || _apiKey!.isEmpty) {
      _log("⚠️ ChatService: OPENAI_API_KEY not set in .env file.");
    }
  }

  /// Check if the service is properly configured
  bool get isConfigured => _apiKey != null && _apiKey!.isNotEmpty;

  /// Check if query has 'advanced' trigger
  static bool hasAdvancedTrigger(String query) {
    return RegExp(r'\badvanced\b', caseSensitive: false).hasMatch(query);
  }

  /// Strip 'advanced' trigger from query
  static String stripAdvancedTrigger(String query) {
    return query.replaceAll(RegExp(r'\badvanced\b[,\s]*', caseSensitive: false), '').trim();
  }

  /// Send a regular chat request (5.1mini mode)
  Future<String> sendChatRequest(String question) async {
    if (!isConfigured) {
      return "Error: OpenAI API key not configured.";
    }

    _log("🤖 ChatService: Using model '$defaultModel' (5.1mini mode).");

    final data = {
      "model": defaultModel,
      "messages": [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": question}
      ],
    };

    try {
      final response = await _dio.post('/chat/completions', data: data);

      if (response.statusCode == 200) {
        final responseData = response.data;
        final content = responseData['choices']?[0]?['message']?['content'] ??
            "Unable to answer the question";
        _log("✅ ChatService: Received response (5.1mini).");
        return content;
      } else {
        _log("❌ ChatService: Request failed with status: ${response.statusCode}");
        return "Request failed with status: ${response.statusCode}";
      }
    } on DioException catch (e) {
      if (e.response != null) {
        _log("❌ ChatService: API error: ${e.response?.statusCode}");
        return "AI request error: ${e.response?.statusCode}, ${e.response?.data}";
      } else {
        _log("❌ ChatService: Network error: ${e.message}");
        return "AI request error: ${e.message}";
      }
    }
  }

  /// Send an advanced chat request (5.1 mode - more thorough)
  Future<String> sendAdvancedChatRequest(String question) async {
    if (!isConfigured) {
      return "Error: OpenAI API key not configured.";
    }

    _log("🧠 ChatService: Using model '$advancedModel' (5.1 advanced mode).");

    final data = {
      "model": advancedModel,
      "messages": [
        {"role": "system", "content": advancedSystemPrompt},
        {"role": "user", "content": question}
      ],
    };

    try {
      final response = await _dio.post('/chat/completions', data: data);

      if (response.statusCode == 200) {
        final responseData = response.data;
        final content = responseData['choices']?[0]?['message']?['content'] ??
            "Unable to answer the question";
        _log("✅ ChatService: Received response (5.1 advanced).");
        return content;
      } else {
        _log("❌ ChatService: Request failed with status: ${response.statusCode}");
        return "Request failed with status: ${response.statusCode}";
      }
    } on DioException catch (e) {
      if (e.response != null) {
        _log("❌ ChatService: API error: ${e.response?.statusCode}");
        return "AI request error: ${e.response?.statusCode}, ${e.response?.data}";
      } else {
        _log("❌ ChatService: Network error: ${e.message}");
        return "AI request error: ${e.message}";
      }
    }
  }

  /// Send a chat request with custom system prompt
  Future<String> sendChatRequestWithPrompt(
      String question, String customSystemPrompt) async {
    if (!isConfigured) {
      return "Error: OpenAI API key not configured.";
    }

    final data = {
      "model": defaultModel,
      "messages": [
        {"role": "system", "content": customSystemPrompt},
        {"role": "user", "content": question}
      ],
    };

    try {
      final response = await _dio.post('/chat/completions', data: data);

      if (response.statusCode == 200) {
        final responseData = response.data;
        return responseData['choices']?[0]?['message']?['content'] ??
            "Unable to answer the question";
      } else {
        return "Request failed with status: ${response.statusCode}";
      }
    } on DioException catch (e) {
      return "AI request error: ${e.message}";
    }
  }
}
