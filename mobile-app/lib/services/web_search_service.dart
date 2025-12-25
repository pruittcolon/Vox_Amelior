import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for handling web search-enabled AI requests via OpenAI API.
///
/// Uses the web_search_preview model for quick web lookups.
class WebSearchService {
  static WebSearchService? _instance;
  static WebSearchService get instance => _instance ??= WebSearchService._();

  late final Dio _dio;
  late final String? _apiKey;

  /// Model for web search queries - uses search-enabled model
  static const String webSearchModel = 'gpt-4o-search-preview';

  /// System prompt for web search - emphasis on brief, factual responses
  static const String systemPrompt = 
      'You are a helpful assistant with access to current web information. '
      'Provide brief, factual answers based on the most recent information. '
      'Keep responses concise - 2-3 sentences maximum. '
      'Focus on the key facts only.';

  WebSearchService._() {
    _apiKey = dotenv.env['OPENAI_API_KEY'];

    _dio = Dio(
      BaseOptions(
        baseUrl: 'https://api.openai.com/v1',
        headers: {
          'Authorization': 'Bearer $_apiKey',
          'Content-Type': 'application/json',
        },
        connectTimeout: const Duration(seconds: 20),
        receiveTimeout: const Duration(seconds: 45), // Longer for web search
      ),
    );

    if (_apiKey == null || _apiKey!.isEmpty) {
      _log("⚠️ WebSearchService: OPENAI_API_KEY not set in .env file.");
    }
  }

  /// Check if the service is properly configured
  bool get isConfigured => _apiKey != null && _apiKey!.isNotEmpty;

  /// Check if query contains Research trigger word
  /// Matches: research
  static bool hasUplinkTrigger(String query) {
    final pattern = RegExp(
      r'\bresearch\b',
      caseSensitive: false,
    );
    return pattern.hasMatch(query);
  }

  /// Remove the research trigger from the query to get the actual question
  static String stripUplinkTrigger(String query) {
    final pattern = RegExp(
      r'\bresearch\b[,\s]*',
      caseSensitive: false,
    );
    return query.replaceAll(pattern, '').trim();
  }

  /// Send a web search request to OpenAI.
  ///
  /// Uses the search-enabled model for current information.
  /// Returns an error message if the request fails.
  Future<String> sendWebSearchRequest(String question) async {
    if (!isConfigured) {
      return "Error: OpenAI API key not configured.";
    }

    // Strip the uplink trigger from the question
    final cleanQuestion = stripUplinkTrigger(question);
    _log("🌐 WebSearchService: Searching for: '$cleanQuestion'");

    // Use Chat Completions with search-enabled model
    // The gpt-4o-search-preview model automatically searches the web
    final data = {
      "model": webSearchModel,
      "messages": [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": cleanQuestion}
      ],
      "max_tokens": 300, // Keep responses brief
    };

    try {
      _log("🌐 WebSearchService: Calling $webSearchModel...");
      final response = await _dio.post('/chat/completions', data: data);

      if (response.statusCode == 200) {
        final responseData = response.data;
        final content = responseData['choices']?[0]?['message']?['content'] ??
            "Unable to find information.";
        _log("✅ WebSearchService: Received web search response.");
        return _formatResponse(content);
      } else {
        _log("❌ WebSearchService: Request failed with status: ${response.statusCode}");
        return "Web search failed: ${response.statusCode}";
      }
    } on DioException catch (e) {
      if (e.response != null) {
        _log("❌ WebSearchService: API error: ${e.response?.statusCode}");
        final errorData = e.response?.data;
        if (errorData != null && errorData['error'] != null) {
          final errorMsg = errorData['error']['message'] ?? 'Unknown error';
          return "Web search error: $errorMsg";
        }
        return "Web search error: ${e.response?.statusCode}";
      } else {
        _log("❌ WebSearchService: Network error: ${e.message}");
        return "Web search error: ${e.message}";
      }
    }
  }

  /// Format the response for glasses display
  /// Removes citations and keeps it brief
  String _formatResponse(String content) {
    // Remove inline citations like [1], [2], etc.
    String cleaned = content.replaceAll(RegExp(r'\[\d+\]'), '');
    
    // Remove URLs
    cleaned = cleaned.replaceAll(
      RegExp(r'https?://[^\s\)]+'), 
      ''
    );
    
    // Remove excessive whitespace
    cleaned = cleaned.replaceAll(RegExp(r'\s+'), ' ').trim();
    
    // Limit to reasonable length for glasses display
    if (cleaned.length > 500) {
      cleaned = '${cleaned.substring(0, 500)}...';
    }
    
    return cleaned;
  }
}
