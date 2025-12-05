import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:demo_ai_even/services/auth_service.dart';

/// GatewayRagClient talks to Nemo Gateway RAG + Gemma endpoints.
/// It is optional and guarded by USE_GATEWAY_RAG in .env.
class GatewayRagClient {
  final Dio _dio;
  final String _baseUrl;
  final String _bearerToken = (dotenv.env['GATEWAY_BEARER_TOKEN'] ?? '').trim();

  factory GatewayRagClient() {
    final envBase = (dotenv.env['GATEWAY_BASE_URL'] ?? '').trim();
    final fallback = (dotenv.env['WHISPER_SERVER_BASE'] ?? '').trim();
    final base = envBase.isNotEmpty ? envBase : fallback;

    final dio = Dio(
      BaseOptions(
        baseUrl: base,
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 30),
      ),
    );

    // Basic logging for troubleshooting
    dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
      logPrint: (obj) => print("🌐 [Gateway] $obj"),
    ));
    return GatewayRagClient._internal(base, dio);
  }

  GatewayRagClient._internal(this._baseUrl, this._dio);

  void debugPrintConfig() {
    print(
        "GatewayRagClient baseUrl=$_baseUrl bearerSet=${_bearerToken.isNotEmpty}");
  }

  DioException _wrap(String stage, DioException e) {
    final status = e.response?.statusCode;
    print(
        "GatewayRagClient: $stage failed: status=$status data=${e.response?.data}");
    return e;
  }

  Map<String, String> _authHeaders() {
    if (_bearerToken.isNotEmpty) {
      print("GatewayRagClient: using bearer token");
      return {"Authorization": "Bearer $_bearerToken"};
    }
    final session = AuthService.instance.sessionToken;
    final csrf = AuthService.instance.csrfToken;
    if (session != null && session.isNotEmpty) {
      final headers = <String, String>{};
      // Supply both cookie and bearer forms for compatibility with gateway middleware.
      headers["Authorization"] = "Bearer $session";
      final cookieParts = <String>["ws_session=$session"];
      if (csrf != null && csrf.isNotEmpty) {
        headers["X-CSRF-Token"] = csrf;
        // Gateway expects ws_csrf; include legacy name for safety.
        cookieParts.add("ws_csrf=$csrf");
        cookieParts.add("csrf_token=$csrf");
      }
      headers["Cookie"] = cookieParts.join("; ");
      print(
          "GatewayRagClient: using session auth headers (session + csrf? ${csrf != null && csrf.isNotEmpty}) cookie=${headers["Cookie"]} xcsrf=${headers["X-CSRF-Token"] ?? ''}");
      return headers;
    }
    print("GatewayRagClient: no auth headers set (bearer and session missing)");
    return {};
  }

  Future<List<Map<String, dynamic>>> fetchContext(String question) async {
    // Normalize query: strip wake word noise and expand simple synonyms.
    String normalized = question.trim();
    // Remove a leading "memory" wake word if captured in transcripts.
    normalized = normalized.replaceFirst(RegExp(r"^\s*memory[\s,:-]*", caseSensitive: false), "").trim();

    String keywordTokens = normalized.toLowerCase();
    try {
      // Extract alphanumeric tokens >2 chars and add light synonyms.
      final tokens = <String>{};
      final re = RegExp(r"[a-z0-9]+");
      for (final m in re.allMatches(keywordTokens)) {
        final t = m.group(0)!;
        if (t.length > 2) tokens.add(t);
      }
      // Simple synonym swaps for better recall.
      if (tokens.contains("mother")) tokens.add("mom");
      if (tokens.contains("mom")) tokens.add("mother");
      if (tokens.contains("freedom")) tokens.add("liberty"); // help freedom queries
      if (tokens.isNotEmpty) {
        // Match CLI behavior: comma-separated keywords for gateway fallback handlers.
        keywordTokens = tokens.join(",");
      }
    } catch (_) {
      // If tokenization fails, fall back to the raw normalized question.
      keywordTokens = normalized;
    }

    // Primary: keyword query on transcripts
    final keywordPayload = {
      "keywords": keywordTokens,
      "search_type": "keyword",
      // Mirror CLI interactive defaults for focused context
      "limit": 3,
      "context_lines": 10,
      "last_n_transcripts": 1000,
      "speakers": ["ericah", "pruitt"],
    };

    final semanticFallback = {
      "query": question,
      "top_k": 50,
      "with_context": true,
      "last_n_transcripts": 1000,
      "speakers": ["ericah", "pruitt"],
    };

    // Try keyword first
    try {
      final resp = await _dio
          .post(
            "/api/transcripts/query",
            data: keywordPayload,
            options: Options(headers: _authHeaders()),
          )
          .timeout(const Duration(seconds: 12));
      print(
          "GatewayRagClient keyword resp status=${resp.statusCode} data=${resp.data}");
      final results = (resp.data?['results'] ??
              resp.data?['items'] ?? // gateway returns `items`
              resp.data?['data'] ??
              []) as List;
      if (results.isNotEmpty) {
        return results.cast<Map<String, dynamic>>();
      }
    } on DioException catch (e) {
      throw _wrap("keyword query", e);
    } catch (e) {
      print(
          "GatewayRagClient: keyword query failed, trying semantic. Error: $e");
    }

    // Semantic fallback
    try {
      final resp = await _dio
          .post(
            "/api/search/semantic",
            data: semanticFallback,
            options: Options(headers: _authHeaders()),
          )
          .timeout(const Duration(seconds: 12));
      print(
          "GatewayRagClient semantic resp status=${resp.statusCode} data=${resp.data}");
      final results = (resp.data?['results'] ??
              resp.data?['items'] ?? // gateway returns `items`
              resp.data?['data'] ??
              []) as List;
      return results.cast<Map<String, dynamic>>();
    } on DioException catch (e) {
      throw _wrap("semantic query", e);
    } catch (e) {
      print("GatewayRagClient: semantic query failed (non-Dio). Error: $e");
      rethrow;
    }
  }

  Future<String?> askGemma(String prompt) async {
    final payload = {
      "messages": [
        {
          "role": "system",
          "content":
              "You are a concise assistant. Use the provided transcript snippets only. If unclear, say you don't know."
        },
        {"role": "user", "content": prompt},
      ],
      "max_tokens": 200,
      "temperature": 0.4,
    };

    try {
      final resp = await _dio
          .post(
            "/api/gemma/chat",
            data: payload,
            options: Options(headers: _authHeaders()),
          )
          .timeout(const Duration(seconds: 15));
      print(
          "GatewayRagClient gemma resp status=${resp.statusCode} data=${resp.data}");

      if (resp.statusCode != 200) return null;

      // Normalize the response shape. The gateway can return either:
      // - OpenAI-style choices list
      // - A flat {"message": "..."} payload
      // - A plain string body
      final data = resp.data;
      String? _extractMessage(dynamic raw) {
        if (raw == null) return null;
        if (raw is String) return raw;
        if (raw is Map<String, dynamic>) {
          final fromChoices =
              raw['choices']?[0]?['message']?['content']?.toString();
          if (fromChoices != null && fromChoices.trim().isNotEmpty) {
            return fromChoices;
          }
          final flat = raw['message']?.toString();
          if (flat != null && flat.trim().isNotEmpty) return flat;
          // Some servers nest the message under `data` or `result`.
          final nested = raw['data']?.toString() ?? raw['result']?.toString();
          if (nested != null && nested.trim().isNotEmpty) return nested;
        }
        // Fallback to toString() so callers never see a null when a body exists.
        return raw.toString();
      }

      final normalized = _extractMessage(data);
      return (normalized != null && normalized.trim().isNotEmpty)
          ? normalized.trim()
          : null;
    } on DioException catch (e) {
      throw _wrap("gemma chat", e);
    }
  }
}
