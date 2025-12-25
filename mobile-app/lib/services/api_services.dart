import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

class ApiService {
  late Dio _dio;

  ApiService() {
    _dio = Dio(
      BaseOptions(
        baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        headers: {
          'Authorization':
              'Bearer ${const String.fromEnvironment("DASHSCOPE_API_KEY", defaultValue: "")}', // Set via --dart-define=DASHSCOPE_API_KEY=xxx
          'Content-Type': 'application/json',
        },
      ),
    );
  }

  Future<String> sendChatRequest(String question) async {
    final data = {
      "model": "qwen-plus",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
      ],
    };
    if (kDebugMode) {
      debugPrint("sendChatRequest------data----------$data--------");
    }

    try {
      final response = await _dio.post('/chat/completions', data: data);

      if (response.statusCode == 200) {
        if (kDebugMode) {
          debugPrint("Response: ${response.data}");
        }

        final data = response.data;
        final content = data['choices']?[0]?['message']?['content'] ??
            "Unable to answer the question";
        return content;
      } else {
        if (kDebugMode) {
          debugPrint("Request failed with status: ${response.statusCode}");
        }
        return "Request failed with status: ${response.statusCode}";
      }
    } on DioException catch (e) {
      final status = e.response?.statusCode;
      final data = e.response?.data;
      if (status != null) {
        if (kDebugMode) {
          debugPrint("Error: $status, $data");
        }
        return "AI request error: $status, $data";
      }
      final message = e.message ?? e.error?.toString() ?? 'Unknown error';
      if (kDebugMode) {
        debugPrint("Error: $message");
      }
      return "AI request error: $message";
    }
  }
}
