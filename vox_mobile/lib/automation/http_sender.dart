import 'package:dio/dio.dart';

class HttpRequestSpec {
  const HttpRequestSpec({
    required this.method,
    required this.url,
    required this.headers,
    required this.body,
    this.timeout = const Duration(seconds: 10),
  });

  final String method;
  final Uri url;
  final Map<String, String> headers;
  final String body;
  final Duration timeout;
}

class HttpResult {
  const HttpResult(this.statusCode);

  final int statusCode;

  bool get isSuccess => statusCode >= 200 && statusCode < 300;
}

/// Seam for sending webhooks so tests can avoid the network.
abstract interface class HttpSender {
  /// Returns any HTTP response (including 4xx/5xx). Throws on network failure.
  Future<HttpResult> send(HttpRequestSpec spec);
}

class DioHttpSender implements HttpSender {
  DioHttpSender({Dio? dio}) : _dio = dio ?? Dio();

  final Dio _dio;

  @override
  Future<HttpResult> send(HttpRequestSpec spec) async {
    final response = await _dio.requestUri<void>(
      spec.url,
      data: spec.method.toUpperCase() == 'GET' ? null : spec.body,
      options: Options(
        method: spec.method.toUpperCase(),
        headers: spec.headers,
        sendTimeout: spec.timeout,
        receiveTimeout: spec.timeout,
        validateStatus: (_) => true,
        // Never forward a (possibly signed) request to another host.
        followRedirects: false,
        responseType: ResponseType.plain,
      ),
    );
    return HttpResult(response.statusCode ?? 0);
  }
}
