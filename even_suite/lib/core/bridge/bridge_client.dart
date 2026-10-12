import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// A chat bubble from the Termux bridge (Agent Hub).
class BridgeMessage {
  final String id;
  final String who; // you, claude, agy
  final String to;
  final String text;
  final String status; // pending, done, error
  final double ts;

  const BridgeMessage({
    required this.id,
    required this.who,
    this.to = '',
    required this.text,
    required this.status,
    this.ts = 0,
  });

  factory BridgeMessage.fromMap(Map m) => BridgeMessage(
        id: '${m['id'] ?? ''}',
        who: '${m['who'] ?? ''}',
        to: '${m['to'] ?? ''}',
        text: '${m['text'] ?? ''}',
        status: '${m['status'] ?? 'done'}',
        ts: (m['ts'] as num?)?.toDouble() ?? 0,
      );

  bool get isMine => who == 'you';
  bool get isPending => status == 'pending';
  bool get isError => status == 'error';

  DateTime get time => DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
}

class BridgeSnapshot {
  final int version;
  final List<BridgeMessage> messages;
  final List<String> busy;

  const BridgeSnapshot(this.version, this.messages, this.busy);

  factory BridgeSnapshot.fromMap(Map m) => BridgeSnapshot(
        (m['version'] as num?)?.toInt() ?? 0,
        [for (final x in (m['messages'] as List? ?? const [])) BridgeMessage.fromMap(x as Map)],
        [for (final x in (m['busy'] as List? ?? const [])) '$x'],
      );
}

class TermScreen {
  final String text;
  final String hash;
  const TermScreen(this.text, this.hash);
}

class BridgeException implements Exception {
  final String message;
  const BridgeException(this.message);
  @override
  String toString() => message;
}

/// HTTP client for the Agent Hub server running in Termux
/// (`~/agent-hub/server.py`, default http://127.0.0.1:8787).
class BridgeClient {
  BridgeClient(String baseUrl) : _base = _clean(baseUrl);

  String _base;
  final HttpClient _http = HttpClient()
    ..connectionTimeout = const Duration(seconds: 3)
    ..idleTimeout = const Duration(seconds: 30);
  final Set<HttpClientRequest> _open = {};

  String get baseUrl => _base;
  set baseUrl(String url) {
    _base = _clean(url);
    abortAll();
  }

  static String _clean(String url) {
    var u = url.trim();
    if (u.isEmpty) u = 'http://127.0.0.1:8787';
    if (!u.contains('://')) u = 'http://$u';
    while (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    return u;
  }

  /// Cancel long polls in flight (e.g. when leaving a screen).
  void abortAll() {
    for (final r in _open.toList()) {
      r.abort(const BridgeException('cancelled'));
    }
    _open.clear();
  }

  Future<Map<String, dynamic>> _send(String method, String path,
      {Map<String, dynamic>? body, Duration timeout = const Duration(seconds: 8)}) async {
    HttpClientRequest? req;
    try {
      req = await _http.openUrl(method, Uri.parse('$_base$path')).timeout(timeout);
      _open.add(req);
      if (body != null) {
        final bytes = utf8.encode(jsonEncode(body));
        req.headers.contentType = ContentType.json;
        req.contentLength = bytes.length;
        req.add(bytes);
      }
      final res = await req.close().timeout(timeout);
      final text = await res.transform(utf8.decoder).join().timeout(timeout);
      if (res.statusCode != 200) {
        throw BridgeException('Bridge answered ${res.statusCode}');
      }
      final data = jsonDecode(text);
      return data is Map<String, dynamic> ? data : <String, dynamic>{};
    } on BridgeException {
      rethrow;
    } on TimeoutException {
      req?.abort();
      throw const BridgeException('The Termux bridge did not answer');
    } on SocketException {
      throw const BridgeException('Termux bridge is not running');
    } on HttpException catch (e) {
      throw BridgeException('Bridge connection failed: ${e.message}');
    } on FormatException {
      throw const BridgeException('Bridge sent something unexpected');
    } finally {
      if (req != null) _open.remove(req);
    }
  }

  /// Which agent sessions exist; throws if the bridge is not reachable.
  Future<Map<String, bool>> status() async {
    final m = await _send('GET', '/api/status', timeout: const Duration(seconds: 3));
    return {for (final e in m.entries) e.key: e.value == true};
  }

  Future<BridgeSnapshot> snapshot({int version = -1, int waitSec = 0, int tail = 60}) async {
    final m = await _send('GET', '/api/bridge?v=$version&wait=$waitSec&tail=$tail',
        timeout: Duration(seconds: waitSec + 8));
    return BridgeSnapshot.fromMap(m);
  }

  /// [system] becomes Claude's system prompt (an older bridge ignores it).
  Future<void> send(String text, {String to = 'claude', int turns = 0, String? system}) =>
      _send('POST', '/api/bridge/send', body: {
        'text': text,
        'to': to,
        'turns': turns,
        if (system != null) 'system': system,
      });

  Future<void> stop() => _send('POST', '/api/bridge/stop', body: const {});

  /// Upload diagnostics events (written to ~/.agent-hub/logs/ in Termux).
  Future<void> log(List<Map<String, Object?>> events, {String session = '', int dropped = 0}) =>
      _send('POST', '/api/log',
          body: {'app': 'even-suite', 'session': session, 'dropped': dropped, 'events': events},
          timeout: const Duration(seconds: 10));

  Future<void> reset() => _send('POST', '/api/bridge/reset', body: const {});

  /// The terminal screen of [agent]. With [waitSec] it waits until the screen
  /// differs from [hash] (long poll), so an idle terminal costs nothing.
  Future<TermScreen> capture(String agent,
      {int lines = 120, int waitSec = 0, String hash = ''}) async {
    final m = await _send(
        'GET', '/api/term/$agent/capture?lines=$lines&wait=$waitSec&h=${Uri.encodeQueryComponent(hash)}',
        timeout: Duration(seconds: waitSec + 8));
    return TermScreen('${m['text'] ?? ''}', '${m['h'] ?? ''}');
  }

  Future<void> paste(String agent, String text, {bool enter = true}) =>
      _send('POST', '/api/term/$agent/paste', body: {'text': text, 'enter': enter});

  Future<void> keys(String agent, List<String> keys) =>
      _send('POST', '/api/term/$agent/keys', body: {'keys': keys});

  void close() {
    abortAll();
    _http.close(force: true);
  }
}
