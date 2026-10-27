import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// A chat bubble from the Termux bridge (Agent Hub).
///
/// While Claude answers, a newer Agent Hub streams its progress into the
/// bubble: [stage] (queued, starting, thinking, tool, writing, done, error),
/// the current [tool] step, every step so far, and [live] - the text being
/// written right now ([block] counts text blocks, so a new one can be told
/// apart from a longer old one).
class BridgeMessage {
  final String id;
  final String who; // you, claude, agy
  final String to;
  final String text;
  final String status; // queued, pending, done, error
  final double ts;
  final String stage;
  final String tool;
  final String live;
  final int block;
  final List<String> steps;
  final double started;
  final int ms;
  final String note;

  const BridgeMessage({
    required this.id,
    required this.who,
    this.to = '',
    required this.text,
    required this.status,
    this.ts = 0,
    this.stage = '',
    this.tool = '',
    this.live = '',
    this.block = 0,
    this.steps = const [],
    this.started = 0,
    this.ms = 0,
    this.note = '',
  });

  factory BridgeMessage.fromMap(Map m) => BridgeMessage(
        id: '${m['id'] ?? ''}',
        who: '${m['who'] ?? ''}',
        to: '${m['to'] ?? ''}',
        text: '${m['text'] ?? ''}',
        status: '${m['status'] ?? 'done'}',
        ts: (m['ts'] as num?)?.toDouble() ?? 0,
        stage: '${m['stage'] ?? ''}',
        tool: '${m['tool'] ?? ''}',
        live: '${m['live'] ?? ''}',
        block: (m['block'] as num?)?.toInt() ?? 0,
        steps: [for (final s in (m['steps'] as List? ?? const [])) '$s'],
        started: (m['started'] as num?)?.toDouble() ?? 0,
        ms: (m['ms'] as num?)?.toInt() ?? 0,
        note: '${m['note'] ?? ''}',
      );

  bool get isMine => who == 'you';

  /// Still on its way (waiting its turn, or being written).
  bool get isPending => status == 'pending' || status == 'queued';
  bool get isQueued => status == 'queued';
  bool get isError => status == 'error';
  bool get isDone => status == 'done';

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

/// A finished Claude Code job, reported by the Stop hook (Work tab).
class WorkItem {
  final String id;
  final double ts;
  final String project;
  final String summary;
  final String text;
  final int ms;
  final int tools;

  /// What you asked (when the start hook saw it).
  final String prompt;

  const WorkItem({
    required this.id,
    required this.ts,
    this.project = '',
    this.summary = '',
    this.text = '',
    this.ms = 0,
    this.tools = 0,
    this.prompt = '',
  });

  factory WorkItem.fromMap(Map m) => WorkItem(
        id: '${m['id'] ?? ''}',
        ts: (m['ts'] as num?)?.toDouble() ?? 0,
        project: '${m['project'] ?? ''}',
        summary: '${m['summary'] ?? ''}',
        text: '${m['text'] ?? ''}',
        ms: (m['ms'] as num?)?.toInt() ?? 0,
        tools: (m['tools'] as num?)?.toInt() ?? 0,
        prompt: '${m['prompt'] ?? ''}',
      );

  DateTime get time => DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
  Duration get took => Duration(milliseconds: ms);
}

/// A Claude Code job still going (from the start and step hooks).
class WorkJob {
  final String session;
  final String project;
  final String prompt;
  final String step;
  final int steps;
  final double started;
  final double updated;

  const WorkJob({
    required this.session,
    this.project = '',
    this.prompt = '',
    this.step = '',
    this.steps = 0,
    this.started = 0,
    this.updated = 0,
  });

  factory WorkJob.fromMap(Map m) => WorkJob(
        session: '${m['session'] ?? ''}',
        project: '${m['project'] ?? ''}',
        prompt: '${m['prompt'] ?? ''}',
        step: '${m['step'] ?? ''}',
        steps: (m['steps'] as num?)?.toInt() ?? 0,
        started: (m['started'] as num?)?.toDouble() ?? 0,
        updated: (m['updated'] as num?)?.toDouble() ?? 0,
      );

  DateTime get startTime => DateTime.fromMillisecondsSinceEpoch((started * 1000).round());

  /// How long it has been going.
  Duration elapsed([DateTime? now]) => (now ?? DateTime.now()).difference(startTime);

  String get key => '$session|$step|$steps|$prompt';
}

class WorkSnapshot {
  final int version;
  final List<WorkItem> items;

  /// Jobs in progress, most recently active first.
  final List<WorkJob> active;
  const WorkSnapshot(this.version, this.items, [this.active = const []]);

  factory WorkSnapshot.fromMap(Map m) => WorkSnapshot(
        (m['version'] as num?)?.toInt() ?? 0,
        [for (final x in (m['items'] as List? ?? const [])) WorkItem.fromMap(x as Map)],
        [for (final x in (m['active'] as List? ?? const [])) WorkJob.fromMap(x as Map)],
      );
}

/// What a running Agent Hub can do (older ones lack live answers and the Work feed).
class HubInfo {
  final int api;
  final Set<String> features;

  /// The "Claude is done" hook is set up in Claude Code.
  final bool hook;
  const HubInfo(this.api, this.features, {this.hook = false});
  bool get live => features.contains('live');
  bool get work => features.contains('work');
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

  Future<BridgeSnapshot> snapshot({int version = -1, int waitSec = 0, int tail = 40}) async {
    final m = await _send('GET', '/api/bridge?v=$version&wait=$waitSec&tail=$tail',
        timeout: Duration(seconds: waitSec + 8));
    return BridgeSnapshot.fromMap(m);
  }

  /// Version and features of the running Agent Hub.
  Future<HubInfo> ping() async {
    final m = await _send('GET', '/api/ping', timeout: const Duration(seconds: 3));
    return HubInfo((m['api'] as num?)?.toInt() ?? 1, {for (final f in (m['features'] as List? ?? const [])) '$f'},
        hook: m['hook'] == true);
  }

  /// Set up Claude Code's "Claude is done" hook (Work tab). True when it is in place.
  Future<bool> installHook() async {
    final m = await _send('POST', '/api/work/install', body: const {}, timeout: const Duration(seconds: 20));
    return m['hook'] == true;
  }

  /// [system] becomes Claude's system prompt; [options] picks the model,
  /// effort and access (an older bridge ignores both).
  Future<void> send(String text,
          {String to = 'claude', int turns = 0, String? system, Map<String, String>? options}) =>
      _send('POST', '/api/bridge/send', body: {
        'text': text,
        'to': to,
        'turns': turns,
        if (system != null) 'system': system,
        if (options != null) 'options': options,
      });

  /// Start Claude now (the user started talking), so the question does not
  /// wait for Claude Code to start.
  Future<void> warm({String? system, Map<String, String>? options}) => _send('POST', '/api/bridge/warm', body: {
        if (system != null) 'system': system,
        if (options != null) 'options': options,
      });

  /// Finished Claude Code jobs; with [waitSec] it waits for a new one.
  Future<WorkSnapshot> work({int version = -1, int waitSec = 0}) async {
    final m = await _send('GET', '/api/work?v=$version&wait=$waitSec&tail=50',
        timeout: Duration(seconds: waitSec + 8));
    return WorkSnapshot.fromMap(m);
  }

  Future<void> clearWork() => _send('POST', '/api/work/clear', body: const {});

  Future<void> stop() => _send('POST', '/api/bridge/stop', body: const {});

  /// Turn Agent Hub off (`even stop`).
  Future<void> quit() => _send('POST', '/api/quit', body: const {});

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
