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

  /// Belongs in [agent]'s conversation ('claude' or 'agy'): what you sent it
  /// (or both, or a bubble not yet filed by the hub) and what it answered.
  bool belongsTo(String agent) {
    if (isMine) return to.isEmpty || to == 'both' || to == agent;
    if (who == 'claude' || who == 'agy') return who == agent;
    return true;
  }

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

  /// "job" (Claude finished, from the Stop hook) or "message" (a Claude Code
  /// session sent an update with `glasses "..."`).
  final String kind;

  bool get isMessage => kind == 'message';

  const WorkItem({
    required this.id,
    required this.ts,
    this.project = '',
    this.summary = '',
    this.text = '',
    this.ms = 0,
    this.tools = 0,
    this.prompt = '',
    this.kind = 'job',
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
        kind: '${m['kind'] ?? 'job'}',
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

  /// The last few steps, oldest first (the newest is [step]).
  final List<String> recent;

  const WorkJob({
    required this.session,
    this.project = '',
    this.prompt = '',
    this.step = '',
    this.steps = 0,
    this.started = 0,
    this.updated = 0,
    this.recent = const [],
  });

  factory WorkJob.fromMap(Map m) => WorkJob(
        session: '${m['session'] ?? ''}',
        project: '${m['project'] ?? ''}',
        prompt: '${m['prompt'] ?? ''}',
        step: '${m['step'] ?? ''}',
        steps: (m['steps'] as num?)?.toInt() ?? 0,
        started: (m['started'] as num?)?.toDouble() ?? 0,
        updated: (m['updated'] as num?)?.toDouble() ?? 0,
        recent: [for (final s in (m['recent'] as List? ?? const [])) '$s'],
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

  /// Sessions can be listed, opened and closed from the app.
  bool get sessions => features.contains('sessions');

  /// Antigravity answers live (stays running, streams its steps, gets the glasses rules).
  bool get agyLive => features.contains('agy-live');

  /// Ctrl, Alt and Shift with a key (C-x, M-b, S-Up ...) and Antigravity sessions.
  bool get modKeys => features.contains('modkeys');
}

class TermScreen {
  final String text;
  final String hash;

  /// The terminal's size (0 when an older Agent Hub does not say).
  final int cols;
  final int rows;
  const TermScreen(this.text, this.hash, {this.cols = 0, this.rows = 0});
}

/// A past Claude Code conversation (to pick up again in a new session).
class PastConversation {
  final String id;
  final String title; // the first thing you asked
  final String cwd; // the folder it ran in
  final double updated; // seconds since 1970
  final bool active; // written to in the last two minutes (open somewhere)
  const PastConversation(this.id, this.title, {this.cwd = '', this.updated = 0, this.active = false});

  factory PastConversation.fromMap(Map m) => PastConversation(
        '${m['id'] ?? ''}',
        '${m['title'] ?? ''}',
        cwd: '${m['cwd'] ?? ''}',
        updated: (m['updated'] as num?)?.toDouble() ?? 0,
        active: m['active'] == true,
      );

  DateTime get time => DateTime.fromMillisecondsSinceEpoch((updated * 1000).round());
}

/// A terminal in Termux (a tmux session): the Claude Code you use in Termux
/// ("claude"), more opened from the app ("claude-2", ...), shells.
class TermSession {
  final String name;
  final String kind; // claude, shell, agy
  final bool main; // the one Termux shows
  final int attached; // Termux screens showing it
  final String state; // Claude Code: working, asking (a question is up), idle
  final String topic; // what it was last asked
  final double activity; // last output, seconds since 1970
  const TermSession(this.name,
      {this.kind = 'claude', this.main = false, this.attached = 0, this.state = '', this.topic = '', this.activity = 0});

  factory TermSession.fromMap(Map m) => TermSession(
        '${m['name'] ?? ''}',
        kind: '${m['kind'] ?? 'shell'}',
        main: m['main'] == true,
        attached: (m['attached'] as num?)?.toInt() ?? 0,
        state: '${m['state'] ?? ''}',
        topic: '${m['topic'] ?? ''}',
        activity: (m['activity'] as num?)?.toDouble() ?? 0,
      );

  bool get isClaude => kind == 'claude';
  bool get isAgy => kind == 'agy';

  /// "Termux Claude" (the one Termux shows), "Claude 2", "Shell", "Shell 2", "Antigravity".
  String get label => labelFor(name, main: main);

  static String labelFor(String name, {bool? main}) {
    if (main ?? name == 'claude') return 'Termux Claude';
    final m = RegExp(r'^(claude|shell|agy)(?:-(\d+))?$').firstMatch(name);
    if (m == null) return name;
    final base = switch (m.group(1)) { 'claude' => 'Claude', 'agy' => 'Antigravity', _ => 'Shell' };
    return m.group(2) == null ? base : '$base ${m.group(2)}';
  }
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

  /// Requests in flight, by who started them ('chat', 'term', 'work'), so a
  /// screen closing cancels only its own long polls - never a message on its way.
  final Map<HttpClientRequest, String?> _open = {};

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

  /// Cancel everything in flight (the bridge address changed).
  void abortAll() {
    for (final r in _open.keys.toList()) {
      r.abort(const BridgeException('cancelled'));
    }
    _open.clear();
  }

  /// Cancel the long polls [tag] started (e.g. when leaving a screen).
  void abort(String tag) {
    for (final e in _open.entries.toList()) {
      if (e.value == tag) {
        e.key.abort(const BridgeException('cancelled'));
        _open.remove(e.key);
      }
    }
  }

  Future<Map<String, dynamic>> _send(String method, String path,
      {Map<String, dynamic>? body, Duration timeout = const Duration(seconds: 8), String? tag}) async {
    HttpClientRequest? req;
    try {
      req = await _http.openUrl(method, Uri.parse('$_base$path')).timeout(timeout);
      _open[req] = tag;
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
        timeout: Duration(seconds: waitSec + 8), tag: 'chat');
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

  /// Start Claude (or, with [to] 'agy', Antigravity) now - the user started
  /// talking - so the question does not wait for it to start.
  Future<void> warm({String? system, Map<String, String>? options, String to = 'claude'}) =>
      _send('POST', '/api/bridge/warm', body: {
        'to': to,
        if (system != null) 'system': system,
        if (options != null) 'options': options,
      });

  /// Finished Claude Code jobs; with [waitSec] it waits for a new one.
  Future<WorkSnapshot> work({int version = -1, int waitSec = 0}) async {
    final m = await _send('GET', '/api/work?v=$version&wait=$waitSec&tail=50',
        timeout: Duration(seconds: waitSec + 8), tag: 'work');
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

  /// The screen of terminal [session]. With [waitSec] it waits until the
  /// screen differs from [hash] (long poll), so an idle terminal costs
  /// nothing. [ansi] keeps the colours (escape codes).
  Future<TermScreen> capture(String session,
      {int lines = 120, int waitSec = 0, String hash = '', bool ansi = false}) async {
    final m = await _send(
        'GET',
        '/api/term/${Uri.encodeComponent(session)}/capture?lines=$lines&wait=$waitSec'
            '&h=${Uri.encodeQueryComponent(hash)}${ansi ? '&ansi=1' : ''}',
        timeout: Duration(seconds: waitSec + 8),
        tag: 'term');
    return TermScreen('${m['text'] ?? ''}', '${m['h'] ?? ''}',
        cols: (m['cols'] as num?)?.toInt() ?? 0, rows: (m['rows'] as num?)?.toInt() ?? 0);
  }

  /// Type [text] into [session] (and press Enter). Says how it went: "sent",
  /// "typed" (no Enter), "menu" (Claude Code is asking a question, nothing
  /// typed) or "stuck" (still in its input box). Older hubs: always "sent".
  Future<String> paste(String session, String text, {bool enter = true}) async {
    final m = await _send('POST', '/api/term/${Uri.encodeComponent(session)}/paste',
        body: {'text': text, 'enter': enter}, timeout: const Duration(seconds: 12));
    return '${m['result'] ?? (enter ? 'sent' : 'typed')}';
  }

  Future<void> keys(String session, List<String> keys) =>
      _send('POST', '/api/term/${Uri.encodeComponent(session)}/keys', body: {'keys': keys});

  /// Every terminal running in Termux (needs Agent Hub with sessions).
  /// [denied]: tmux turned Agent Hub away (allow-app has not been run).
  Future<({List<TermSession> list, bool denied})> sessions() async {
    final m = await _send('GET', '/api/sessions', timeout: const Duration(seconds: 6));
    return (
      list: [for (final x in (m['sessions'] as List? ?? const [])) TermSession.fromMap(x as Map)],
      denied: m['denied'] == true,
    );
  }

  /// Open a Claude Code ([kind] 'claude') or a shell; returns its name.
  /// [resumeId] picks up that past conversation instead of a fresh one.
  Future<String> newSession({String kind = 'claude', String resumeId = ''}) async {
    final m = await _send('POST', '/api/sessions',
        body: {'kind': kind, if (resumeId.isNotEmpty) 'resume_id': resumeId});
    return '${m['name'] ?? ''}';
  }

  /// Your recent Claude Code conversations, newest first.
  Future<List<PastConversation>> conversations() async {
    final m = await _send('GET', '/api/conversations?limit=20', timeout: const Duration(seconds: 8));
    return [for (final x in (m['conversations'] as List? ?? const [])) PastConversation.fromMap(x as Map)];
  }

  /// End a session; returns why not (the Termux one stays), or null.
  Future<String?> closeSession(String session) async {
    final m = await _send('POST', '/api/sessions/${Uri.encodeComponent(session)}/close', body: const {});
    return m['ok'] == true ? null : '${m['error'] ?? 'Could not close it'}';
  }

  /// Size [session] to the phone's view (skipped while Termux shows it).
  Future<void> fitSession(String session, int cols, int rows) => _send(
      'POST', '/api/term/${Uri.encodeComponent(session)}/size', body: {'cols': cols, 'rows': rows});

  void close() {
    abortAll();
    _http.close(force: true);
  }
}
