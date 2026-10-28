import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../glasses/glasses_display.dart';
import '../glasses/glasses_text.dart';
import '../log.dart';
import 'ansi.dart';
import 'bridge_client.dart';

/// The terminals in Termux (tmux sessions Agent Hub runs): the Claude Code
/// you use in Termux ("claude"), more Claude Codes opened from the app
/// ("claude-2", ...), shells.
///
/// [session] is the one you watch and talk to. Its screen (with colours) is
/// fetched with a long poll that only answers when it changed. Mirroring to
/// the glasses switches itself off after [mirrorIdle] without changes, so
/// it never runs forever.
class TerminalController extends ChangeNotifier {
  TerminalController(this.client, this.display);

  final BridgeClient client;
  final GlassesDisplay display;

  /// Older Agent Hubs only know these three.
  static const agents = {'claude': 'Claude Code', 'shell': 'Shell', 'agy': 'Antigravity'};

  String session = 'claude';

  /// The screen, colour codes included ([plain] without them).
  String text = '';
  int cols = 0;
  int rows = 0;
  String? error;
  bool online = false;
  bool mirroring = false;
  /// The glasses stop showing the terminal this long after it last changed
  /// (null: until you close it).
  Duration? mirrorIdle = const Duration(minutes: 10);

  /// Every terminal in Termux (refreshed while a screen shows them).
  List<TermSession> sessions = const [];

  /// The running Agent Hub lists sessions (false: an older one - `even restart`).
  bool sessionsSupported = true;

  /// tmux turns Agent Hub away, so no session can be seen or typed into
  /// until `~/agent-hub/allow-app` runs in Termux's Claude.
  bool denied = false;

  /// What the last line typed in did: sent, typed, menu, stuck (or an error).
  String lastSend = '';

  /// The session you picked is gone (closed elsewhere): pick another.
  void Function()? onSessionGone;

  String _hash = ''; // last screen seen
  String _serverHash = ''; // the bridge's hash of it (long poll key)
  bool _looping = false;
  final Set<Object> _owners = {};
  final Set<Object> _listOwners = {};
  Timer? _mirrorOff;
  Timer? _sessionsTimer;
  String _fitted = '';

  bool get _shouldRun => _owners.isNotEmpty || mirroring;

  String get plain => Ansi.strip(text);

  TermSession? get current {
    for (final s in sessions) {
      if (s.name == session) return s;
    }
    return null;
  }

  /// "Claude Code", "Claude 2", "Shell" ...
  String get label => current?.label ?? TermSession.labelFor(session);

  /// A Claude Code session (its hooks report steps and answers).
  bool get isClaude => current?.isClaude ?? session.startsWith('claude');

  /// Claude Code sessions, the Termux one first.
  List<TermSession> get claudes => [for (final s in sessions) if (s.isClaude) s];

  /// Keep [session]'s screen live while [owner] (a screen) shows it.
  void watch(Object owner) {
    _owners.add(owner);
    _loop();
    watchSessions(owner);
  }

  void unwatch(Object owner) {
    _owners.remove(owner);
    unwatchSessions(owner);
    if (!_shouldRun) client.abort('term');
  }

  /// Keep the list of sessions fresh (every 5 s) while [owner] shows it.
  void watchSessions(Object owner) {
    _listOwners.add(owner);
    _sessionsTimer ??= Timer.periodic(const Duration(seconds: 5), (_) => refreshSessions());
    unawaited(refreshSessions());
  }

  void unwatchSessions(Object owner) {
    _listOwners.remove(owner);
    if (_listOwners.isEmpty) {
      _sessionsTimer?.cancel();
      _sessionsTimer = null;
    }
  }

  /// Watch and talk to [name] from now on.
  void setSession(String name) {
    if (name == session || name.isEmpty) return;
    session = name;
    text = '';
    cols = rows = 0;
    _hash = '';
    _serverHash = '';
    client.abort('term'); // the loop picks up the new session
    if (mirroring) display.showMirror(const ['...'], title: label);
    notifyListeners();
  }

  Future<void> refreshSessions() async {
    try {
      final got = await client.sessions();
      sessions = got.list;
      denied = got.denied;
      sessionsSupported = true;
      online = true;
      if (sessions.isNotEmpty && current == null) onSessionGone?.call();
    } on BridgeException catch (e) {
      // An Agent Hub from before sessions: just the three fixed terminals.
      if (e.message.contains('404')) {
        sessionsSupported = false;
        sessions = [for (final a in agents.keys) TermSession(a, kind: a, main: a == 'claude')];
      }
    } catch (_) {}
    notifyListeners();
  }

  /// Open a Claude Code (or shell), fresh or picking up [resumeId]. Returns its name.
  Future<String> openSession({String kind = 'claude', String resumeId = ''}) async {
    final name = await client.newSession(kind: kind, resumeId: resumeId);
    EventLog.add('terminal', 'open', {'session': name, 'kind': kind, 'resume': resumeId});
    await refreshSessions();
    return name;
  }

  /// End [name]; returns why not, or null. Watching it? Back to the Termux one.
  Future<String?> closeSession(String name) async {
    final error = await client.closeSession(name);
    EventLog.add('terminal', 'close', {'session': name, 'error': error});
    await refreshSessions();
    return error;
  }

  /// Size the session to the phone's view (only one nobody shows in Termux).
  void fit(int cols, int rows) {
    final key = '$session:$cols:$rows';
    if (key == _fitted || current?.attached != 0 || !sessionsSupported) return;
    _fitted = key;
    unawaited(client.fitSession(session, cols, rows).catchError((Object _) {}));
  }

  /// Show this terminal on the glasses (or stop).
  void setMirror(bool on) {
    if (on == mirroring) return;
    mirroring = on;
    if (on) {
      _armMirrorOff();
      if (text.isNotEmpty) display.showMirror(glassesLines(text), title: label);
      _loop();
    } else {
      _mirrorOff?.cancel();
      if (display.mode == DisplayMode.mirror) display.clear();
      if (!_shouldRun) client.abort('term');
    }
    notifyListeners();
  }

  /// The glasses left our screen (double tap).
  void onGlassesExit() {
    if (!mirroring) return;
    mirroring = false;
    _mirrorOff?.cancel();
    notifyListeners();
  }

  /// Type a line into the terminal and press Enter. Returns what happened:
  /// "sent", "typed" (no Enter), "menu" (Claude Code is asking a question,
  /// nothing typed) or "stuck" (still in its input box).
  Future<String> sendLine(String line, {bool enter = true}) async {
    final sw = Stopwatch()..start();
    try {
      lastSend = await client.paste(session, line, enter: enter);
      error = null;
      online = true;
      return lastSend;
    } catch (e) {
      error = '$e';
      lastSend = 'error';
      rethrow;
    } finally {
      EventLog.add('terminal', 'send',
          {'session': session, 'chars': line.length, 'enter': enter, 'result': lastSend, 'ms': sw.elapsedMilliseconds});
      notifyListeners();
    }
  }

  /// Special keys: Enter, Escape, Up, Down, Tab, C-c, 1, 2, 3 ...
  Future<void> key(String k) async {
    try {
      await client.keys(session, [k]);
      error = null;
    } catch (e) {
      error = '$e';
      notifyListeners();
    }
  }

  void _armMirrorOff() {
    _mirrorOff?.cancel();
    final idle = mirrorIdle;
    if (idle != null) _mirrorOff = Timer(idle, () => setMirror(false));
  }

  Future<void> _loop() async {
    if (_looping) return;
    _looping = true;
    var backoff = 1;
    try {
      while (_shouldRun) {
        final want = session;
        final wasOnline = online;
        try {
          final screen = await client.capture(want, lines: 200, waitSec: 20, hash: _serverHash, ansi: true);
          backoff = 1;
          if (want != session) continue;
          online = true;
          error = null;
          // An older Agent Hub answers at once without a hash: poll once a second.
          final oldServer = screen.hash.isEmpty;
          final hash = oldServer ? '${screen.text.length}:${screen.text.hashCode}' : screen.hash;
          _serverHash = screen.hash;
          if (hash != _hash) {
            _hash = hash;
            text = screen.text;
            cols = screen.cols;
            rows = screen.rows;
            if (mirroring) {
              _armMirrorOff();
              display.showMirror(glassesLines(text), title: label);
            }
            notifyListeners();
          } else if (!wasOnline) {
            notifyListeners();
          }
          if (oldServer) await Future.delayed(const Duration(seconds: 1));
        } catch (e) {
          if (!_shouldRun) break;
          if ('$e' == 'cancelled') continue;
          online = false;
          error = '$e';
          notifyListeners();
          await Future.delayed(Duration(seconds: backoff));
          backoff = min(backoff * 2, 15);
        }
      }
    } finally {
      _looping = false;
    }
  }

  /// Terminal text -> wrapped glasses lines (last 40 non-empty lines).
  static List<String> glassesLines(String text) {
    final kept = Ansi.strip(text)
        .split('\n')
        .map(GlassesText.clean)
        .where((l) => l.trim().isNotEmpty && !RegExp(r'^[\s\->_=*]+$').hasMatch(l))
        .toList();
    final tail = kept.length > 40 ? kept.sublist(kept.length - 40) : kept;
    return GlassesText.wrap(tail.join('\n'));
  }

  @override
  void dispose() {
    _mirrorOff?.cancel();
    _sessionsTimer?.cancel();
    _owners.clear();
    _listOwners.clear();
    mirroring = false;
    super.dispose();
  }
}
