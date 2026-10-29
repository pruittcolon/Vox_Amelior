import 'dart:math';

import 'package:flutter/foundation.dart';

import '../log.dart';
import 'bridge_client.dart';

enum BridgeState { unknown, online, offline, starting }

/// Conversation with Claude through the bridge (one Claude Code session that
/// Agent Hub keeps running). The bridge keeps the history, so the web page and
/// this app see the same conversation.
///
/// Efficient: one long poll that returns only when something changes, and
/// only while a screen shows the chat or a reply is still on its way. While
/// Claude answers, [onProgress] reports each step (thinking, a tool, the text
/// so far) and [onReply] the finished answer.
class ChatController extends ChangeNotifier {
  ChatController(this.client);

  final BridgeClient client;

  BridgeState state = BridgeState.unknown;
  List<BridgeMessage> messages = const [];
  bool busy = false;
  String? error;

  /// What the running Agent Hub supports (null until checked).
  HubInfo? hub;

  /// Who answers: 'claude', or 'agy' (Google Antigravity - for when Claude has run out).
  String get agent => _agent;
  String _agent = 'claude';
  set agent(String value) {
    final next = value == 'agy' ? 'agy' : 'claude';
    if (next == _agent) return;
    _agent = next;
    // An answer on its way from the other one is no longer ours to wait for.
    _awaitingSince = null;
    active = null;
    _activeKey = '';
    _pending.clear();
    busy = false;
    notifyListeners();
  }

  String get agentName => agentLabel(_agent);

  static String agentLabel(String agent) => agent == 'agy' ? 'Antigravity' : 'Claude';

  /// Sent with every message as the system prompt (glasses formatting).
  String? system;

  /// Model, effort and access, sent with every message.
  Map<String, String>? options;

  /// A finished reply to something sent from this app.
  void Function(BridgeMessage reply)? onReply;

  /// The reply on its way changed (stage, tool step, or more text).
  void Function(BridgeMessage reply)? onProgress;

  /// The reply currently being worked on, if any.
  BridgeMessage? active;

  int _version = -1;
  bool _looping = false;
  final Set<Object> _owners = {};
  final Set<String> _pending = {};
  DateTime? _awaitingSince;
  bool _primed = false;
  String _activeKey = '';

  bool get _awaiting =>
      _awaitingSince != null && DateTime.now().difference(_awaitingSince!) < const Duration(minutes: 30);

  /// A question is out and its answer has not arrived yet.
  bool get waiting => _awaitingSince != null;

  BridgeMessage? get lastReply {
    for (final m in messages.reversed) {
      if (m.who == _agent && m.status == 'done') return m;
    }
    return null;
  }

  /// Is the bridge reachable? Updates [state] and [hub].
  Future<bool> check() async {
    try {
      await client.status();
      state = BridgeState.online;
      error = null;
      try {
        hub = await client.ping();
      } catch (_) {
        hub = const HubInfo(1, {});
      }
      notifyListeners();
      return true;
    } catch (e) {
      if (state != BridgeState.starting) state = BridgeState.offline;
      error = '$e';
      notifyListeners();
      return false;
    }
  }

  void markStarting() {
    state = BridgeState.starting;
    error = null;
    notifyListeners();
  }

  /// Keep the conversation live while [owner] (a screen) is visible.
  void watch(Object owner) {
    _owners.add(owner);
    _loop();
  }

  void unwatch(Object owner) {
    _owners.remove(owner);
    if (!_shouldRun) client.abortAll();
  }

  bool get _shouldRun => _owners.isNotEmpty || busy || _awaiting;

  /// Start Claude early (the user started talking). Errors are ignored.
  Future<void> warm() async {
    try {
      await client.warm(system: system, options: options, to: _agent);
    } catch (_) {}
  }

  Future<void> send(String text) async {
    final t = text.trim();
    if (t.isEmpty) return;
    _awaitingSince = DateTime.now();
    active = null;
    _activeKey = '';
    EventLog.add('chat', 'send', {'chars': t.length, 'system': system?.length ?? 0, 'agent': _agent, ...?options});
    // Show the bubble straight away; the next poll replaces it.
    messages = [
      ...messages,
      BridgeMessage(id: 'local-${DateTime.now().microsecondsSinceEpoch}', who: 'you', text: t, status: 'done'),
    ];
    busy = true;
    notifyListeners();
    try {
      await client.send(t, to: _agent, system: system, options: options);
      state = BridgeState.online;
      error = null;
    } catch (e) {
      error = '$e';
      state = BridgeState.offline;
      busy = false;
      _awaitingSince = null;
      messages = messages.where((m) => !m.id.startsWith('local-')).toList();
      notifyListeners();
      rethrow;
    }
    _loop();
  }

  /// Give up waiting for the answer (it may still arrive in the chat).
  void forget() {
    _awaitingSince = null;
    active = null;
    _activeKey = '';
    notifyListeners();
  }

  Future<void> stopReply() async {
    try {
      await client.stop();
    } catch (e) {
      error = '$e';
      notifyListeners();
    }
  }

  Future<void> newChat() async {
    await client.reset();
    _pending.clear();
    _awaitingSince = null;
    active = null;
    messages = const [];
    notifyListeners();
  }

  Future<void> _loop() async {
    if (_looping) return;
    _looping = true;
    var backoff = 1;
    try {
      while (_shouldRun) {
        try {
          final snap = await client.snapshot(version: _version, waitSec: 25);
          backoff = 1;
          apply(snap);
        } catch (e) {
          if (!_shouldRun) break;
          if ('$e' != 'cancelled') {
            state = BridgeState.offline;
            error = '$e';
            notifyListeners();
          }
          await Future.delayed(Duration(seconds: backoff));
          backoff = min(backoff * 2, 15);
        }
      }
    } finally {
      _looping = false;
    }
  }

  /// Take in a snapshot from the bridge (public for tests).
  @visibleForTesting
  void apply(BridgeSnapshot snap) {
    final firstLook = !_primed;
    _primed = true;
    _version = snap.version;
    state = BridgeState.online;
    error = null;
    busy = snap.busy.contains(_agent) || snap.messages.any((m) => m.who == _agent && m.isQueued);
    BridgeMessage? progress;
    for (final m in snap.messages) {
      if (m.who != _agent) continue;
      if (m.isPending) {
        _pending.add(m.id);
        if (_awaiting && !firstLook) progress = m;
      } else if (_pending.remove(m.id) ||
          (!firstLook && _awaiting && m.time.isAfter(_awaitingSince!.subtract(const Duration(seconds: 5))) &&
              !messages.any((old) => old.id == m.id && !old.isPending))) {
        EventLog.add('chat', 'reply', {
          'chars': m.text.length,
          'ok': !m.isError,
          'ms': _awaitingSince == null ? null : DateTime.now().difference(_awaitingSince!).inMilliseconds,
          'claudeMs': m.ms,
          'steps': m.steps.length,
        });
        if (!busy) _awaitingSince = null;
        if (active?.id == m.id) {
          active = null;
          _activeKey = '';
        }
        onReply?.call(m);
      }
    }
    messages = snap.messages;
    if (progress != null) {
      final key = '${progress.id}|${progress.status}|${progress.stage}|${progress.tool}|${progress.block}|'
          '${progress.live.length}';
      active = progress;
      if (key != _activeKey) {
        _activeKey = key;
        onProgress?.call(progress);
      }
    }
    notifyListeners();
  }
}
