import 'dart:math';

import 'package:flutter/foundation.dart';

import '../log.dart';
import 'bridge_client.dart';

enum BridgeState { unknown, online, offline, starting }

/// Conversation with Claude through the bridge (`claude -p` with one ongoing
/// session). The bridge keeps the history, so the web page and this app see
/// the same conversation.
///
/// Efficient: one long poll that returns only when something changes, and
/// only while a screen shows the chat or a reply is still on its way.
class ChatController extends ChangeNotifier {
  ChatController(this.client);

  final BridgeClient client;

  BridgeState state = BridgeState.unknown;
  List<BridgeMessage> messages = const [];
  bool busy = false;
  String? error;

  /// Sent with every message as Claude's system prompt (glasses formatting).
  String? system;

  /// A finished reply to something sent from this app.
  void Function(BridgeMessage reply)? onReply;

  int _version = -1;
  bool _looping = false;
  final Set<Object> _owners = {};
  final Set<String> _pending = {};
  DateTime? _awaitingSince;
  bool _primed = false;

  bool get _awaiting =>
      _awaitingSince != null && DateTime.now().difference(_awaitingSince!) < const Duration(minutes: 15);

  BridgeMessage? get lastReply {
    for (final m in messages.reversed) {
      if (m.who == 'claude' && m.status == 'done') return m;
    }
    return null;
  }

  /// Is the bridge reachable? Updates [state].
  Future<bool> check() async {
    try {
      await client.status();
      state = BridgeState.online;
      error = null;
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

  Future<void> send(String text) async {
    final t = text.trim();
    if (t.isEmpty) return;
    _awaitingSince = DateTime.now();
    EventLog.add('chat', 'send', {'chars': t.length, 'system': system?.length ?? 0});
    // Show the bubble straight away; the next poll replaces it.
    messages = [
      ...messages,
      BridgeMessage(id: 'local-${DateTime.now().microsecondsSinceEpoch}', who: 'you', text: t, status: 'done'),
    ];
    busy = true;
    notifyListeners();
    try {
      await client.send(t, to: 'claude', system: system);
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
    busy = snap.busy.contains('claude');
    for (final m in snap.messages) {
      if (m.who != 'claude') continue;
      if (m.isPending) {
        _pending.add(m.id);
      } else if (_pending.remove(m.id) ||
          (!firstLook && _awaiting && m.time.isAfter(_awaitingSince!.subtract(const Duration(seconds: 5))) &&
              !messages.any((old) => old.id == m.id && !old.isPending))) {
        EventLog.add('chat', 'reply', {
          'chars': m.text.length,
          'ok': !m.isError,
          'ms': _awaitingSince == null ? null : DateTime.now().difference(_awaitingSince!).inMilliseconds,
        });
        if (!busy) _awaitingSince = null;
        onReply?.call(m);
      }
    }
    messages = snap.messages;
    notifyListeners();
  }
}
