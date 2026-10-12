import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../glasses/glasses_display.dart';
import '../glasses/glasses_text.dart';
import 'bridge_client.dart';

/// A live terminal in Termux (the tmux sessions Agent Hub runs): Claude Code
/// itself, a plain shell, or Antigravity.
///
/// The screen is fetched with a long poll that only answers when it changed.
/// Mirroring to the glasses switches itself off after [mirrorIdle] without
/// changes, so it never runs forever.
class TerminalController extends ChangeNotifier {
  TerminalController(this.client, this.display);

  final BridgeClient client;
  final GlassesDisplay display;

  static const agents = {'claude': 'Claude Code', 'shell': 'Shell', 'agy': 'Antigravity'};

  String agent = 'claude';
  String text = '';
  String? error;
  bool online = false;
  bool mirroring = false;
  Duration mirrorIdle = const Duration(minutes: 10);

  String _hash = ''; // last screen seen
  String _serverHash = ''; // the bridge's hash of it (long poll key)
  bool _looping = false;
  final Set<Object> _owners = {};
  Timer? _mirrorOff;

  bool get _shouldRun => _owners.isNotEmpty || mirroring;

  void watch(Object owner) {
    _owners.add(owner);
    _loop();
  }

  void unwatch(Object owner) {
    _owners.remove(owner);
    if (!_shouldRun) client.abortAll();
  }

  void setAgent(String a) {
    if (a == agent || !agents.containsKey(a)) return;
    agent = a;
    text = '';
    _hash = '';
    _serverHash = '';
    client.abortAll(); // the loop picks up the new session
    notifyListeners();
  }

  /// Show this terminal on the glasses (or stop).
  void setMirror(bool on) {
    if (on == mirroring) return;
    mirroring = on;
    if (on) {
      _armMirrorOff();
      if (text.isNotEmpty) display.showMirror(glassesLines(text), title: agents[agent]!);
      _loop();
    } else {
      _mirrorOff?.cancel();
      if (display.mode == DisplayMode.mirror) display.clear();
      if (!_shouldRun) client.abortAll();
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

  /// Type a line into the terminal and press Enter.
  Future<void> sendLine(String line, {bool enter = true}) async {
    try {
      await client.paste(agent, line, enter: enter);
      error = null;
      online = true;
    } catch (e) {
      error = '$e';
      online = false;
      rethrow;
    } finally {
      notifyListeners();
    }
  }

  /// Special keys: Enter, Escape, Up, Down, Tab, C-c, 1, 2, 3 ...
  Future<void> key(String k) async {
    try {
      await client.keys(agent, [k]);
      error = null;
    } catch (e) {
      error = '$e';
      notifyListeners();
    }
  }

  void _armMirrorOff() {
    _mirrorOff?.cancel();
    _mirrorOff = Timer(mirrorIdle, () => setMirror(false));
  }

  Future<void> _loop() async {
    if (_looping) return;
    _looping = true;
    var backoff = 1;
    try {
      while (_shouldRun) {
        final want = agent;
        final wasOnline = online;
        try {
          final screen = await client.capture(want, lines: 150, waitSec: 20, hash: _serverHash);
          backoff = 1;
          if (want != agent) continue;
          online = true;
          error = null;
          // An older Agent Hub answers at once without a hash: poll once a second.
          final oldServer = screen.hash.isEmpty;
          final hash = oldServer ? '${screen.text.length}:${screen.text.hashCode}' : screen.hash;
          _serverHash = screen.hash;
          if (hash != _hash) {
            _hash = hash;
            text = screen.text;
            if (mirroring) {
              _armMirrorOff();
              display.showMirror(glassesLines(text), title: agents[agent]!);
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
    final kept = text
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
    _owners.clear();
    mirroring = false;
    super.dispose();
  }
}
