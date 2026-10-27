import 'dart:math';

import 'package:flutter/foundation.dart';

import '../log.dart';
import 'bridge_client.dart';

/// Finished Claude Code jobs (Work tab). Claude Code's Stop hook tells Agent
/// Hub when Claude finishes; this keeps one cheap long poll open and calls
/// [onDone] for each new job that took at least [minDuration], so the app can
/// say so on the phone and the glasses - only when Claude is done.
class WorkController extends ChangeNotifier {
  WorkController(this.client);

  final BridgeClient client;

  List<WorkItem> items = const [];
  bool online = false;
  String? error;

  /// Jobs not looked at yet (badge on the Work tab).
  int unread = 0;

  Duration minDuration = const Duration(seconds: 30);

  void Function(WorkItem item)? onDone;

  int _version = -1;
  bool _running = false;
  bool _primed = false;

  /// Keep watching for finished jobs (runs until [stop]).
  void start() {
    if (_running) return;
    _running = true;
    _loop();
  }

  void stop() => _running = false;

  void markRead() {
    if (unread == 0) return;
    unread = 0;
    notifyListeners();
  }

  Future<void> clear() async {
    await client.clearWork();
    items = const [];
    unread = 0;
    notifyListeners();
  }

  Future<void> _loop() async {
    var backoff = 2;
    while (_running) {
      try {
        final snap = await client.work(version: _version, waitSec: 25);
        backoff = 2;
        apply(snap);
      } catch (e) {
        if (!_running) break;
        if ('$e' == 'cancelled') continue;
        if (online || error == null) {
          online = false;
          error = '$e';
          notifyListeners();
        }
        await Future.delayed(Duration(seconds: backoff));
        backoff = min(backoff * 2, 30);
      }
    }
  }

  /// Take in a snapshot (public for tests).
  @visibleForTesting
  void apply(WorkSnapshot snap) {
    final firstLook = !_primed;
    _primed = true;
    final known = {for (final i in items) i.id};
    _version = snap.version;
    online = true;
    error = null;
    final fresh = firstLook ? const <WorkItem>[] : snap.items.where((i) => !known.contains(i.id)).toList();
    items = snap.items;
    for (final item in fresh) {
      if (item.took < minDuration) continue;
      unread++;
      EventLog.add('work', 'done', {'project': item.project, 'ms': item.ms, 'tools': item.tools});
      onDone?.call(item);
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _running = false;
    super.dispose();
  }
}

/// "1 min 20 s", "45 s", "2 h 5 min".
String formatDuration(Duration d) {
  if (d.inHours > 0) return '${d.inHours} h ${d.inMinutes % 60} min';
  if (d.inMinutes > 0) return d.inSeconds % 60 == 0 ? '${d.inMinutes} min' : '${d.inMinutes} min ${d.inSeconds % 60} s';
  return '${d.inSeconds} s';
}
