
import 'package:flutter/foundation.dart';

import 'ble/proto.dart';
import 'log.dart';
import 'notify_model.dart';

/// A notification posted on the phone (from Android's notification access).
class PhoneNote {
  final String key;
  final String package;
  final String app;
  final String title;
  final String text;
  final DateTime time;
  const PhoneNote(this.key, this.package, this.app, this.title, this.text, this.time);

  factory PhoneNote.fromMap(Map m) => PhoneNote(
        '${m['key'] ?? ''}',
        '${m['package'] ?? ''}',
        '${m['app'] ?? m['package'] ?? ''}',
        '${m['title'] ?? ''}',
        '${m['text'] ?? ''}',
        DateTime.fromMillisecondsSinceEpoch((m['time'] as num?)?.toInt() ?? 0),
      );
}

/// Phone notifications on the glasses (the G1 shows them as they arrive and
/// keeps them for when you look up), plus the app's own short notices
/// ("Claude finished ..."). The glasses only show apps on their allow list,
/// so the list is sent whenever a new app needs it.
class PhoneNotifications extends ChangeNotifier {
  PhoneNotifications(this.proto);

  final Proto proto;

  static const selfId = 'com.vox.evensuite';

  bool forward = false;
  bool private = false;

  /// Allowed packages ('' set = all).
  Set<String> allowed = const {};

  /// Apps seen lately: package -> name (to pick from in the app).
  final Map<String, String> seen = {};
  final List<PhoneNote> recent = [];

  /// Each forwarded notification as one line, for the look-up notes.
  void Function(String line)? onForwarded;

  final Set<String> _listed = {};
  final Map<String, DateTime> _lastByText = {};
  int _msgId = 0;

  bool get _connected => proto.ble.isConnected;

  void onDisconnected() => _listed.clear();

  /// A notification arrived on the phone.
  Future<void> onPosted(PhoneNote n) async {
    if (n.package == selfId) return;
    seen[n.package] = n.app;
    recent.insert(0, n);
    if (recent.length > 30) recent.removeLast();
    notifyListeners();
    if (!forward || !_connected) return;
    if (allowed.isNotEmpty && !allowed.contains(n.package)) return;
    // Chat apps re-post the same message when anything changes: once is enough.
    final dedupe = '${n.package}|${n.title}|${n.text}';
    final last = _lastByText[dedupe];
    if (last != null && DateTime.now().difference(last) < const Duration(minutes: 2)) return;
    _lastByText[dedupe] = DateTime.now();
    if (_lastByText.length > 200) _lastByText.remove(_lastByText.keys.first);
    final ok = await send(n.package, n.app, n.title, private ? '' : n.text);
    EventLog.add('notify', 'forward', {'app': n.package, 'ok': ok});
    if (ok) onForwarded?.call('${n.app}: ${n.title}'.trim());
  }

  /// Show a notification on the glasses as [app] ([appName] is what the glasses print).
  Future<bool> send(String app, String appName, String title, String text) async {
    if (!_connected) return false;
    if (!_listed.contains(app)) {
      _listed.add(app);
      final apps = {selfId: 'Claude', ...seen, app: appName};
      final list = NotifyWhitelistModel([for (final e in apps.entries) NotifyAppModel(e.key, e.value)]);
      if (!await proto.sendAppWhitelist(list.toJson())) {
        _listed.remove(app);
        return false;
      }
      _listed.addAll(apps.keys);
    }
    final now = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    final model = NotifyModel(now, app, _short(title, 60), '', _short(text, 180), now, appName);
    _msgId = (_msgId + 1) % 256;
    return proto.sendNotify(model.toMap(), _msgId, retry: 3);
  }

  static String _short(String s, int n) {
    final t = s.replaceAll(RegExp(r'\s+'), ' ').trim();
    return t.length <= n ? t : '${t.substring(0, n - 3)}...';
  }
}
