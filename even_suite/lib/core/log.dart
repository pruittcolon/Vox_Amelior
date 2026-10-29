import 'dart:async';

import 'bridge/bridge_client.dart';

/// One diagnostics record.
class LogEvent {
  final DateTime time;
  final String cat; // ble, display, bmp, screen, voice, chat, glasses, app, native
  final String name;
  final Map<String, Object?> data;
  const LogEvent(this.time, this.cat, this.name, this.data);

  Map<String, Object?> toJson() => {'t': time.millisecondsSinceEpoch, 'c': cat, 'e': name, ...data};

  String get summary => data.entries.map((e) => '${e.key}=${e.value}').join(' ');
}

/// Raw timing data for tuning: every send to the glasses, every reply, every
/// recognizer event, plus native counters. Kept in memory (last 500 for the
/// Diagnostics screen) and uploaded in batches to Agent Hub, which writes
/// them to `~/.agent-hub/logs/even-suite-<date>.jsonl` in Termux.
///
/// Cheap by design: adding an event is a list append; one HTTP request every
/// [flushEvery] at most, and only while there is something to send.
class EventLog {
  EventLog._();
  static final EventLog instance = EventLog._();

  static void add(String cat, String name, [Map<String, Object?> data = const {}]) =>
      instance._add(cat, name, data);

  bool enabled = true;
  Duration flushEvery = const Duration(seconds: 5);
  static const int keepRecent = 500;
  static const int maxPending = 5000;

  final List<LogEvent> recent = [];
  final List<Map<String, Object?>> _pending = [];
  int dropped = 0;
  int uploaded = 0;
  String? lastError;
  DateTime? lastUpload;
  final String session = DateTime.now().toIso8601String();

  /// Where batches go (set when the app starts; tests leave it null).
  BridgeClient? client;

  /// Native counters (Bluetooth writes, mic, recognizer, screen capture),
  /// read and reset at each upload.
  Future<Map<String, Object?>> Function()? nativeStats;

  Timer? _timer;
  bool _flushing = false;

  int get pending => _pending.length;

  void _add(String cat, String name, Map<String, Object?> data) {
    if (!enabled) return;
    final e = LogEvent(DateTime.now(), cat, name, data);
    recent.add(e);
    if (recent.length > keepRecent) recent.removeRange(0, recent.length - keepRecent);
    _pending.add(e.toJson());
    if (_pending.length > maxPending) {
      final n = _pending.length - maxPending;
      _pending.removeRange(0, n);
      dropped += n;
    }
    if (client != null) _timer ??= Timer(flushEvery, flush);
  }

  /// Upload what is pending now.
  Future<bool> flush() async {
    _timer = null;
    final c = client;
    if (c == null || _flushing) return false;
    _flushing = true;
    try {
      final stats = nativeStats;
      if (stats != null) {
        try {
          final m = await stats();
          if (m.isNotEmpty) _pending.add(LogEvent(DateTime.now(), 'native', 'stats', m).toJson());
        } catch (_) {}
      }
      while (_pending.isNotEmpty) {
        final n = _pending.length > 1000 ? 1000 : _pending.length;
        final batch = _pending.sublist(0, n);
        await c.log(batch, session: session, dropped: dropped);
        _pending.removeRange(0, n);
        uploaded += n;
      }
      lastUpload = DateTime.now();
      lastError = null;
      return true;
    } catch (e) {
      lastError = '$e';
      return false;
    } finally {
      _flushing = false;
      if (_pending.isNotEmpty && client != null) _timer ??= Timer(flushEvery * 3, flush);
    }
  }

  void clear() {
    recent.clear();
    _pending.clear();
    dropped = 0;
  }

  /// Stop timers (tests, app exit).
  void stop() {
    _timer?.cancel();
    _timer = null;
  }
}
