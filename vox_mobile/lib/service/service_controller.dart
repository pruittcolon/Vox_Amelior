import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:record/record.dart';
import 'package:vox_amelior_mobile/service/listening_runtime.dart';
import 'package:vox_amelior_mobile/service/listening_service.dart';

/// App-side control of the always-on listening service.
class ServiceController extends ChangeNotifier {
  bool _running = false;
  bool _paused = false;
  String? _lastError;
  int _heardToday = 0;
  final StreamController<Map<Object?, Object?>> _events = StreamController.broadcast();

  bool get isRunning => _running;
  bool get isPaused => _paused;
  String? get lastError => _lastError;
  int get heardToday => _heardToday;

  /// Raw events from the service (segments, questions, status).
  Stream<Map<Object?, Object?>> get events => _events.stream;

  void initialize() {
    FlutterForegroundTask.initCommunicationPort();
    FlutterForegroundTask.addTaskDataCallback(_onData);
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'vox_listening',
        channelName: 'Listening',
        channelDescription: 'Shown while Vox is listening',
        onlyAlertOnce: true,
      ),
      iosNotificationOptions: const IOSNotificationOptions(showNotification: false),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.repeat(30000),
        // Android does not allow starting the microphone from boot.
        autoRunOnBoot: false,
        allowWakeLock: true,
        allowWifiLock: true,
      ),
    );
    unawaited(refresh());
  }

  Future<void> refresh() async {
    _running = await FlutterForegroundTask.isRunningService;
    notifyListeners();
  }

  /// Starts listening. Returns an error message for the user, or null.
  Future<String?> start() async {
    final recorder = AudioRecorder();
    final micOk = await recorder.hasPermission();
    await recorder.dispose();
    if (!micOk) return 'Vox needs microphone permission to listen.';

    if (await FlutterForegroundTask.checkNotificationPermission() != NotificationPermission.granted) {
      await FlutterForegroundTask.requestNotificationPermission();
    }
    if (!await FlutterForegroundTask.isIgnoringBatteryOptimizations) {
      await FlutterForegroundTask.requestIgnoreBatteryOptimization();
    }

    final result = await FlutterForegroundTask.startService(
      serviceId: 417,
      serviceTypes: [ForegroundServiceTypes.microphone],
      notificationTitle: 'Vox is listening',
      notificationText: 'Starting speech models…',
      notificationButtons: [const NotificationButton(id: 'toggle', text: 'Pause')],
      callback: startListeningService,
    );
    _lastError = result is ServiceRequestFailure ? '${result.error}' : null;
    _paused = false;
    await refresh();
    return _lastError;
  }

  Future<void> stop() async {
    await FlutterForegroundTask.stopService();
    await refresh();
  }

  void pause() => _send({'cmd': ServiceCommands.pause});
  void resume() => _send({'cmd': ServiceCommands.resume});

  /// Tells the service to re-read people, rules and settings.
  void reload() => _send({'cmd': ServiceCommands.reload});

  void acknowledgeRequest(int id) => _send({'cmd': ServiceCommands.ackRequest, 'id': id});

  /// Frees the microphone for recording voice samples. Returns true if the
  /// service was listening (so the caller can [resume] afterwards).
  Future<bool> pauseForRecording() async {
    if (!_running || _paused) return false;
    pause();
    await Future<void>.delayed(const Duration(milliseconds: 600));
    return true;
  }

  void _send(Map<String, Object?> message) {
    if (_running) FlutterForegroundTask.sendDataToTask(message);
  }

  void _onData(Object data) {
    if (data is! Map) return;
    switch (data['type']) {
      case ServiceEvents.status:
        _paused = data['paused'] == true;
        _heardToday = (data['today'] as int?) ?? _heardToday;
        notifyListeners();
      case ServiceEvents.error:
        _lastError = '${data['message']}';
        notifyListeners();
    }
    _events.add(data);
  }

  @override
  void dispose() {
    FlutterForegroundTask.removeTaskDataCallback(_onData);
    unawaited(_events.close());
    super.dispose();
  }
}
