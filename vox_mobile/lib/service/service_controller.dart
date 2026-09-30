import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:record/record.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/location/location_monitor.dart';
import 'package:vox_amelior_mobile/service/listening_service.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';

/// App-side control of the always-on service, and its live status.
class ServiceController extends ChangeNotifier {
  static const int _serviceId = 417;

  bool _running = false;
  String? _mode;
  ListenState _state = ListenState.starting;
  String _reason = '';
  String _activity = 'idle';
  int _backlog = 0;
  int _heardToday = 0;
  bool _assistantReady = false;
  String? _lastError;
  bool _fatal = false;
  bool _busy = false;
  bool _batteryOk = true;
  final StreamController<Map<Object?, Object?>> _events = StreamController.broadcast();

  /// The service runs to listen (not just to keep a download alive).
  bool get isListening => _running && _mode == kModeListen;
  ListenState get state => isListening ? (_fatal ? ListenState.error : _state) : ListenState.paused;
  String get reason => _reason;
  bool get isThinking => _activity == 'assistant';
  bool get isReviewing => _activity == 'reviewing' || _reason.contains('reviewing');
  int get backlog => _backlog;
  int get heardToday => _heardToday;
  bool get assistantReadyInService => _assistantReady;
  String? get lastError => _lastError;

  /// Start/stop is in progress (permission prompts etc.).
  bool get isBusy => _busy;

  /// Android lets Vox run without battery restrictions.
  bool get batteryOk => _batteryOk;

  /// Raw events from the service (segments, answers, status).
  Stream<Map<Object?, Object?>> get events => _events.stream;

  void initialize() {
    FlutterForegroundTask.initCommunicationPort();
    FlutterForegroundTask.addTaskDataCallback(_onData);
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'vox_listening',
        channelName: 'Listening',
        channelDescription: 'Shown while Vox is on',
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
    try {
      _running = await FlutterForegroundTask.isRunningService;
      _mode = _running ? await FlutterForegroundTask.getData<String>(key: kServiceModeKey) : null;
      _batteryOk = await FlutterForegroundTask.isIgnoringBatteryOptimizations;
    } on Object catch (e) {
      Log.w('service', 'status refresh failed', e);
    }
    notifyListeners();
  }

  /// Starts listening. Returns a message for the user, or null on success.
  Future<String?> start({bool useLocation = false}) async {
    if (_busy) return null;
    _busy = true;
    _lastError = null;
    _fatal = false;
    notifyListeners();
    try {
      if (!await _micPermission()) return _fail('Vox needs microphone permission to listen. Allow it in Android settings → Apps → Vox → Permissions.');
      await _notificationPermission();
      final types = [ForegroundServiceTypes.microphone];
      if (useLocation && (await LocationMonitor.access() == LocationAccess.granted || await LocationMonitor.requestPermission())) {
        types.add(ForegroundServiceTypes.location);
      }
      await FlutterForegroundTask.saveData(key: kServiceModeKey, value: kModeListen);
      if (await FlutterForegroundTask.isRunningService) {
        await FlutterForegroundTask.stopService();
        await Future<void>.delayed(const Duration(milliseconds: 400));
      }
      final result = await FlutterForegroundTask.startService(
        serviceId: _serviceId,
        serviceTypes: types,
        notificationTitle: 'Vox is starting',
        notificationText: 'Loading speech models…',
        notificationButtons: [const NotificationButton(id: 'toggle', text: 'Pause')],
        callback: startListeningService,
      );
      if (result is ServiceRequestFailure) return _fail('Android refused to start listening: ${result.error}');
      _state = ListenState.starting;
      _reason = 'Loading speech models…';
      return null;
    } on Object catch (e) {
      return _fail('Could not start listening: $e');
    } finally {
      _busy = false;
      await refresh();
    }
  }

  Future<void> stop() async {
    await FlutterForegroundTask.stopService();
    await FlutterForegroundTask.removeData(key: kServiceModeKey);
    await refresh();
  }

  final Set<String> _keepAlive = {};

  /// Keeps the app process alive while large models download in the background.
  Future<void> keepAliveForDownloads(bool downloading) => keepAliveFor('download', downloading);

  /// Keeps the app alive in the background while it works on its own
  /// (downloads, or reviews while Vox is not listening).
  Future<void> keepAliveFor(String reason, bool on) async {
    if (on) {
      _keepAlive.add(reason);
    } else {
      _keepAlive.remove(reason);
    }
    try {
      final running = await FlutterForegroundTask.isRunningService;
      final mode = running ? await FlutterForegroundTask.getData<String>(key: kServiceModeKey) : null;
      if (_keepAlive.isNotEmpty && !running) {
        await FlutterForegroundTask.saveData(key: kServiceModeKey, value: kModeDownload);
        await FlutterForegroundTask.startService(
          serviceId: _serviceId,
          serviceTypes: [ForegroundServiceTypes.dataSync],
          notificationTitle: _keepAlive.contains('download') ? 'Downloading models' : 'Vox is reviewing',
          notificationText: 'Vox keeps working while you use other apps',
          callback: startListeningService,
        );
      } else if (_keepAlive.isEmpty && running && mode == kModeDownload) {
        await FlutterForegroundTask.stopService();
      }
    } on Object catch (e) {
      Log.w('service', 'keep-alive failed', e);
    }
    await refresh();
  }

  Future<void> updateDownloadNotification(String text) => updateWorkNotification('Downloading models', text);

  Future<void> updateWorkNotification(String title, String text) async {
    if (_running && _mode == kModeDownload) {
      await FlutterForegroundTask.updateService(notificationTitle: title, notificationText: text);
    }
  }

  /// Opens Android's "allow background activity" prompt.
  Future<void> requestBatteryExemption() async {
    try {
      await FlutterForegroundTask.requestIgnoreBatteryOptimization().timeout(const Duration(minutes: 2));
    } on Object catch (e) {
      Log.w('service', 'battery exemption request failed', e);
    }
    await refresh();
  }

  void pause() => _send({'cmd': ServiceCommands.pause});
  void resume() => _send({'cmd': ServiceCommands.resume});
  void retry() => _send({'cmd': ServiceCommands.retryStart});

  /// Tells the service to re-read people, rules, settings and models.
  void reload() => _send({'cmd': ServiceCommands.reload});

  /// Asks the service to answer stored assistant request [id].
  void ask(int id) => _send({'cmd': ServiceCommands.ask, 'id': id});

  /// Stops answering request [id].
  void cancelAsk(int id) => _send({'cmd': ServiceCommands.cancelAsk, 'id': id});

  /// Runs the phone context test in the service.
  void probe() => _send({'cmd': ServiceCommands.probe});

  /// Turns the live microphone level ([ServiceEvents.level]) on or off.
  void levelMeter(bool on) => _send({'cmd': ServiceCommands.levelMeter, 'on': on});

  /// New or resumed reviews are waiting.
  void reviewKick() => _send({'cmd': ServiceCommands.reviewKick});

  /// Frees the microphone for recording voice samples. Returns true if the
  /// service was listening (so the caller can [resume] afterwards).
  Future<bool> pauseForRecording() async {
    if (!isListening || _state == ListenState.paused) return false;
    pause();
    await Future<void>.delayed(const Duration(milliseconds: 700));
    return true;
  }

  Future<bool> _micPermission() async {
    final recorder = AudioRecorder();
    try {
      return await recorder.hasPermission().timeout(const Duration(minutes: 2));
    } on Object {
      return false;
    } finally {
      unawaited(recorder.dispose());
    }
  }

  Future<void> _notificationPermission() async {
    try {
      if (await FlutterForegroundTask.checkNotificationPermission() != NotificationPermission.granted) {
        await FlutterForegroundTask.requestNotificationPermission().timeout(const Duration(minutes: 1));
      }
    } on Object catch (e) {
      // Not fatal: the service still runs, the notification is just hidden.
      Log.w('service', 'notification permission request failed', e);
    }
  }

  String _fail(String message) {
    _lastError = message;
    _fatal = true;
    return message;
  }

  void _send(Map<String, Object?> message) {
    if (_running) FlutterForegroundTask.sendDataToTask(message);
  }

  void _onData(Object data) {
    if (data is! Map) return;
    switch (data['type']) {
      case ServiceEvents.status:
        _state = ListenState.values.asNameMap()[data['state']] ?? _state;
        _reason = '${data['reason'] ?? ''}';
        _activity = '${data['activity'] ?? 'idle'}';
        _backlog = (data['backlog'] as int?) ?? 0;
        _heardToday = (data['today'] as int?) ?? _heardToday;
        _assistantReady = data['assistantReady'] == true;
        _fatal = false;
        _lastError = null;
        notifyListeners();
      case ServiceEvents.error:
        _lastError = '${data['message']}';
        _fatal = data['fatal'] == true;
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
