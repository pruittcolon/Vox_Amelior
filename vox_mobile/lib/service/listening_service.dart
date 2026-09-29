import 'dart:async';
import 'dart:io';

import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/native/local_notifier.dart';
import 'package:vox_amelior_mobile/service/listening_runtime.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';

/// Key in the plugin's shared store: what the service was started for.
const String kServiceModeKey = 'vox_service_mode';
const String kModeListen = 'listen';
const String kModeDownload = 'download';

/// File the app writes and the service reads (see ServiceConfig).
Future<File> serviceConfigFile() async =>
    File(p.join((await getApplicationSupportDirectory()).path, 'service_config.json'));

/// Entry point of the foreground service isolate.
@pragma('vm:entry-point')
void startListeningService() {
  FlutterForegroundTask.setTaskHandler(VoxTaskHandler());
}

class VoxTaskHandler extends TaskHandler {
  static const List<Duration> _retryDelays = [Duration(seconds: 3), Duration(seconds: 10), Duration(seconds: 30)];

  ListeningRuntime? _runtime;
  String _mode = kModeListen;
  int _failures = 0;
  bool _starting = false;
  Timer? _retry;

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    _mode = await FlutterForegroundTask.getData<String>(key: kServiceModeKey) ?? kModeListen;
    if (_mode == kModeDownload) return; // only keeps the app alive while models download
    await LocalNotifier.instance.initialize();
    await _startRuntime();
  }

  Future<void> _startRuntime() async {
    if (_starting || _runtime != null) return;
    _starting = true;
    _retry?.cancel();
    try {
      _runtime = await ListeningRuntime.start(
        configFile: await serviceConfigFile(),
        emit: FlutterForegroundTask.sendDataToMain,
        notifier: LocalNotifier.instance,
      );
      _failures = 0;
      await _updateNotification();
    } on Object catch (e, st) {
      Log.e('service', 'failed to start listening', e, st);
      _failures++;
      final giveUp = _failures > _retryDelays.length;
      FlutterForegroundTask.sendDataToMain({'type': ServiceEvents.error, 'message': '$e', 'fatal': giveUp});
      if (giveUp) {
        await FlutterForegroundTask.updateService(
          notificationTitle: 'Vox could not start',
          notificationText: '$e — open Vox to retry',
        );
      } else {
        final delay = _retryDelays[_failures - 1];
        await FlutterForegroundTask.updateService(
          notificationTitle: 'Vox is starting',
          notificationText: 'Retrying in ${delay.inSeconds}s ($e)',
        );
        _retry = Timer(delay, () => unawaited(_startRuntime()));
      }
    } finally {
      _starting = false;
    }
  }

  @override
  void onRepeatEvent(DateTime timestamp) {
    final runtime = _runtime;
    if (runtime == null) return;
    unawaited(runtime.tick().then((_) => _updateNotification()));
  }

  @override
  void onReceiveData(Object data) {
    if (data is! Map) return;
    final runtime = _runtime;
    final cmd = data['cmd'];
    if (cmd == ServiceCommands.retryStart) {
      _failures = 0;
      unawaited(_startRuntime());
      return;
    }
    if (runtime == null) return;
    switch (cmd) {
      case ServiceCommands.pause:
        unawaited(runtime.pause().then((_) => _updateNotification()));
      case ServiceCommands.resume:
        unawaited(runtime.resume().then((_) => _updateNotification()));
      case ServiceCommands.reload:
        unawaited(runtime.reload().then((_) => _updateNotification()));
      case ServiceCommands.ask:
        final id = data['id'];
        if (id is int) unawaited(runtime.answer(id).then((_) => _updateNotification()));
      case ServiceCommands.holdTranscription:
        runtime.holdTranscription = data['on'] == true;
      case ServiceCommands.cancelAsk:
        final id = data['id'];
        if (id is int) runtime.cancelAnswer(id);
      case ServiceCommands.probe:
        unawaited(runtime.runProbe());
      case ServiceCommands.reviewKick:
        runtime.reviewWorker.kick();
    }
  }

  @override
  void onNotificationButtonPressed(String id) {
    final runtime = _runtime;
    if (runtime == null || id != 'toggle') return;
    final action = runtime.isManuallyPaused ? runtime.resume() : runtime.pause();
    unawaited(action.then((_) => _updateNotification()));
  }

  @override
  void onNotificationPressed() => FlutterForegroundTask.launchApp('/');

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    _retry?.cancel();
    await _runtime?.stop();
    _runtime = null;
  }

  Future<void> _updateNotification() async {
    final runtime = _runtime;
    if (runtime == null) return;
    await FlutterForegroundTask.updateService(
      notificationTitle: runtime.statusTitle(),
      notificationText: runtime.statusText(),
      notificationButtons: [NotificationButton(id: 'toggle', text: runtime.isManuallyPaused ? 'Resume' : 'Pause')],
    );
  }
}
