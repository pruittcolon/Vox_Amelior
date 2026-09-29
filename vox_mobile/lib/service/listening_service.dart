import 'dart:async';
import 'dart:io';

import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/native/local_notifier.dart';
import 'package:vox_amelior_mobile/service/listening_runtime.dart';

/// File the app writes and the service reads (see ServiceConfig).
Future<File> serviceConfigFile() async =>
    File(p.join((await getApplicationSupportDirectory()).path, 'service_config.json'));

/// Entry point of the foreground service isolate.
@pragma('vm:entry-point')
void startListeningService() {
  FlutterForegroundTask.setTaskHandler(ListeningTaskHandler());
}

class ListeningTaskHandler extends TaskHandler {
  ListeningRuntime? _runtime;

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    try {
      await LocalNotifier.instance.initialize();
      _runtime = await ListeningRuntime.start(
        configFile: await serviceConfigFile(),
        emit: FlutterForegroundTask.sendDataToMain,
        notifier: LocalNotifier.instance,
      );
      await _updateNotification();
    } on Object catch (e, st) {
      Log.e('service', 'failed to start listening', e, st);
      FlutterForegroundTask.sendDataToMain({'type': ServiceEvents.error, 'message': '$e'});
      await FlutterForegroundTask.updateService(
        notificationTitle: 'Vox could not start listening',
        notificationText: '$e',
      );
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
    final runtime = _runtime;
    if (runtime == null || data is! Map) return;
    switch (data['cmd']) {
      case ServiceCommands.pause:
        unawaited(runtime.pause().then((_) => _updateNotification()));
      case ServiceCommands.resume:
        unawaited(runtime.resume().then((_) => _updateNotification()));
      case ServiceCommands.reload:
        runtime.reload();
      case ServiceCommands.ackRequest:
        final id = data['id'];
        if (id is int) runtime.acknowledge(id);
    }
  }

  @override
  void onNotificationButtonPressed(String id) {
    final runtime = _runtime;
    if (runtime == null) return;
    if (id == 'toggle') {
      unawaited((runtime.isPaused ? runtime.resume() : runtime.pause()).then((_) => _updateNotification()));
    }
  }

  @override
  void onNotificationPressed() => FlutterForegroundTask.launchApp('/');

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    await _runtime?.stop();
    _runtime = null;
  }

  Future<void> _updateNotification() async {
    final runtime = _runtime;
    if (runtime == null) return;
    await FlutterForegroundTask.updateService(
      notificationTitle: runtime.isPaused ? 'Vox is paused' : 'Vox is listening',
      notificationText: runtime.statusText(),
      notificationButtons: [NotificationButton(id: 'toggle', text: runtime.isPaused ? 'Resume' : 'Pause')],
    );
  }
}
