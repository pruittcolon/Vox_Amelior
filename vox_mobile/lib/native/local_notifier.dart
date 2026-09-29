import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';

/// Phone notifications for automation rules and assistant replies.
/// Works from the UI and from the background service isolate.
class LocalNotifier implements Notifier {
  LocalNotifier._();

  static final LocalNotifier instance = LocalNotifier._();

  static const AndroidNotificationChannel _channel = AndroidNotificationChannel(
    'vox_alerts',
    'Vox alerts',
    description: 'Automation results and assistant replies',
    importance: Importance.high,
  );

  final FlutterLocalNotificationsPlugin _plugin = FlutterLocalNotificationsPlugin();
  bool _ready = false;
  int _nextId = 2000;

  Future<void> initialize() async {
    if (_ready) return;
    await _plugin.initialize(
      settings: const InitializationSettings(android: AndroidInitializationSettings('@mipmap/ic_launcher')),
    );
    await _plugin
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(_channel);
    _ready = true;
  }

  @override
  Future<void> show(String title, String body) async {
    await initialize();
    await _plugin.show(
      id: _nextId++,
      title: title,
      body: body,
      notificationDetails: NotificationDetails(
        android: AndroidNotificationDetails(
          _channel.id,
          _channel.name,
          channelDescription: _channel.description,
          importance: Importance.high,
          priority: Priority.high,
          styleInformation: BigTextStyleInformation(body),
        ),
      ),
    );
  }
}
