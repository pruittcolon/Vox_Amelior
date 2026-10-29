import 'dart:convert';

/// A notification for the glasses (the demo's NotifyModel, same JSON keys).
class NotifyModel {
  final int msgId;
  final String appIdentifier;
  final String title;
  final String subTitle;
  final String message;
  final int timestamp;
  final String displayName;

  const NotifyModel(this.msgId, this.appIdentifier, this.title, this.subTitle, this.message,
      this.timestamp, this.displayName);

  Map<String, dynamic> toMap() => {
        'msg_id': msgId,
        'app_identifier': appIdentifier,
        'title': title,
        'subtitle': subTitle,
        'message': message,
        'time_s': timestamp,
        'display_name': displayName,
      };

  String toJson() => jsonEncode(toMap());
}

/// Apps allowed to show notifications on the glasses (the demo's whitelist).
class NotifyWhitelistModel {
  final List<NotifyAppModel> apps;
  const NotifyWhitelistModel(this.apps);

  Map<String, dynamic> toMap() => {
        'calendar_enable': false,
        'call_enable': false,
        'msg_enable': false,
        'ios_mail_enable': false,
        'app': {
          'list': apps.map((app) => app.toMap()).toList(),
          'enable': true,
        },
      };

  String toJson() => jsonEncode(toMap());
}

class NotifyAppModel {
  final String identifier;
  final String displayName;
  const NotifyAppModel(this.identifier, this.displayName);

  Map<String, String> toMap() => {'id': identifier, 'name': displayName};
}
