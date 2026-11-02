import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/notify_model.dart';
import '../../widgets/common.dart';

/// The demo's Notification page as a form instead of raw JSON.
class NotifyPage extends StatefulWidget {
  const NotifyPage({super.key});

  @override
  State<NotifyPage> createState() => _NotifyPageState();
}

class _NotifyPageState extends State<NotifyPage> {
  final _apps = <NotifyAppModel>[
    const NotifyAppModel('com.even.test', 'Even'),
    const NotifyAppModel('com.google.android.youtube', 'YouTube'),
  ];
  final _appId = TextEditingController();
  final _appName = TextEditingController();
  final _title = TextEditingController(text: 'Even Realities');
  final _subtitle = TextEditingController(text: 'Notify');
  final _message = TextEditingController(text: 'This is a notification');
  NotifyAppModel? _from;
  int _notifyId = 0;
  bool _savingList = false;
  bool _sending = false;

  @override
  void initState() {
    super.initState();
    _from = _apps.first;
  }

  @override
  void dispose() {
    for (final c in [_appId, _appName, _title, _subtitle, _message]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _saveWhitelist() async {
    final suite = SuiteScope.of(context);
    if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
    setState(() => _savingList = true);
    final ok = await suite.proto.sendAppWhitelist(NotifyWhitelistModel(_apps).toJson());
    if (!mounted) return;
    setState(() => _savingList = false);
    showToast(context, ok ? 'Allowed apps saved on the glasses' : 'The glasses did not confirm');
  }

  Future<void> _send() async {
    final suite = SuiteScope.of(context);
    if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
    final app = _from ?? _apps.first;
    final n = NotifyModel(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      app.identifier,
      _title.text,
      _subtitle.text,
      _message.text,
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      app.displayName,
    );
    setState(() => _sending = true);
    _notifyId = (_notifyId + 1) % 256;
    final ok = await suite.proto.sendNotify(n.toMap(), _notifyId);
    if (!mounted) return;
    setState(() => _sending = false);
    showToast(context, ok ? 'Notification sent' : 'The glasses did not confirm. Is the app allowed?');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Notifications')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
        children: [
          SectionCard(
            title: '1 · Allowed apps',
            icon: Icons.verified_user_rounded,
            iconColor: AppColors.ok,
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              const Text('The glasses only show notifications from apps on this list.',
                  style: TextStyle(color: AppColors.muted, fontSize: 13.5)),
              const SizedBox(height: 10),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final a in _apps)
                  InputChip(
                    label: Text(a.displayName),
                    tooltip: a.identifier,
                    onDeleted: _apps.length > 1
                        ? () => setState(() {
                              _apps.remove(a);
                              if (_from == a) _from = _apps.first;
                            })
                        : null,
                  ),
              ]),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(
                  flex: 3,
                  child: TextField(
                    controller: _appId,
                    decoration: const InputDecoration(hintText: 'Package, e.g. com.whatsapp'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 2,
                  child: TextField(controller: _appName, decoration: const InputDecoration(hintText: 'Name')),
                ),
                IconButton(
                  tooltip: 'Add app',
                  onPressed: () {
                    final id = _appId.text.trim();
                    if (id.isEmpty) return;
                    setState(() {
                      _apps.add(NotifyAppModel(id, _appName.text.trim().isEmpty ? id : _appName.text.trim()));
                      _appId.clear();
                      _appName.clear();
                    });
                  },
                  icon: const Icon(Icons.add_circle_rounded, color: AppColors.ok),
                ),
              ]),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: _savingList ? null : _saveWhitelist,
                icon: const Icon(Icons.upload_rounded),
                label: Text(_savingList ? 'Saving…' : 'Save list to glasses'),
              ),
            ]),
          ),
          const SizedBox(height: 12),
          SectionCard(
            title: '2 · Send a notification',
            icon: Icons.notifications_active_rounded,
            iconColor: AppColors.warn,
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              DropdownButtonFormField<NotifyAppModel>(
                value: _from,
                decoration: const InputDecoration(labelText: 'From app'),
                items: [for (final a in _apps) DropdownMenuItem(value: a, child: Text(a.displayName))],
                onChanged: (a) => setState(() => _from = a),
              ),
              const SizedBox(height: 10),
              TextField(controller: _title, decoration: const InputDecoration(labelText: 'Title')),
              const SizedBox(height: 10),
              TextField(controller: _subtitle, decoration: const InputDecoration(labelText: 'Subtitle')),
              const SizedBox(height: 10),
              TextField(
                controller: _message,
                minLines: 2,
                maxLines: 4,
                decoration: const InputDecoration(labelText: 'Message'),
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: _sending ? null : _send,
                icon: const Icon(Icons.send_rounded),
                label: Text(_sending ? 'Sending…' : 'Send to glasses'),
              ),
            ]),
          ),
        ],
      ),
    );
  }
}
