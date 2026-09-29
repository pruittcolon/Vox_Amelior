import 'dart:convert';

import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/automation/template.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/log.dart';

/// Shows a notification (implemented with flutter_local_notifications).
abstract interface class Notifier {
  Future<void> show(String title, String body);
}

/// Carries out the actions of rules that fired.
class ActionExecutor {
  ActionExecutor({
    required this._outbox,
    required this._dispatcher,
    required this._notes,
    required this._notifier,
  });

  final OutboxRepository _outbox;
  final WebhookDispatcher _dispatcher;
  final NoteRepository _notes;
  final Notifier _notifier;

  Future<void> execute(RuleFire fire) async {
    var queuedWebhook = false;
    for (final action in fire.rule.actions) {
      try {
        switch (action) {
          case WebhookAction():
            _enqueueWebhook(fire, action);
            queuedWebhook = true;
          case NotifyAction():
            await _notifier.show(
              renderTemplate(action.title, fire.context),
              renderTemplate(action.body, fire.context),
            );
          case NoteAction():
            _notes.add(renderTemplate(action.template, fire.context), source: fire.rule.name);
        }
      } on Object catch (e, st) {
        Log.e('automation', 'action failed for rule ${fire.rule.name}', e, st);
      }
    }
    if (queuedWebhook) await _dispatcher.flush();
  }

  /// Queues a one-off webhook, used by the "send test" button.
  Future<void> sendTest(WebhookAction action, Map<String, String> context) async {
    _enqueueWebhook(RuleFire(
      AutomationRule(id: 'test', name: 'Test', trigger: const RuleTrigger(), actions: [action]),
      context,
    ), action);
    await _dispatcher.flush();
  }

  void _enqueueWebhook(RuleFire fire, WebhookAction a) {
    final ctx = fire.context;
    final body = (a.bodyTemplate == null || a.bodyTemplate!.trim().isEmpty)
        ? jsonEncode({
            'source': 'vox-amelior',
            'rule': fire.rule.name,
            'text': ctx['text'],
            'speaker': ctx['speaker'],
            'match': ctx['match'],
            'command': ctx['command'],
            'timestamp': ctx['time_iso'],
          })
        : renderTemplate(a.bodyTemplate!, ctx);
    final headers = <String, String>{
      if (a.method.toUpperCase() != 'GET') 'Content-Type': 'application/json',
      for (final e in a.headers.entries) e.key: renderTemplate(e.value, ctx),
    };
    _outbox.enqueue(
      ruleId: fire.rule.id,
      url: renderTemplate(a.url, {for (final e in ctx.entries) e.key: e.value}),
      method: a.method.toUpperCase(),
      headers: headers,
      body: a.method.toUpperCase() == 'GET' ? '' : body,
      secret: a.secret,
      allowInsecure: a.allowInsecureHttp,
    );
  }
}
