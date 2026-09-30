import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';

/// Create or edit an automation rule. Pure UI: persistence is up to [onSave].
class RuleEditorScreen extends StatefulWidget {
  const RuleEditorScreen({super.key, this.initial, required this.people, required this.onSave, this.onTestWebhook});

  final AutomationRule? initial;
  final List<String> people;
  final void Function(AutomationRule rule) onSave;

  /// Sends one webhook right now; returns a human-readable result.
  final Future<String> Function(WebhookAction action)? onTestWebhook;

  @override
  State<RuleEditorScreen> createState() => _RuleEditorScreenState();
}

class _ActionDraft {
  _ActionDraft.webhook([WebhookAction? a])
      : type = 'webhook',
        url = TextEditingController(text: a?.url ?? ''),
        body = TextEditingController(text: a?.bodyTemplate ?? ''),
        secret = TextEditingController(text: a?.secret ?? ''),
        headers = TextEditingController(text: a?.headers.entries.map((e) => '${e.key}: ${e.value}').join('\n') ?? ''),
        method = a?.method ?? 'POST',
        allowHttp = a?.allowInsecureHttp ?? false,
        title = TextEditingController(),
        text = TextEditingController();

  _ActionDraft.notify([NotifyAction? a])
      : type = 'notify',
        title = TextEditingController(text: a?.title ?? 'Vox heard {{speaker}}'),
        text = TextEditingController(text: a?.body ?? '{{text}}'),
        url = TextEditingController(),
        body = TextEditingController(),
        secret = TextEditingController(),
        headers = TextEditingController(),
        method = 'POST',
        allowHttp = false;

  _ActionDraft.note([NoteAction? a])
      : type = 'note',
        text = TextEditingController(text: a?.template ?? '{{text}}'),
        title = TextEditingController(),
        url = TextEditingController(),
        body = TextEditingController(),
        secret = TextEditingController(),
        headers = TextEditingController(),
        method = 'POST',
        allowHttp = false;

  factory _ActionDraft.from(RuleAction a) => switch (a) {
        WebhookAction() => _ActionDraft.webhook(a),
        NotifyAction() => _ActionDraft.notify(a),
        NoteAction() => _ActionDraft.note(a),
      };

  final String type;
  final TextEditingController url, body, secret, headers, title, text;
  String method;
  bool allowHttp;

  RuleAction build() => switch (type) {
        'webhook' => WebhookAction(
            url: url.text.trim(),
            method: method,
            bodyTemplate: body.text.trim().isEmpty ? null : body.text,
            secret: secret.text.trim().isEmpty ? null : secret.text.trim(),
            headers: {
              for (final line in headers.text.split('\n'))
                if (line.contains(':')) line.substring(0, line.indexOf(':')).trim(): line.substring(line.indexOf(':') + 1).trim(),
            },
            allowInsecureHttp: allowHttp,
          ),
        'notify' => NotifyAction(title: title.text, body: text.text),
        _ => NoteAction(template: text.text),
      };

  void dispose() {
    for (final c in [url, body, secret, headers, title, text]) {
      c.dispose();
    }
  }
}

class _RuleEditorScreenState extends State<RuleEditorScreen> {
  late final TextEditingController _name;
  late final TextEditingController _phrases;
  late final TextEditingController _pattern;
  late final TextEditingController _cooldown;
  late TriggerScope _scope;
  String? _speaker;
  late bool _enabled;
  late final List<_ActionDraft> _actions;
  List<String> _errors = const [];

  @override
  void initState() {
    super.initState();
    final r = widget.initial;
    _name = TextEditingController(text: r?.name ?? '');
    _phrases = TextEditingController(text: r?.trigger.phrases.join(', ') ?? '');
    _pattern = TextEditingController(text: r?.trigger.pattern ?? '');
    _cooldown = TextEditingController(text: '${r?.cooldownSeconds ?? 30}');
    _scope = r?.trigger.scope ?? TriggerScope.anySpeech;
    _speaker = r?.trigger.speakerName;
    _enabled = r?.enabled ?? true;
    _actions = r?.actions.map(_ActionDraft.from).toList() ?? [_ActionDraft.notify()];
  }

  @override
  void dispose() {
    for (final c in [_name, _phrases, _pattern, _cooldown]) {
      c.dispose();
    }
    for (final a in _actions) {
      a.dispose();
    }
    super.dispose();
  }

  AutomationRule _build() => AutomationRule(
        id: widget.initial?.id ?? SpeakerRepository.newId(),
        name: _name.text.trim(),
        enabled: _enabled,
        trigger: RuleTrigger(
          scope: _scope,
          phrases: _phrases.text.split(',').map((p) => p.trim()).where((p) => p.isNotEmpty).toList(),
          pattern: _pattern.text.trim().isEmpty ? null : _pattern.text.trim(),
          speakerName: _speaker,
        ),
        actions: _actions.map((a) => a.build()).toList(),
        cooldownSeconds: int.tryParse(_cooldown.text.trim()) ?? -1,
        lastFiredAt: widget.initial?.lastFiredAt,
      );

  void _save() {
    final rule = _build();
    final errors = validateRule(rule);
    setState(() => _errors = errors);
    if (errors.isNotEmpty) return;
    widget.onSave(rule);
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.initial == null ? 'New automation' : 'Edit automation'),
        actions: [TextButton(onPressed: _save, child: const Text('Save'))],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (_errors.isNotEmpty)
            Card(
              color: theme.colorScheme.errorContainer,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [for (final e in _errors) Text('• $e')],
                ),
              ),
            ),
          TextField(
            key: const Key('rule-name'),
            controller: _name,
            decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Enabled'),
            value: _enabled,
            onChanged: (v) => setState(() => _enabled = v),
          ),
          Text('When', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          SegmentedButton<TriggerScope>(
            segments: const [
              ButtonSegment(value: TriggerScope.anySpeech, label: Text('Anyone says')),
              ButtonSegment(value: TriggerScope.wakeCommand, label: Text('"Hey Vox, …"')),
            ],
            selected: {_scope},
            onSelectionChanged: (v) => setState(() => _scope = v.first),
          ),
          const SizedBox(height: 12),
          TextField(
            key: const Key('rule-phrases'),
            controller: _phrases,
            decoration: InputDecoration(
              labelText: 'Phrases (comma separated)',
              hintText: 'turn on the lights, lights on',
              helperText: _scope == TriggerScope.wakeCommand ? 'Leave empty to run on every command.' : null,
              border: const OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _pattern,
            decoration: const InputDecoration(
              labelText: 'Or a pattern (advanced, regular expression)',
              hintText: r'remind me to (.+)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String?>(
            initialValue: widget.people.contains(_speaker) ? _speaker : null,
            decoration: const InputDecoration(labelText: 'Only when said by', border: OutlineInputBorder()),
            items: [
              const DropdownMenuItem<String?>(value: null, child: Text('Anyone')),
              for (final p in widget.people) DropdownMenuItem<String?>(value: p, child: Text(p)),
            ],
            onChanged: (v) => setState(() => _speaker = v),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _cooldown,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(
              labelText: 'Wait at least (seconds) before firing again',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Text('Then', style: theme.textTheme.titleMedium),
              const Spacer(),
              PopupMenuButton<String>(
                tooltip: 'Add action',
                icon: const Icon(Icons.add),
                onSelected: (t) => setState(() => _actions.add(switch (t) {
                      'webhook' => _ActionDraft.webhook(),
                      'notify' => _ActionDraft.notify(),
                      _ => _ActionDraft.note(),
                    })),
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'webhook', child: Text('Call a webhook')),
                  PopupMenuItem(value: 'notify', child: Text('Show a notification')),
                  PopupMenuItem(value: 'note', child: Text('Save a note')),
                ],
              ),
            ],
          ),
          Text(
            'Placeholders: {{text}} {{speaker}} {{match}} {{command}} {{time}} {{date}}. '
            'Add |json or |url to escape, e.g. {{text|json}}.',
            style: theme.textTheme.bodySmall,
          ),
          for (var i = 0; i < _actions.length; i++) _actionCard(context, i),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _actionCard(BuildContext context, int i) {
    final a = _actions[i];
    final remove = IconButton(
      tooltip: 'Remove action',
      icon: const Icon(Icons.delete_outline),
      onPressed: () => setState(() => _actions.removeAt(i).dispose()),
    );
    final children = <Widget>[];
    switch (a.type) {
      case 'webhook':
        children.addAll([
          Row(children: [const Icon(Icons.webhook), const SizedBox(width: 8), const Text('Webhook'), const Spacer(), remove]),
          Row(
            children: [
              DropdownButton<String>(
                value: a.method,
                items: [for (final m in const ['POST', 'GET', 'PUT', 'PATCH']) DropdownMenuItem(value: m, child: Text(m))],
                onChanged: (v) => setState(() => a.method = v ?? 'POST'),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  key: Key('webhook-url-$i'),
                  controller: a.url,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(labelText: 'URL', hintText: 'https://…'),
                ),
              ),
            ],
          ),
          TextField(
            controller: a.body,
            maxLines: 4,
            minLines: 1,
            decoration: const InputDecoration(labelText: 'Body (optional; default is JSON with text and speaker)'),
          ),
          TextField(controller: a.headers, maxLines: 3, minLines: 1, decoration: const InputDecoration(labelText: 'Headers (one "Name: value" per line)')),
          TextField(controller: a.secret, decoration: const InputDecoration(labelText: 'Signing secret (optional, adds X-Vox-Signature)')),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Allow plain http:// (home network devices)'),
            value: a.allowHttp,
            onChanged: (v) => setState(() => a.allowHttp = v),
          ),
          if (widget.onTestWebhook != null)
            Align(
              alignment: Alignment.centerRight,
              child: OutlinedButton.icon(
                icon: const Icon(Icons.send),
                label: const Text('Send test'),
                onPressed: () async {
                  final action = a.build() as WebhookAction;
                  final problems = validateRule(AutomationRule(
                    id: 't',
                    name: 't',
                    trigger: const RuleTrigger(scope: TriggerScope.wakeCommand),
                    actions: [action],
                  ));
                  if (problems.isNotEmpty) {
                    setState(() => _errors = problems);
                    return;
                  }
                  final result = await widget.onTestWebhook!(action);
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(result)));
                  }
                },
              ),
            ),
        ]);
      case 'notify':
        children.addAll([
          Row(children: [const Icon(Icons.notifications), const SizedBox(width: 8), const Text('Notification'), const Spacer(), remove]),
          TextField(controller: a.title, decoration: const InputDecoration(labelText: 'Title')),
          TextField(controller: a.text, decoration: const InputDecoration(labelText: 'Text')),
        ]);
      default:
        children.addAll([
          Row(children: [const Icon(Icons.sticky_note_2), const SizedBox(width: 8), const Text('Save note'), const Spacer(), remove]),
          TextField(controller: a.text, decoration: const InputDecoration(labelText: 'Note text')),
        ]);
    }
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: children),
      ),
    );
  }
}
