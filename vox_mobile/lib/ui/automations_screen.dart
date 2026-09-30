import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/rule_editor_screen.dart';

/// Rules that react to speech, their webhook deliveries, and saved notes.
class AutomationsScreen extends StatefulWidget {
  const AutomationsScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<AutomationsScreen> createState() => _AutomationsScreenState();
}

class _AutomationsScreenState extends State<AutomationsScreen> {
  int _tab = 0;

  AppServices get s => widget.services;

  Future<void> _edit([AutomationRule? rule]) async {
    await Navigator.push(
      context,
      MaterialPageRoute<void>(
        builder: (_) => RuleEditorScreen(
          initial: rule,
          people: s.speakers.profiles().map((p) => p.name).toList(),
          onSave: (r) {
            s.rules.save(r);
            s.dataChanged();
          },
          onTestWebhook: _test,
        ),
      ),
    );
  }

  Future<String> _test(WebhookAction action) async {
    final now = DateTime.now();
    await s.executor.sendTest(action, {
      'text': 'This is a test from Vox',
      'speaker': 'Test',
      'match': 'test',
      'command': 'test',
      'time_iso': now.toIso8601String(),
      'date': '${now.year}-${two(now.month)}-${two(now.day)}',
      'time': formatTime(now),
      'segment_id': '0',
      'conversation_id': '0',
      'rule': 'Test',
    });
    final last = s.outbox.recent(limit: 1);
    if (last.isEmpty) return 'Nothing was sent.';
    final item = last.first;
    return switch (item.status) {
      OutboxStatus.delivered => 'Delivered ✓',
      OutboxStatus.pending => 'Not delivered yet (${item.lastError ?? 'will retry'})',
      OutboxStatus.failed => 'Failed: ${item.lastError}',
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Automations'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(56),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: SegmentedButton<int>(
              segments: const [
                ButtonSegment(value: 0, label: Text('Rules')),
                ButtonSegment(value: 1, label: Text('Deliveries')),
                ButtonSegment(value: 2, label: Text('Notes')),
              ],
              selected: {_tab},
              onSelectionChanged: (v) => setState(() => _tab = v.first),
            ),
          ),
        ),
      ),
      floatingActionButton: _tab == 0
          ? FloatingActionButton.extended(onPressed: _edit, icon: const Icon(Icons.add), label: const Text('New rule'))
          : null,
      body: ValueListenableBuilder<int>(
        valueListenable: s.dataVersion,
        builder: (context, _, _) => switch (_tab) {
          0 => _rules(context),
          1 => _deliveries(context),
          _ => _notes(context),
        },
      ),
    );
  }

  Widget _rules(BuildContext context) {
    s.rules.invalidate(); // the service may have updated "last fired"
    final rules = s.rules.all();
    if (rules.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(24),
        child: Text(
          'Rules react to what is said, about 2 seconds after the sentence ends '
          '(rules about one person wait until speakers are checked). Examples:\n\n'
          '• When anyone says "add to the shopping list", save a note.\n'
          '• When anyone says "lights on in the kitchen", call your Home Assistant webhook.\n'
          '• When Sam says "I\'m leaving", send a notification.',
        ),
      );
    }
    return ListView(
      padding: const EdgeInsets.only(bottom: 96),
      children: [
        for (final r in rules)
          ListTile(
            title: Text(r.name),
            subtitle: Text(_describe(r), maxLines: 2, overflow: TextOverflow.ellipsis),
            onTap: () => _edit(r),
            onLongPress: () async {
              if (await confirm(context, 'Delete "${r.name}"?', 'This rule will stop running.')) {
                s.rules.delete(r.id);
                s.dataChanged();
              }
            },
            trailing: Switch(
              value: r.enabled,
              onChanged: (v) {
                s.rules.setEnabled(r.id, enabled: v);
                s.dataChanged();
              },
            ),
          ),
      ],
    );
  }

  String _describe(AutomationRule r) {
    final t = r.trigger;
    final when = t.scope == TriggerScope.wakeCommand ? 'Command' : 'Speech';
    final what = t.phrases.isNotEmpty ? '"${t.phrases.join('", "')}"' : (t.pattern ?? 'anything');
    final who = t.speakerName == null ? '' : ' by ${t.speakerName}';
    final actions = r.actions.map((a) => switch (a) {
          WebhookAction() => 'webhook',
          NotifyAction() => 'notify',
          NoteAction() => 'note',
        });
    return '$when: $what$who → ${actions.join(', ')}';
  }

  Widget _deliveries(BuildContext context) {
    final items = s.outbox.recent();
    if (items.isEmpty) return const Center(child: Text('No webhook deliveries yet.'));
    return ListView(
      children: [
        for (final i in items)
          ListTile(
            leading: Icon(switch (i.status) {
              OutboxStatus.delivered => Icons.check_circle,
              OutboxStatus.pending => Icons.schedule,
              OutboxStatus.failed => Icons.error,
            }, color: switch (i.status) {
              OutboxStatus.delivered => Colors.green,
              OutboxStatus.pending => Colors.orange,
              OutboxStatus.failed => Theme.of(context).colorScheme.error,
            }),
            title: Text('${i.method} ${Uri.tryParse(i.url)?.host ?? i.url}'),
            subtitle: Text('${formatDay(i.createdAt)} ${formatTime(i.createdAt)} · ${i.attempts} attempt(s)'
                '${i.lastError == null ? '' : ' · ${i.lastError}'}'),
            trailing: i.status == OutboxStatus.failed
                ? IconButton(
                    tooltip: 'Retry',
                    icon: const Icon(Icons.refresh),
                    onPressed: () {
                      s.outbox.retryNow(i.id);
                      s.dataChanged();
                    },
                  )
                : null,
          ),
      ],
    );
  }

  Widget _notes(BuildContext context) {
    final notes = s.notes.all();
    if (notes.isEmpty) return const Center(child: Text('Notes saved by rules appear here.'));
    return ListView(
      children: [
        for (final n in notes)
          ListTile(
            title: Text(n.text),
            subtitle: Text('${n.source} · ${formatDay(n.createdAt)} ${formatTime(n.createdAt)}'),
            trailing: IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: () {
                s.notes.delete(n.id);
                s.dataChanged();
              },
            ),
          ),
      ],
    );
  }
}
