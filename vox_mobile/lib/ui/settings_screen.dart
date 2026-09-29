import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key, required this.services});

  final AppServices services;

  Future<void> _editInstructions(BuildContext context, AppSettings st) async {
    final controller = TextEditingController(text: st.instructions.isEmpty ? PromptBuilder.defaultInstructions : st.instructions);
    final result = await showDialog<String>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Assistant instructions'),
        content: SizedBox(
          width: double.maxFinite,
          child: TextField(
            controller: controller,
            maxLines: 10,
            minLines: 5,
            decoration: const InputDecoration(hintText: 'How should Vox answer?'),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, ''), child: const Text('Reset')),
          TextButton(onPressed: () => Navigator.pop(c), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(c, controller.text.trim()), child: const Text('Save')),
        ],
      ),
    );
    controller.dispose();
    if (result == null) return;
    final value = result == PromptBuilder.defaultInstructions ? '' : result;
    await services.updateSettings(st.copyWith(instructions: value));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) {
          void update(AppSettings next) => services.updateSettings(next);
          return ListView(
            padding: const EdgeInsets.only(bottom: 32),
            children: [
              const SectionHeader('Assistant'),
              SwitchListTile(
                title: const Text('Agent mode'),
                subtitle: const Text('Let Gemma search, read timelines, save notes, set reminders and run automations on its own'),
                value: st.agentMode,
                onChanged: (v) => update(st.copyWith(agentMode: v)),
              ),
              ListTile(
                title: const Text('Instructions'),
                subtitle: Text(st.instructions.isEmpty ? 'Default' : st.instructions, maxLines: 2, overflow: TextOverflow.ellipsis),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () => _editInstructions(context, st),
              ),
              ListTile(
                title: const Text('Wake phrases'),
                subtitle: Text(st.wakePhrases.join(', ')),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () async {
                  final v = await askText(context, 'Wake phrases (comma separated)', initial: st.wakePhrases.join(', '));
                  if (v == null) return;
                  final list = v.split(',').map((e) => e.trim().toLowerCase()).where((e) => e.isNotEmpty).toList();
                  if (list.isNotEmpty) update(st.copyWith(wakePhrases: list));
                },
              ),
              SwitchListTile(
                title: const Text('Read answers aloud'),
                subtitle: const Text('For questions asked with the wake phrase'),
                value: st.speakReplies,
                onChanged: (v) => update(st.copyWith(speakReplies: v)),
              ),
              const SectionHeader('Voice recognition'),
              _slider(context, 'How sure before naming someone', st.matchThreshold, 0.3, 0.9,
                  (v) => update(st.copyWith(matchThreshold: v)),
                  help: 'Higher = fewer wrong names, more "Guest" labels.'),
              _slider(context, 'Grouping of unknown voices', st.guestThreshold, 0.3, 0.9,
                  (v) => update(st.copyWith(guestThreshold: v)),
                  help: 'Higher = more separate guests.'),
              _slider(context, 'Speech detection sensitivity', st.vadThreshold, 0.2, 0.8,
                  (v) => update(st.copyWith(vadThreshold: v)),
                  help: 'Lower hears quieter speech but more noise. Applies next time Vox starts.'),
              const SectionHeader('Privacy & data'),
              ListTile(
                title: const Text('Keep transcripts for'),
                trailing: DropdownButton<int>(
                  value: const [7, 30, 90, 365, 0].contains(st.retentionDays) ? st.retentionDays : 90,
                  underline: const SizedBox.shrink(),
                  items: const [
                    DropdownMenuItem(value: 7, child: Text('1 week')),
                    DropdownMenuItem(value: 30, child: Text('1 month')),
                    DropdownMenuItem(value: 90, child: Text('3 months')),
                    DropdownMenuItem(value: 365, child: Text('1 year')),
                    DropdownMenuItem(value: 0, child: Text('Forever')),
                  ],
                  onChanged: (v) => update(st.copyWith(retentionDays: v)),
                ),
              ),
              ListTile(
                leading: const Icon(Icons.copy_all_rounded),
                title: const Text('Copy all transcripts'),
                subtitle: Text('${services.transcripts.count()} lines stored on this phone'),
                onTap: () async {
                  await Clipboard.setData(ClipboardData(text: services.transcripts.exportText()));
                  if (context.mounted) showMessage(context, 'Copied to clipboard.');
                },
              ),
              ListTile(
                leading: Icon(Icons.delete_forever_rounded, color: Theme.of(context).colorScheme.error),
                title: const Text('Delete all transcripts'),
                subtitle: const Text('People and their voices are kept.'),
                onTap: () async {
                  if (await confirm(context, 'Delete all transcripts?', 'This cannot be undone.')) {
                    services.transcripts.deleteAllTranscripts();
                    services.dataChanged();
                    if (context.mounted) showMessage(context, 'Deleted.');
                  }
                },
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _slider(BuildContext context, String title, double value, double min, double max, ValueChanged<double> onChanged,
          {String? help}) =>
      ListTile(
        title: Text(title),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Slider(value: value.clamp(min, max), min: min, max: max, divisions: 12, label: value.toStringAsFixed(2), onChanged: onChanged),
            if (help != null) Text(help),
          ],
        ),
      );
}
