import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/capacity_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/prompt_editor.dart';
import 'package:vox_amelior_mobile/ui/voice_clips_screen.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key, required this.services});

  final AppServices services;

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
                title: const Text('Instructions (prompt)'),
                subtitle: Text(st.instructions.isEmpty ? 'Default: short, plain answers' : st.instructions,
                    maxLines: 2, overflow: TextOverflow.ellipsis),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () => showPromptEditor(context, services),
              ),
              ListTile(
                title: const Text('Gemma capacity'),
                subtitle: Text('${capacitySummary(st)} · test this phone'),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => CapacityScreen(services: services))),
              ),
              ListTile(
                title: const Text('Wake word'),
                subtitle: Text(st.wakePhrases.isEmpty ? 'Off — add one to ask Gemma out loud' : st.wakePhrases.join(', ')),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () async {
                  final v = await askText(context, 'Wake words, comma separated (empty = off)', initial: st.wakePhrases.join(', '));
                  if (v == null) return;
                  final list = v.split(',').map((e) => e.trim().toLowerCase()).where((e) => e.isNotEmpty).toList();
                  update(st.copyWith(wakePhrases: list));
                },
              ),
              SwitchListTile(
                title: const Text('Read answers aloud'),
                subtitle: const Text('For questions asked with the wake phrase'),
                value: st.speakReplies,
                onChanged: (v) => update(st.copyWith(speakReplies: v)),
              ),
              const SectionHeader('Voice recognition'),
              SwitchListTile(
                title: const Text('Multiple voice patterns'),
                subtitle: const Text('Recognise each person close up, across the room or with a cold. '
                    'Off = one average voice per person (the original way).'),
                value: st.multiPatterns,
                onChanged: (v) => update(st.copyWith(multiPatterns: v)),
              ),
              SwitchListTile(
                title: const Text('Split lines when the speaker changes'),
                subtitle: Text(services.models.isInstalled(ModelCatalog.diarizer)
                    ? 'NVIDIA Nemotron diarizer: separate lines for quick back-and-forth, and "talking at once" marks.'
                    : 'Needs the speaker-change model (More → Models).'),
                value: st.splitSpeakers,
                onChanged: (v) => update(st.copyWith(splitSpeakers: v)),
              ),
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
                leading: const Icon(Icons.graphic_eq_rounded),
                title: const Text('Voice clips'),
                subtitle: Text(switch (st.clipMode) {
                  ClipMode.off => 'Off',
                  ClipMode.everyone => 'Saving everyone\'s speech',
                  ClipMode.chosen => 'Saving chosen people',
                }),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => VoiceClipsScreen(services: services))),
              ),
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
                subtitle: const Text('People, their voices and saved voice clips are kept.'),
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
