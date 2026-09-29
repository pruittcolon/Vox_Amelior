import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/setup_screen.dart';

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
            children: [
              ListTile(
                leading: const Icon(Icons.download),
                title: const Text('Models & Hugging Face token'),
                subtitle: Text(services.gemmaReady ? 'Speech and assistant installed' : 'Download or manage models'),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute<void>(builder: (_) => SetupScreen(services: services)),
                ),
              ),
              const Divider(),
              _header(context, 'Assistant'),
              ListTile(
                title: const Text('Wake phrases'),
                subtitle: Text(st.wakePhrases.join(', ')),
                onTap: () async {
                  final v = await askText(context, 'Wake phrases (comma separated)', initial: st.wakePhrases.join(', '));
                  if (v == null) return;
                  final list = v.split(',').map((e) => e.trim().toLowerCase()).where((e) => e.isNotEmpty).toList();
                  if (list.isNotEmpty) update(st.copyWith(wakePhrases: list));
                },
              ),
              SwitchListTile(
                title: const Text('Read answers aloud'),
                value: st.speakReplies,
                onChanged: (v) => update(st.copyWith(speakReplies: v)),
              ),
              ListTile(
                title: const Text('Custom Gemma download address'),
                subtitle: Text(st.gemmaUrlOverride ?? 'Default (Google on Hugging Face)'),
                onTap: () async {
                  final v = await askText(context, 'Gemma .litertlm URL', initial: st.gemmaUrlOverride ?? '', hint: 'https://…');
                  if (v == null) return;
                  update(v.isEmpty ? st.copyWith(clearGemmaUrl: true) : st.copyWith(gemmaUrlOverride: v));
                },
              ),
              const Divider(),
              _header(context, 'Voice recognition'),
              _slider(context, 'How sure before naming someone', st.matchThreshold, 0.3, 0.9,
                  (v) => update(st.copyWith(matchThreshold: v)),
                  help: 'Higher = fewer wrong names, more "Guest" labels.'),
              _slider(context, 'Grouping of unknown voices', st.guestThreshold, 0.3, 0.9,
                  (v) => update(st.copyWith(guestThreshold: v)),
                  help: 'Higher = more separate guests.'),
              _slider(context, 'Speech detection sensitivity', st.vadThreshold, 0.2, 0.8,
                  (v) => update(st.copyWith(vadThreshold: v)),
                  help: 'Lower hears quieter speech but more noise. Restart listening to apply.'),
              const Divider(),
              _header(context, 'Privacy & data'),
              ListTile(
                title: const Text('Keep transcripts for'),
                trailing: DropdownButton<int>(
                  value: const [7, 30, 90, 365, 0].contains(st.retentionDays) ? st.retentionDays : 90,
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
                leading: const Icon(Icons.copy_all),
                title: const Text('Copy all transcripts'),
                subtitle: Text('${services.transcripts.count()} utterances stored on this phone'),
                onTap: () async {
                  await Clipboard.setData(ClipboardData(text: services.transcripts.exportText()));
                  if (context.mounted) showMessage(context, 'Copied to clipboard.');
                },
              ),
              ListTile(
                leading: Icon(Icons.delete_forever, color: Theme.of(context).colorScheme.error),
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
              const AboutListTile(
                applicationName: 'Vox Amelior',
                aboutBoxChildren: [
                  Text('Private, on-device assistant. Speech: NVIDIA Parakeet, Silero VAD, TitaNet via sherpa-onnx. '
                      'Assistant: Google Gemma 3n via LiteRT-LM.'),
                ],
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _header(BuildContext context, String text) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
        child: Text(text, style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Theme.of(context).colorScheme.primary)),
      );

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
