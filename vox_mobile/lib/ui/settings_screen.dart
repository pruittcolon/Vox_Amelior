import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/appearance_screen.dart';
import 'package:vox_amelior_mobile/ui/capacity_screen.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/listening_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/mic_tune.dart';
import 'package:vox_amelior_mobile/ui/prompt_editor.dart';
import 'package:vox_amelior_mobile/ui/speakers_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/voice_clips_screen.dart';

/// [st] with every microphone, hearing and speaker setting back at its default.
AppSettings restoreListeningDefaults(AppSettings st) {
  const d = AppSettings();
  return st.copyWith(
    micGain: d.micGain,
    vadThreshold: d.vadThreshold,
    pauseSeconds: d.pauseSeconds,
    minSpeechSeconds: d.minSpeechSeconds,
    matchThreshold: d.matchThreshold,
    matchMargin: d.matchMargin,
    guestThreshold: d.guestThreshold,
    multiPatterns: d.multiPatterns,
    splitSpeakers: d.splitSpeakers,
    splitMinSeconds: d.splitMinSeconds,
    splitMinWords: d.splitMinWords,
  );
}

/// Settings home: a few clear categories, each with a live summary of what is set.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key, required this.services});

  final AppServices services;

  void _go(BuildContext context, Widget screen) => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => screen));

  String _hearingSummary(AppSettings st) {
    final preset = MicPreset.of(st);
    return 'Mic boost ${formatBoost(st.micGain)} · ${preset?.name ?? 'Custom'} · ${st.speechModel} model';
  }

  String _speakersSummary(AppSettings st) =>
      'Naming at ${(st.matchThreshold * 100).round()}% · lines ${st.splitSpeakers ? 'split at speaker changes' : 'kept whole'}';

  String _privacySummary(AppSettings st) {
    final keep = switch (st.retentionDays) {
      0 => 'Kept forever',
      7 => 'Kept 1 week',
      30 => 'Kept 1 month',
      90 => 'Kept 3 months',
      365 => 'Kept 1 year',
      final d => 'Kept $d days',
    };
    return '$keep · voice clips ${st.clipMode == ClipMode.off ? 'off' : 'on'}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) => ListView(
          padding: const EdgeInsets.only(bottom: 32),
          children: [
            SettingsGroup(
              children: [
                NavRow(
                  icon: Icons.mic_rounded,
                  title: 'Microphone & hearing',
                  summary: _hearingSummary(st),
                  onTap: () => _go(context, ListeningSettingsScreen(services: services)),
                ),
                NavRow(
                  icon: Icons.record_voice_over_rounded,
                  title: 'Voices & speakers',
                  summary: _speakersSummary(st),
                  color: const Color(0xFF0B7285),
                  onTap: () => _go(context, SpeakersSettingsScreen(services: services)),
                ),
                NavRow(
                  icon: Icons.auto_awesome_rounded,
                  title: 'Assistant',
                  summary: '${st.agentMode ? 'Agent mode on' : 'Agent mode off'} · ${st.wakePhrases.isEmpty ? 'no wake word' : 'wake word set'}',
                  color: const Color(0xFF9C36B5),
                  onTap: () => _go(context, _AssistantSettings(services: services)),
                ),
                NavRow(
                  icon: Icons.lock_rounded,
                  title: 'Privacy & data',
                  summary: _privacySummary(st),
                  color: const Color(0xFF2B8A3E),
                  onTap: () => _go(context, _PrivacySettings(services: services)),
                ),
                NavRow(
                  icon: Icons.palette_rounded,
                  title: 'Appearance',
                  summary: 'Colour, dark mode, text size',
                  color: const Color(0xFFE8590C),
                  onTap: () => _go(context, AppearanceScreen(services: services)),
                ),
              ],
            ),
            SettingsGroup(
              footer: 'Puts the microphone, hearing and speaker settings back to what Vox recommends. '
                  'Your speech model, people, assistant and appearance are not changed.',
              children: [
                ListTile(
                  leading: const IconBadge(Icons.restart_alt_rounded, size: 36),
                  title: const Text('Restore recommended settings'),
                  onTap: () async {
                    if (!await confirm(context, 'Restore recommended settings?', 'Microphone boost, sensitivity, pauses and speaker settings go back to their defaults.', action: 'Restore')) return;
                    await services.updateSettings(restoreListeningDefaults(services.settings.value));
                    if (context.mounted) showMessage(context, 'Restored.');
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _AssistantSettings extends StatelessWidget {
  const _AssistantSettings({required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Assistant')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) {
          void update(AppSettings next) => services.updateSettings(next);
          return ListView(
            padding: const EdgeInsets.only(bottom: 32),
            children: [
              SettingsGroup(
                children: [
                  SettingSwitch(
                    title: 'Agent mode',
                    subtitle: 'Let Gemma search, read timelines, save notes, set reminders and run automations on its own',
                    value: st.agentMode,
                    onChanged: (v) => update(st.copyWith(agentMode: v)),
                  ),
                  ListTile(
                    title: const Text('Instructions (prompt)'),
                    subtitle: Text(st.instructions.isEmpty ? 'Default: short, plain answers' : st.instructions, maxLines: 2, overflow: TextOverflow.ellipsis),
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
                  SettingSwitch(
                    title: 'Read answers aloud',
                    subtitle: 'For questions asked with the wake phrase',
                    value: st.speakReplies,
                    onChanged: (v) => update(st.copyWith(speakReplies: v)),
                  ),
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}

class _PrivacySettings extends StatelessWidget {
  const _PrivacySettings({required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Privacy & data')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) {
          void update(AppSettings next) => services.updateSettings(next);
          return ListView(
            padding: const EdgeInsets.only(bottom: 32),
            children: [
              SettingsGroup(
                children: [
                  ListTile(
                    leading: const IconBadge(Icons.graphic_eq_rounded, size: 36),
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
                    leading: const IconBadge(Icons.copy_all_rounded, size: 36),
                    title: const Text('Copy all transcripts'),
                    subtitle: Text('${services.transcripts.count()} lines stored on this phone'),
                    onTap: () async {
                      await Clipboard.setData(ClipboardData(text: services.transcripts.exportText()));
                      if (context.mounted) showMessage(context, 'Copied to clipboard.');
                    },
                  ),
                  ListTile(
                    leading: const IconBadge(Icons.save_alt_rounded, size: 36),
                    title: const Text('Save all transcripts to a file'),
                    subtitle: const Text('A text file you can keep, and import again later'),
                    onTap: () => _saveTranscripts(context, services),
                  ),
                  ListTile(
                    leading: const IconBadge(Icons.upload_file_rounded, size: 36),
                    title: const Text('Import transcripts'),
                    subtitle: const Text('From a saved file or copied text (e.g. from before reinstalling)'),
                    onTap: () => _importTranscripts(context, services),
                  ),
                  ListTile(
                    leading: IconBadge(Icons.delete_forever_rounded, size: 36, color: Theme.of(context).colorScheme.error),
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
              ),
            ],
          );
        },
      ),
    );
  }
}

Future<void> _saveTranscripts(BuildContext context, AppServices services) async {
  final now = DateTime.now();
  final name = 'vox-transcripts-${now.year}-${two(now.month)}-${two(now.day)}.txt';
  try {
    final saved = await FilePicker.saveFile(
      fileName: name,
      bytes: utf8.encode(services.transcripts.exportText()),
      mimeType: 'text/plain',
    );
    if (saved != null && context.mounted) showMessage(context, 'Saved $name');
  } on Object catch (e) {
    if (context.mounted) showMessage(context, 'Could not save: $e');
  }
}

Future<void> _importTranscripts(BuildContext context, AppServices services) async {
  final source = await showModalBottomSheet<String>(
    context: context,
    builder: (c) => SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const ListTile(
            title: Text('Import transcripts'),
            subtitle: Text('Use text from "Copy all transcripts" or a file from "Save all transcripts to a file". '
                'Conversations already on this phone are skipped.'),
          ),
          ListTile(
            leading: const Icon(Icons.content_paste_rounded),
            title: const Text('Paste copied text'),
            onTap: () => Navigator.pop(c, 'paste'),
          ),
          ListTile(
            leading: const Icon(Icons.folder_open_rounded),
            title: const Text('Choose a file'),
            onTap: () => Navigator.pop(c, 'file'),
          ),
        ],
      ),
    ),
  );
  if (source == null) return;
  String? text;
  try {
    if (source == 'paste') {
      text = (await Clipboard.getData(Clipboard.kTextPlain))?.text;
    } else {
      final picked = await FilePicker.pickFile();
      final path = picked?.path;
      if (path != null) text = await File(path).readAsString();
    }
  } on Object catch (e) {
    if (context.mounted) showMessage(context, 'Could not read it: $e');
    return;
  }
  if (text == null || text.trim().isEmpty) {
    if (context.mounted) showMessage(context, source == 'paste' ? 'The clipboard is empty.' : 'Nothing to import.');
    return;
  }
  final r = services.transcripts.importText(text);
  services.dataChanged();
  if (!context.mounted) return;
  if (r.conversations == 0 && r.skipped == 0) {
    showMessage(context, 'No transcripts found in that text.');
  } else {
    showMessage(context,
        'Imported ${r.conversations} conversation${r.conversations == 1 ? '' : 's'} (${r.lines} lines)'
        '${r.skipped > 0 ? ', ${r.skipped} already here' : ''}.');
  }
}
