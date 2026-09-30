import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Keeps the audio of what was said (with its text) so Vox's voice and
/// speech models can be trained on your household later. Off by default.
class VoiceClipsScreen extends StatefulWidget {
  const VoiceClipsScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<VoiceClipsScreen> createState() => _VoiceClipsScreenState();
}

class _VoiceClipsScreenState extends State<VoiceClipsScreen> {
  static const List<int> _limits = [512, 1024, 2048, 5120, 10240, 20480];

  AppServices get s => widget.services;
  late ClipStats _stats = s.clips.stats();

  void _refresh() => setState(() => _stats = s.clips.stats());

  Future<void> _deleteAll() async {
    if (!await confirm(context, 'Delete all voice clips?', 'Transcripts are kept; only the saved audio is removed.')) return;
    s.clips.deleteAll();
    _refresh();
    if (mounted) showMessage(context, 'All clips deleted.');
  }

  Future<void> _deleteGroup(ClipGroup g) async {
    if (!await confirm(context, 'Delete ${g.name}\'s clips?', '${g.count} clips · ${formatBytes(g.bytes)}')) return;
    s.clips.deleteForSpeaker(g.speakerId);
    _refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Voice clips')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: s.settings,
        builder: (context, st, _) {
          final t = Theme.of(context);
          void set(AppSettings next) => s.updateSettings(next);
          final on = st.clipMode != ClipMode.off;
          final limitBytes = st.clipLimitMb * 1024 * 1024;
          final used = (_stats.bytes / limitBytes).clamp(0.0, 1.0);
          final people = s.speakers.profiles();
          return RefreshIndicator(
            onRefresh: () async => _refresh(),
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
              children: [
                VoxCard(
                  padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(color: t.colorScheme.primaryContainer, shape: BoxShape.circle),
                        child: Icon(Icons.graphic_eq_rounded, color: t.colorScheme.onPrimaryContainer),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Save voice clips', style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                            const Text('Keeps each line\'s audio with its text, for training Vox on your voices later. '
                                'Stays on this phone.'),
                          ],
                        ),
                      ),
                      Switch(
                        value: on,
                        onChanged: (v) => set(st.copyWith(clipMode: v ? ClipMode.everyone : ClipMode.off)),
                      ),
                    ],
                  ),
                ),
                if (on) ...[
                  const SectionHeader('Whose voices', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
                  SegmentedButton<ClipMode>(
                    segments: const [
                      ButtonSegment(value: ClipMode.everyone, label: Text('Everyone'), icon: Icon(Icons.groups_rounded)),
                      ButtonSegment(value: ClipMode.chosen, label: Text('Only chosen'), icon: Icon(Icons.person_pin_rounded)),
                    ],
                    selected: {st.clipMode},
                    onSelectionChanged: (v) => set(st.copyWith(clipMode: v.first)),
                  ),
                  if (st.clipMode == ClipMode.chosen) ...[
                    const SizedBox(height: 12),
                    if (people.isEmpty)
                      const Text('Add people on the People tab first.')
                    else
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          for (final p in people)
                            FilterChip(
                              avatar: SpeakerAvatar(label: p.name, radius: 12),
                              label: Text(p.name),
                              selected: st.clipPeople.contains(p.id),
                              onSelected: (sel) => set(st.copyWith(
                                clipPeople: sel ? [...st.clipPeople, p.id] : st.clipPeople.where((id) => id != p.id).toList(),
                              )),
                            ),
                        ],
                      ),
                    if (people.isNotEmpty && st.clipPeople.isEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text('Pick at least one person, or nothing is saved.', style: TextStyle(color: t.colorScheme.error)),
                      ),
                  ],
                ],
                const SectionHeader('Storage', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
                VoxCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Wrap(
                        crossAxisAlignment: WrapCrossAlignment.end,
                        spacing: 6,
                        children: [
                          Text(formatBytes(_stats.bytes), style: t.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
                          Padding(
                            padding: const EdgeInsets.only(bottom: 3),
                            child: Text('of ${formatBytes(limitBytes)}', style: t.textTheme.bodyMedium),
                          ),
                        ],
                      ),
                      Text('${_stats.count} clips · ${formatDuration(_stats.duration)} of speech',
                          style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                      const SizedBox(height: 10),
                      LinearProgressIndicator(
                        value: used,
                        minHeight: 8,
                        color: used > 0.9 ? t.colorScheme.error : null,
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          const Expanded(child: Text('Limit')),
                          DropdownButton<int>(
                            value: _limits.contains(st.clipLimitMb) ? st.clipLimitMb : 2048,
                            underline: const SizedBox.shrink(),
                            items: [
                              for (final mb in _limits) DropdownMenuItem(value: mb, child: Text(formatBytes(mb * 1024 * 1024))),
                            ],
                            onChanged: (v) {
                              if (v != null) set(st.copyWith(clipLimitMb: v));
                            },
                          ),
                        ],
                      ),
                      Text('When full, Vox stops saving clips. Nothing is deleted automatically. '
                          'About 115 MB per hour of speech.', style: t.textTheme.bodySmall),
                    ],
                  ),
                ),
                if (_stats.groups.isNotEmpty) ...[
                  const SectionHeader('Saved so far', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
                  VoxCard(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Column(
                      children: [
                        for (final g in _stats.groups)
                          ListTile(
                            leading: SpeakerAvatar(label: g.name, known: g.speakerId != null),
                            title: Text(g.name),
                            subtitle: Text('${g.count} clips · ${formatDuration(g.duration)} · ${formatBytes(g.bytes)}'),
                            trailing: IconButton(
                              tooltip: 'Delete these clips',
                              icon: const Icon(Icons.delete_outline_rounded),
                              onPressed: () => _deleteGroup(g),
                            ),
                          ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    onPressed: _deleteAll,
                    style: OutlinedButton.styleFrom(foregroundColor: t.colorScheme.error),
                    icon: const Icon(Icons.delete_forever_rounded),
                    label: const Text('Delete all clips'),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}
