import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/insights_repository.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/enroll_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/insights_screen.dart';
import 'package:vox_amelior_mobile/ui/more_screen.dart';
import 'package:vox_amelior_mobile/ui/name_voice.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Enrolled people and voices Vox has heard but cannot name yet.
class PeopleScreen extends StatelessWidget {
  const PeopleScreen({super.key, required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('People'), actions: [SettingsButton(builder: (_) => MoreScreen(services: services))]),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _openEnroll(context),
        icon: const Icon(Icons.person_add_rounded),
        label: const Text('Add person'),
      ),
      body: ValueListenableBuilder<int>(
        valueListenable: services.dataVersion,
        builder: (context, _, _) {
          final people = services.speakers.profiles();
          final voices = services.transcripts.voicesToName(detailed: 30, samples: 1);
          final background = services.speakers.clusters().where((g) => g.background).toList();
          final now = DateTime.now();
          final month = {
            for (final p in services.insights.people(InsightsScope(from: DateTime(now.year, now.month, now.day - 29)))) p.id: p,
          };
          final heard = services.insights.lastHeard();
          if (people.isEmpty && voices.isEmpty && background.isEmpty) {
            return const EmptyState(
              icon: Icons.people_alt_rounded,
              title: 'Teach Vox who is who',
              message: 'Add each person with a few voice samples (or WAV recordings). Transcripts then show their names.',
            );
          }
          return ListView(
            key: const ValueKey('people-list'),
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 100),
            children: [
              VoicesToNameCard(
                count: voices.length,
                onOpen: () => openNameVoices(context, services),
                margin: const EdgeInsets.only(top: 8, bottom: 4),
              ),
              if (people.isNotEmpty) const SectionHeader('Household', padding: EdgeInsets.fromLTRB(4, 8, 4, 8)),
              for (final p in people)
                Padding(padding: const EdgeInsets.only(bottom: 8), child: _personCard(context, p, month[p.id], heard[p.id])),
              if (voices.isNotEmpty) ...[
                const SectionHeader('Voices without a name', padding: EdgeInsets.fromLTRB(4, 16, 4, 4)),
                Padding(
                  padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
                  child: Text('Name a voice to label everything it has said.', style: Theme.of(context).textTheme.bodyMedium),
                ),
                for (final v in voices)
                  Padding(padding: const EdgeInsets.only(bottom: 8), child: _voiceCard(context, v)),
              ],
              if (background.isNotEmpty) ...[
                const SectionHeader('TV and background voices', padding: EdgeInsets.fromLTRB(4, 16, 4, 4)),
                for (final g in background)
                  Padding(padding: const EdgeInsets.only(bottom: 8), child: _backgroundCard(context, g)),
              ],
            ],
          );
        },
      ),
    );
  }

  /// One person: tap for their statistics and conversations.
  Widget _personCard(BuildContext context, SpeakerProfile p, PersonStats? month, DateTime? lastHeard) => VoxCard(
        onTap: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => InsightsScreen(services: services, personId: p.id))),
        padding: const EdgeInsets.fromLTRB(16, 10, 4, 10),
        child: Row(
          children: [
            SpeakerAvatar(label: p.name, radius: 22),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(p.name, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                  Text(
                    lastHeard == null
                        ? 'Not heard yet · ${p.sampleCount} voice samples'
                        : 'Last heard ${formatWhen(lastHeard)}'
                            '${month == null ? '' : ' · ${formatTalk(month.talk)} this month'}',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  if (month != null && month.moods.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    MoodStrip(counts: month.moods),
                  ],
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: [
                      Pill(
                        p.patterns.isEmpty ? 'Learning' : '${p.patterns.length} voice pattern${p.patterns.length == 1 ? '' : 's'}',
                        icon: Icons.graphic_eq_rounded,
                      ),
                      if (p.negatives.isNotEmpty)
                        Pill('${p.negatives.length} "not ${p.name}"', icon: Icons.person_off_rounded, color: Theme.of(context).colorScheme.error),
                    ],
                  ),
                ],
              ),
            ),
            PopupMenuButton<String>(
              onSelected: (v) async {
                switch (v) {
                  case 'more':
                    await _openEnroll(context, person: p);
                  case 'rename':
                    final name = await askText(context, 'Rename', initial: p.name);
                    if (name == null || name.isEmpty) return;
                    try {
                      services.speakers.rename(p.id, name);
                      services.dataChanged();
                    } on StateError catch (e) {
                      if (context.mounted) showMessage(context, e.message);
                    }
                  case 'clearNot':
                    if (await confirm(
                      context,
                      'Clear "not ${p.name}" examples?',
                      'Voices you marked as "not ${p.name}" can be called ${p.name} again.',
                      action: 'Clear',
                    )) {
                      services.speakers.clearNegatives(p.id);
                      services.dataChanged();
                    }
                  case 'delete':
                    if (await confirm(context, 'Delete ${p.name}?', 'Their voice profile is removed. Past transcripts stay, unlabelled.')) {
                      services.speakers.delete(p.id);
                      services.dataChanged();
                    }
                }
              },
              itemBuilder: (_) => [
                const PopupMenuItem(value: 'more', child: Text('Add voice samples')),
                const PopupMenuItem(value: 'rename', child: Text('Rename')),
                if (p.negatives.isNotEmpty) PopupMenuItem(value: 'clearNot', child: Text('Clear "not ${p.name}" examples')),
                const PopupMenuItem(value: 'delete', child: Text('Delete')),
              ],
            ),
          ],
        ),
      );

  /// A voice without a name: what it said, and a button to say who it is.
  Widget _voiceCard(BuildContext context, VoiceToName v) {
    final t = Theme.of(context);
    final sample = v.samples.firstOrNull;
    return VoxCard(
      onTap: () => nameVoice(context, services, v.clusterId),
      padding: const EdgeInsets.fromLTRB(16, 12, 12, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SpeakerAvatar(label: v.label, known: false, radius: 22),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(v.label, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                    Text(
                      '${v.lines == 1 ? '1 line' : '${v.lines} lines'} · last heard ${formatWhen(v.lastHeard)}'
                      '${v.heardWith.isEmpty ? '' : ' · with ${v.heardWith.join(', ')}'}',
                    ),
                    if (sample != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(
                          '"${sample.text}"',
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: t.textTheme.bodyMedium?.copyWith(fontStyle: FontStyle.italic, color: t.colorScheme.onSurfaceVariant),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(
                onPressed: () async {
                  if (await confirm(
                    context,
                    'Forget ${v.label}?',
                    'Vox stops grouping lines under this voice. Its lines stay, without a name.',
                    action: 'Forget',
                  )) {
                    services.speakers.deleteCluster(v.clusterId);
                    services.dataChanged();
                  }
                },
                child: const Text('Forget'),
              ),
              const SizedBox(width: 4),
              FilledButton.tonal(onPressed: () => nameVoice(context, services, v.clusterId), child: const Text("Who's this?")),
            ],
          ),
        ],
      ),
    );
  }

  /// A voice hidden as TV / background, with the way back.
  Widget _backgroundCard(BuildContext context, UnknownCluster g) => VoxCard(
        padding: const EdgeInsets.fromLTRB(16, 10, 8, 10),
        child: Row(
          children: [
            SpeakerAvatar(label: g.label, known: false, radius: 18),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(g.label, style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
                  Text('Hidden · last heard ${formatWhen(g.updatedAt)}'),
                ],
              ),
            ),
            TextButton(
              onPressed: () {
                services.speakers.setClusterBackground(g.id, false);
                services.dataChanged();
              },
              child: const Text('Show again'),
            ),
          ],
        ),
      );

  Future<void> _openEnroll(BuildContext context, {SpeakerProfile? person}) async {
    if (!services.speechReady) {
      showMessage(context, 'Download the speech models first (More → Models).');
      return;
    }
    await Navigator.push(
      context,
      MaterialPageRoute<void>(builder: (_) => EnrollScreen(services: services, person: person)),
    );
  }
}
