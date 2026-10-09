import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/insights_repository.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/enroll_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/insights_screen.dart';
import 'package:vox_amelior_mobile/ui/more_screen.dart';
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
          // Voices marked as TV / background last: they are rarely worth naming.
          final guests = services.speakers.clusters()..sort((a, b) => (a.background ? 1 : 0) - (b.background ? 1 : 0));
          final now = DateTime.now();
          final month = {
            for (final p in services.insights.people(InsightsScope(from: DateTime(now.year, now.month, now.day - 29)))) p.id: p,
          };
          final heard = services.insights.lastHeard();
          if (people.isEmpty && guests.isEmpty) {
            return const EmptyState(
              icon: Icons.people_alt_rounded,
              title: 'Teach Vox who is who',
              message: 'Add each person with a few voice samples (or WAV recordings). Transcripts then show their names.',
            );
          }
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 100),
            children: [
              if (people.isNotEmpty) const SectionHeader('Household', padding: EdgeInsets.fromLTRB(4, 8, 4, 8)),
              for (final p in people)
                Padding(padding: const EdgeInsets.only(bottom: 8), child: _personCard(context, p, month[p.id], heard[p.id])),
              if (guests.isNotEmpty) ...[
                const SectionHeader('Voices Vox has heard', padding: EdgeInsets.fromLTRB(4, 16, 4, 4)),
                Padding(
                  padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
                  child: Text('Name a voice to label everything it has said.', style: Theme.of(context).textTheme.bodyMedium),
                ),
                for (final g in guests)
                  Padding(padding: const EdgeInsets.only(bottom: 8), child: _guestCard(context, g)),
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

  Widget _guestCard(BuildContext context, UnknownCluster g) => VoxCard(
        padding: const EdgeInsets.fromLTRB(16, 10, 4, 10),
        child: Row(
          children: [
            SpeakerAvatar(label: g.label, known: false, radius: 22),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(g.label, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                  Text('Heard ${g.count} times · last ${formatWhen(g.updatedAt)}'),
                  if (g.background) ...[
                    const SizedBox(height: 4),
                    const Pill('TV / background', icon: Icons.tv_rounded),
                  ],
                ],
              ),
            ),
            FilledButton.tonal(onPressed: () => _nameGuest(context, g), child: const Text('Name')),
            IconButton(
              tooltip: 'Forget',
              icon: const Icon(Icons.close_rounded),
              onPressed: () {
                services.speakers.deleteCluster(g.id);
                services.dataChanged();
              },
            ),
          ],
        ),
      );

  Future<void> _nameGuest(BuildContext context, UnknownCluster g) async {
    final people = services.speakers.profiles();
    final choice = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (c) => SafeArea(
        child: ListView(
          shrinkWrap: true,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              child: Text('Who is ${g.label}?', style: Theme.of(c).textTheme.titleMedium),
            ),
            for (final p in people)
              ListTile(leading: SpeakerAvatar(label: p.name), title: Text(p.name), onTap: () => Navigator.pop(c, p.id)),
            ListTile(
              leading: const Icon(Icons.person_add_rounded),
              title: const Text('Someone new…'),
              onTap: () => Navigator.pop(c, '#new'),
            ),
          ],
        ),
      ),
    );
    if (choice == null || !context.mounted) return;
    try {
      if (choice == '#new') {
        final name = await askText(context, 'Name this voice', hint: 'e.g. Grandma');
        if (name == null || name.isEmpty) return;
        services.speakers.promoteCluster(g.id, name, embeddingModel: SherpaSpeakerEmbedder.modelIdConst);
      } else {
        services.speakers.assignClusterToSpeaker(g.id, choice);
      }
      services.dataChanged();
    } on StateError catch (e) {
      if (context.mounted) showMessage(context, e.message);
    }
  }

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
