import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/enroll_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

/// Enrolled people and voices Vox has heard but cannot name yet.
class PeopleScreen extends StatelessWidget {
  const PeopleScreen({super.key, required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('People')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _openEnroll(context),
        icon: const Icon(Icons.person_add),
        label: const Text('Add person'),
      ),
      body: ValueListenableBuilder<int>(
        valueListenable: services.dataVersion,
        builder: (context, _, _) {
          final people = services.speakers.profiles();
          final guests = services.speakers.clusters();
          return ListView(
            padding: const EdgeInsets.only(bottom: 96),
            children: [
              if (people.isEmpty)
                const ListTile(
                  leading: Icon(Icons.info_outline),
                  title: Text('No one added yet'),
                  subtitle: Text('Add each person in the household so transcripts show their name.'),
                ),
              for (final p in people) _personTile(context, p),
              if (guests.isNotEmpty) ...[
                const Padding(
                  padding: EdgeInsets.fromLTRB(16, 24, 16, 4),
                  child: Text('Voices Vox has heard', style: TextStyle(fontWeight: FontWeight.w600)),
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Text('Name a voice to label everything it has said.'),
                ),
                for (final g in guests) _guestTile(context, g),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _personTile(BuildContext context, SpeakerProfile p) {
    final color = speakerColor(p.name);
    return ListTile(
      leading: CircleAvatar(
        backgroundColor: color.withValues(alpha: 0.15),
        foregroundColor: color,
        child: Text(p.name.characters.first.toUpperCase()),
      ),
      title: Text(p.name),
      subtitle: Text('${p.sampleCount} voice samples'),
      trailing: PopupMenuButton<String>(
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
            case 'delete':
              if (await confirm(context, 'Delete ${p.name}?', 'Their voice profile is removed. Past transcripts stay, unlabelled.')) {
                services.speakers.delete(p.id);
                services.dataChanged();
              }
          }
        },
        itemBuilder: (_) => const [
          PopupMenuItem(value: 'more', child: Text('Add voice samples')),
          PopupMenuItem(value: 'rename', child: Text('Rename')),
          PopupMenuItem(value: 'delete', child: Text('Delete')),
        ],
      ),
    );
  }

  Widget _guestTile(BuildContext context, UnknownCluster g) => ListTile(
        leading: const CircleAvatar(child: Icon(Icons.question_mark)),
        title: Text(g.label),
        subtitle: Text('Heard ${g.count} times · last ${formatDay(g.updatedAt)} ${formatTime(g.updatedAt)}'),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextButton(onPressed: () => _nameGuest(context, g), child: const Text('Name')),
            IconButton(
              tooltip: 'Forget',
              icon: const Icon(Icons.close),
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
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: Text('Who is ${g.label}?', style: Theme.of(c).textTheme.titleMedium),
            ),
            for (final p in people)
              ListTile(leading: const Icon(Icons.person), title: Text(p.name), onTap: () => Navigator.pop(c, p.id)),
            ListTile(
              leading: const Icon(Icons.person_add),
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
        services.speakers.promoteCluster(g.id, name, embeddingModel: 'nemo-titanet-small');
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
      showMessage(context, 'Download the speech models first (Settings → Models).');
      return;
    }
    await Navigator.push(
      context,
      MaterialPageRoute<void>(builder: (_) => EnrollScreen(services: services, person: person)),
    );
  }
}
