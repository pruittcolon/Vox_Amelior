import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../core/library/importer.dart';
import '../core/library/library.dart';
import '../core/models/catalog.dart';
import '../core/models/model_store.dart';
import '../core/render/styles.dart';
import '../core/samples.dart';
import 'edit_text_screen.dart';
import 'reader_screen.dart';
import 'scope.dart';
import 'settings_screen.dart';

class LibraryScreen extends StatelessWidget {
  const LibraryScreen({super.key, required this.onOpenVoices, required this.onOpenWrite});

  final VoidCallback onOpenVoices;
  final VoidCallback onOpenWrite;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Narrator Studio'),
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => showAddSheet(context, onOpenWrite: onOpenWrite),
        icon: const Icon(Icons.add),
        label: const Text('Add'),
      ),
      body: ListenableBuilder(
        listenable: app,
        builder: (context, _) {
          final docs = app.library.docs;
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 96),
            children: [
              if (!app.hasAnyVoices) ...[
                _GetVoicesCard(onMore: onOpenVoices),
                const SizedBox(height: 16),
              ],
              if (docs.isEmpty) _EmptyLibrary(onOpenWrite: onOpenWrite) else ...[
                for (final d in docs) ...[_DocCard(doc: d), const SizedBox(height: 10)],
              ],
            ],
          );
        },
      ),
    );
  }
}

Future<void> showAddSheet(BuildContext context, {required VoidCallback onOpenWrite}) {
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (sheet) => SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            leading: const Icon(Icons.file_open_outlined),
            title: const Text('Open a file'),
            subtitle: const Text('.txt, .md or .pdf'),
            onTap: () {
              Navigator.pop(sheet);
              openFile(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.content_paste),
            title: const Text('Paste a script or text'),
            subtitle: const Text('For example, Claude\'s reply'),
            onTap: () {
              Navigator.pop(sheet);
              Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const EditTextScreen()));
            },
          ),
          ListTile(
            leading: const Icon(Icons.auto_awesome_outlined),
            title: const Text('Write a script with Claude'),
            subtitle: const Text('Copy a ready-made prompt for any style'),
            onTap: () {
              Navigator.pop(sheet);
              onOpenWrite();
            },
          ),
          ListTile(
            leading: const Icon(Icons.podcasts),
            title: const Text('Sample: two-host podcast'),
            onTap: () {
              Navigator.pop(sheet);
              addSample(context, podcast: true);
            },
          ),
          ListTile(
            leading: const Icon(Icons.menu_book_outlined),
            title: const Text('Sample: audiobook chapter'),
            onTap: () {
              Navigator.pop(sheet);
              addSample(context, podcast: false);
            },
          ),
        ],
      ),
    ),
  );
}

Future<void> openFile(BuildContext context) async {
  final app = context.app;
  final nav = Navigator.of(context);
  final result = await FilePicker.platform.pickFiles(type: FileType.custom, allowedExtensions: supportedExtensions);
  final file = result?.files.single;
  if (file == null || file.path == null) return;
  if (!context.mounted) return;
  final messenger = ScaffoldMessenger.of(context);
  final loading = messenger.showSnackBar(
    SnackBar(content: Text('Reading ${file.name}…'), duration: const Duration(minutes: 2)),
  );
  try {
    final imported = await importFile(file.path!, displayName: file.name);
    final doc = await app.library.add(title: imported.title, kind: imported.kind, text: imported.text);
    loading.close();
    await nav.push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc)));
  } catch (e) {
    loading.close();
    messenger.showSnackBar(SnackBar(content: Text(e.toString())));
  }
}

Future<void> addSample(BuildContext context, {required bool podcast}) async {
  final nav = Navigator.of(context);
  final doc = await context.app.library.add(
    title: podcast ? 'Why Is the Sky Blue?' : 'The Library That Remembered Everything',
    kind: 'sample',
    text: podcast ? samplePodcast : sampleAudiobook,
  );
  await nav.push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc)));
}

class _GetVoicesCard extends StatelessWidget {
  const _GetVoicesCard({required this.onMore});
  final VoidCallback onMore;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final pack = voicePacks['kokoro']!;
    final status = app.models.status('kokoro');
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.record_voice_over, color: scheme.onPrimaryContainer),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'First, download the voices',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(color: scheme.onPrimaryContainer),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Narrator Studio speaks entirely on your phone: no account, no internet once downloaded. '
              'The Kokoro pack has 54 natural voices (${formatBytes(pack.bytes)}). Use Wi-Fi if you can.',
              style: TextStyle(color: scheme.onPrimaryContainer),
            ),
            const SizedBox(height: 12),
            if (status.state == PackState.downloading) ...[
              LinearProgressIndicator(value: status.progress),
              const SizedBox(height: 6),
              Text(
                '${formatBytes(status.received)} of ${formatBytes(status.total)}',
                style: TextStyle(color: scheme.onPrimaryContainer),
              ),
            ] else ...[
              if (status.state == PackState.failed)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(status.error ?? 'Download failed', style: TextStyle(color: scheme.error)),
                ),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  FilledButton.icon(
                    onPressed: () => app.models.install('kokoro'),
                    icon: const Icon(Icons.download),
                    label: Text(status.state == PackState.failed ? 'Resume download' : 'Download voices'),
                  ),
                  TextButton(onPressed: onMore, child: const Text('More voice packs')),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _EmptyLibrary extends StatelessWidget {
  const _EmptyLibrary({required this.onOpenWrite});
  final VoidCallback onOpenWrite;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          Icon(Icons.auto_stories_outlined, size: 56, color: Theme.of(context).colorScheme.primary),
          const SizedBox(height: 12),
          Text('Your library is empty', style: text.titleLarge),
          const SizedBox(height: 6),
          Text(
            'Open a book or article, paste a script from Claude, or start with a sample.',
            textAlign: TextAlign.center,
            style: text.bodyMedium,
          ),
          const SizedBox(height: 20),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children: [
              FilledButton.tonalIcon(
                onPressed: () => addSample(context, podcast: true),
                icon: const Icon(Icons.podcasts),
                label: const Text('Sample podcast'),
              ),
              OutlinedButton.icon(
                onPressed: () => openFile(context),
                icon: const Icon(Icons.file_open_outlined),
                label: const Text('Open a file'),
              ),
              OutlinedButton.icon(
                onPressed: onOpenWrite,
                icon: const Icon(Icons.auto_awesome_outlined),
                label: const Text('Write with Claude'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _DocCard extends StatelessWidget {
  const _DocCard({required this.doc});
  final Doc doc;

  IconData get _icon => switch (doc.kind) {
        'pdf' => Icons.picture_as_pdf_outlined,
        'md' => Icons.description_outlined,
        'shared' || 'pasted' => Icons.auto_awesome_outlined,
        'sample' => Icons.star_outline,
        _ => Icons.article_outlined,
      };

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final minutes = doc.words == 0 ? null : (doc.words / 150).ceil();
    final style = doc.style == null ? null : styleById(doc.style).name;
    final details = [
      ?style,
      if (minutes != null) '~$minutes min',
      if (doc.progress > 0) '${(doc.progress * 100).round()}% listened',
    ].join(' · ');
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc))),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 4, 12),
          child: Row(
            children: [
              CircleAvatar(backgroundColor: scheme.secondaryContainer, child: Icon(_icon, color: scheme.onSecondaryContainer)),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(doc.title, maxLines: 2, overflow: TextOverflow.ellipsis, style: Theme.of(context).textTheme.titleMedium),
                    if (details.isNotEmpty) ...[
                      const SizedBox(height: 2),
                      Text(details, style: Theme.of(context).textTheme.bodySmall),
                    ],
                    if (doc.progress > 0) ...[
                      const SizedBox(height: 8),
                      LinearProgressIndicator(value: doc.progress, borderRadius: BorderRadius.circular(4)),
                    ],
                  ],
                ),
              ),
              PopupMenuButton<String>(
                tooltip: 'More',
                onSelected: (v) => _menu(context, v),
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'rename', child: Text('Rename')),
                  PopupMenuItem(value: 'edit', child: Text('Edit text')),
                  PopupMenuItem(value: 'restart', child: Text('Start from the beginning')),
                  PopupMenuItem(value: 'delete', child: Text('Delete')),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _menu(BuildContext context, String action) async {
    final lib = context.app.library;
    switch (action) {
      case 'rename':
        final name = await promptText(context, title: 'Rename', initial: doc.title);
        if (name != null && name.trim().isNotEmpty) {
          doc.title = name.trim();
          await lib.changed(doc);
        }
      case 'edit':
        await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => EditTextScreen(doc: doc)));
      case 'restart':
        doc.position = 0;
        await lib.changed(doc);
      case 'delete':
        final player = context.scope.player;
        final ok = await showDialog<bool>(
          context: context,
          builder: (d) => AlertDialog(
            title: Text('Delete "${doc.title}"?'),
            content: const Text('This removes it from your library. Exported audio files are kept.'),
            actions: [
              TextButton(onPressed: () => Navigator.pop(d, false), child: const Text('Cancel')),
              FilledButton(onPressed: () => Navigator.pop(d, true), child: const Text('Delete')),
            ],
          ),
        );
        if (ok == true) {
          if (player.session?.doc.id == doc.id) await player.stop();
          await lib.remove(doc);
        }
    }
  }
}

Future<String?> promptText(BuildContext context, {required String title, String initial = '', String? hint}) {
  final controller = TextEditingController(text: initial);
  return showDialog<String>(
    context: context,
    builder: (d) => AlertDialog(
      title: Text(title),
      content: TextField(
        controller: controller,
        autofocus: true,
        decoration: InputDecoration(hintText: hint),
        onSubmitted: (v) => Navigator.pop(d, v),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(d), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(d, controller.text), child: const Text('Save')),
      ],
    ),
  );
}
