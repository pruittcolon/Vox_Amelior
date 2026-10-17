import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../core/library/importer.dart';
import '../core/library/library.dart';
import '../core/models/catalog.dart';
import '../core/models/model_store.dart';
import '../core/render/styles.dart';
import '../core/samples.dart';
import 'art.dart';
import 'edit_text_screen.dart';
import 'reader_screen.dart';
import 'scope.dart';
import 'settings_screen.dart';
import 'theme.dart';

class LibraryScreen extends StatelessWidget {
  const LibraryScreen({super.key, required this.onOpenVoices, required this.onOpenWrite});

  final VoidCallback onOpenVoices;
  final VoidCallback onOpenWrite;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    return Scaffold(
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => showAddSheet(context, onOpenWrite: onOpenWrite),
        icon: const Icon(Icons.add_rounded),
        label: const Text('Add'),
      ),
      body: SafeArea(
        bottom: false,
        child: ListenableBuilder(
          listenable: app,
          builder: (context, _) {
            final docs = app.library.docs;
            final resume = docs.where((d) => d.progress > 0 && d.progress < 1).firstOrNull;
            return ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 104),
              children: [
                const _Masthead(),
                const SizedBox(height: 18),
                if (!app.hasAnyVoices || starterPacks.any((id) => app.models.status(id).state == PackState.downloading)) ...[
                  _GetVoicesCard(onMore: onOpenVoices),
                  const SizedBox(height: 20),
                ],
                if (resume != null) ...[
                  const SectionLabel('Continue listening'),
                  _ResumeCard(doc: resume),
                  const SizedBox(height: 22),
                ],
                if (docs.isEmpty)
                  _EmptyLibrary(onOpenWrite: onOpenWrite)
                else ...[
                  SectionLabel('Your library · ${docs.length}'),
                  for (final d in docs) ...[_DocCard(doc: d), const SizedBox(height: 10)],
                ],
              ],
            );
          },
        ),
      ),
    );
  }
}

class _Masthead extends StatelessWidget {
  const _Masthead();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hour = DateTime.now().hour;
    final greeting = hour < 5
        ? 'Late-night listening'
        : hour < 12
            ? 'Good morning'
            : hour < 18
                ? 'Good afternoon'
                : 'Good evening';
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(greeting, style: theme.textTheme.labelLarge?.copyWith(color: theme.colorScheme.secondary)),
              const SizedBox(height: 2),
              Text('Narrator Studio', style: theme.textTheme.headlineLarge),
            ],
          ),
        ),
        IconButton(
          tooltip: 'Settings',
          icon: const Icon(Icons.tune_rounded),
          onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const SettingsScreen())),
        ),
      ],
    );
  }
}

String? coverStyle(Doc doc) => doc.style ?? (doc.kind == 'pdf' || doc.kind == 'txt' ? 'audiobook' : null);

/// The episode you were listening to, big.
class _ResumeCard extends StatelessWidget {
  const _ResumeCard({required this.doc});
  final Doc doc;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = paletteFor(doc.title);
    final minutes = doc.words == 0 ? null : (doc.words / 150 * (1 - doc.progress)).ceil();
    return Material(
      borderRadius: BorderRadius.circular(24),
      clipBehavior: Clip.antiAlias,
      child: Ink(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color.lerp(p.a, Colors.black, 0.15)!, Color.lerp(p.a, p.b, 0.5)!],
          ),
        ),
        child: InkWell(
          onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc))),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(18),
                    boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.3), blurRadius: 16, offset: const Offset(0, 6))],
                  ),
                  child: CoverArt(seed: doc.title, style: coverStyle(doc), size: 92, radius: 18),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        doc.title,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.titleLarge?.copyWith(color: Colors.white),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        [
                          '${(doc.progress * 100).round()}% listened',
                          if (minutes != null) '~$minutes min left',
                        ].join(' · '),
                        style: theme.textTheme.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.82)),
                      ),
                      const SizedBox(height: 10),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: doc.progress,
                          color: p.glow,
                          backgroundColor: Colors.white.withValues(alpha: 0.2),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Container(
                  width: 48,
                  height: 48,
                  decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                  child: Icon(Icons.play_arrow_rounded, color: p.a, size: 30),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

Future<void> showAddSheet(BuildContext context, {required VoidCallback onOpenWrite}) {
  Widget item(BuildContext sheet, IconData icon, String seed, String title, String? subtitle, VoidCallback onTap) => ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
        leading: IconTile(icon: icon, seed: seed, size: 42),
        title: Text(title),
        subtitle: subtitle == null ? null : Text(subtitle),
        onTap: () {
          Navigator.pop(sheet);
          onTap();
        },
      );
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (sheet) => SafeArea(
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            item(sheet, Icons.file_open_rounded, 'open', 'Open a file', '.txt, .md or .pdf', () => openFile(context)),
            item(sheet, Icons.content_paste_rounded, 'paste', 'Paste a script or text', 'For example, Claude\'s reply',
                () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const EditTextScreen()))),
            item(sheet, Icons.auto_awesome_rounded, 'claude-write', 'Write a script with Claude',
                'Copy a ready-made prompt for any style', onOpenWrite),
            item(sheet, Icons.podcasts_rounded, 'Why Is the Sky Blue?', 'Sample: two-host podcast', null,
                () => addSample(context, podcast: true)),
            item(sheet, Icons.menu_book_rounded, 'The Library That Remembered Everything', 'Sample: audiobook chapter', null,
                () => addSample(context, podcast: false)),
          ],
        ),
      ),
    ),
  );
}

Future<void> openFile(BuildContext context) async {
  final app = context.app;
  final nav = Navigator.of(context);
  final picked = await FilePicker.pickFiles(type: FileType.custom, allowedExtensions: supportedExtensions);
  final file = picked.isEmpty ? null : picked.first;
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

/// The best-quality set, in the order they become useful.
const starterPacks = ['supertonic', 'kokoro'];

Future<void> installStarterPacks(BuildContext context) async {
  final models = context.app.models;
  for (final id in starterPacks) {
    if (!models.isInstalled(id)) await models.install(id);
    if (!models.isInstalled(id)) return; // failed or paused: stop here
  }
}

class _GetVoicesCard extends StatelessWidget {
  const _GetVoicesCard({required this.onMore});
  final VoidCallback onMore;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final missing = starterPacks.where((id) => !app.models.isInstalled(id)).toList();
    final total = missing.fold<int>(0, (a, id) => a + voicePacks[id]!.bytes);
    final active = starterPacks.where((id) => app.models.status(id).state == PackState.downloading).firstOrNull;
    final failed = starterPacks.where((id) => app.models.status(id).state == PackState.failed).firstOrNull;
    final status = app.models.status(active ?? failed ?? missing.firstOrNull ?? 'kokoro');
    final theme = Theme.of(context);
    const p = CoverPalette(Color(0xFF231E6B), Color(0xFF5B6CD6), Color(0xFFFFC28A));
    const onCard = Colors.white;
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [p.a, p.b]),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Equalizer(playing: true, color: amber, size: 22),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  active != null ? 'Downloading the studio voices' : 'First, download the voices',
                  style: theme.textTheme.titleLarge?.copyWith(color: onCard),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Narrator Studio speaks entirely on your phone: no account, no internet once downloaded. '
            'The studio set has the 10 crispest voices (Supertonic HD, full 44 kHz) and 54 more with accents '
            '(Kokoro): ${formatBytes(total)}. Use Wi-Fi if you can.',
            style: theme.textTheme.bodyMedium?.copyWith(color: onCard.withValues(alpha: 0.88)),
          ),
          const SizedBox(height: 16),
          if (active != null) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(value: status.progress, color: amber, backgroundColor: Colors.white24),
            ),
            const SizedBox(height: 8),
            Text(
              '${voicePacks[active]!.title} · ${formatBytes(status.received)} of ${formatBytes(status.total)}',
              style: theme.textTheme.bodySmall?.copyWith(color: onCard.withValues(alpha: 0.85)),
            ),
          ] else ...[
            if (status.state == PackState.failed)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Text(status.error ?? 'Download failed', style: const TextStyle(color: Color(0xFFFFB4A8))),
              ),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  style: FilledButton.styleFrom(backgroundColor: amber, foregroundColor: const Color(0xFF2A1A00)),
                  onPressed: () => installStarterPacks(context),
                  icon: const Icon(Icons.download_rounded),
                  label: Text(failed != null ? 'Resume download' : 'Download voices'),
                ),
                TextButton(
                  style: TextButton.styleFrom(foregroundColor: onCard),
                  onPressed: onMore,
                  child: const Text('More voice packs'),
                ),
              ],
            ),
          ],
        ],
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
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Column(
        children: [
          SizedBox(
            height: 120,
            width: 260,
            child: Stack(
              clipBehavior: Clip.none,
              alignment: Alignment.center,
              children: [
                Transform.translate(
                  offset: const Offset(-56, 10),
                  child: Transform.rotate(angle: -0.18, child: const CoverArt(seed: 'story', style: 'story', size: 84)),
                ),
                Transform.translate(
                  offset: const Offset(56, 10),
                  child: Transform.rotate(angle: 0.18, child: const CoverArt(seed: 'lecture', style: 'lecture', size: 84)),
                ),
                const CoverArt(seed: 'Why Is the Sky Blue?', style: 'podcast', size: 100),
              ],
            ),
          ),
          const SizedBox(height: 18),
          Text('Your library is empty', style: text.headlineSmall),
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
              FilledButton.icon(
                onPressed: () => addSample(context, podcast: true),
                icon: const Icon(Icons.podcasts_rounded),
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

  String get _kind => switch (doc.kind) {
        'pdf' => 'PDF',
        'md' => 'Notes',
        'shared' || 'pasted' => 'Script',
        'sample' => 'Sample',
        _ => 'Text',
      };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final minutes = doc.words == 0 ? null : (doc.words / 150).ceil();
    final style = doc.style == null ? null : styleById(doc.style).name;
    final details = [
      style ?? _kind,
      if (minutes != null) '~$minutes min',
      if (doc.progress > 0) '${(doc.progress * 100).round()}% listened',
    ].join(' · ');
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc))),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 0, 12),
          child: Row(
            children: [
              CoverArt(seed: doc.title, style: coverStyle(doc), size: 60, radius: 14),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(doc.title, maxLines: 2, overflow: TextOverflow.ellipsis, style: theme.textTheme.titleMedium),
                    const SizedBox(height: 3),
                    Text(details, style: theme.textTheme.bodySmall),
                    if (doc.progress > 0) ...[
                      const SizedBox(height: 8),
                      LinearProgressIndicator(value: doc.progress, borderRadius: BorderRadius.circular(4), minHeight: 3),
                    ],
                  ],
                ),
              ),
              PopupMenuButton<String>(
                tooltip: 'More',
                icon: const Icon(Icons.more_vert_rounded),
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
