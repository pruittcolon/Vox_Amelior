import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/models/catalog.dart';
import '../core/models/model_store.dart';
import '../core/tts/synth.dart';
import 'clone_screen.dart';
import 'gallery_screen.dart';
import 'mix_screen.dart';
import 'scope.dart';
import 'voice_picker.dart';

class VoicesScreen extends StatelessWidget {
  const VoicesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    return Scaffold(
      appBar: AppBar(title: const Text('Voices')),
      body: ListenableBuilder(
        listenable: Listenable.merge([app, context.scope.preview]),
        builder: (context, _) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
          children: [
            const _Heading('Voice packs', 'Free, open models. Download once; everything then runs offline on your phone.'),
            for (final id in ['kokoro', 'supertonic', 'pocket']) ...[
              PackCard(packId: id),
              const SizedBox(height: 10),
            ],
            _CompactKokoro(app: app),
            const SizedBox(height: 20),
            const _Heading('Make it yours', 'More voices than any one pack: borrow, clone or blend.'),
            _ActionCard(
              icon: Icons.groups_outlined,
              title: 'Voice Gallery',
              subtitle: 'Hundreds of real voices donated or freely licensed for speech (CC0 / CC-BY): LibriVox narrators, '
                  'voice actors, UK and US accents. Needs the cloning pack.',
              onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const GalleryScreen())),
            ),
            const SizedBox(height: 10),
            _ActionCard(
              icon: Icons.mic_none,
              title: 'Clone a voice',
              subtitle: 'Record 10 seconds of your own voice (or a voice you have permission to use) and narrate with it.',
              onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const CloneScreen())),
            ),
            const SizedBox(height: 10),
            _ActionCard(
              icon: Icons.tune,
              title: 'Mix voices',
              subtitle: 'Blend two Kokoro voices into a brand-new one, like 60% George + 40% Onyx.',
              onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const MixScreen())),
            ),
            if (app.settings.clones.isNotEmpty || app.settings.mixes.isNotEmpty || app.settings.gallery.isNotEmpty) ...[
              const SizedBox(height: 20),
              const _Heading('Your voices', ''),
              for (final v in app.allVoices.where((v) => v.group == 'Cloned' || v.group == 'Mixes' || v.group == 'Gallery'))
                VoiceTile(
                  title: v.name,
                  subtitle: app.isAvailable(v) ? '${v.group} · ${v.blurb}' : '${v.group} · needs the ${v.engine.name} pack',
                  voice: app.isAvailable(v) ? v : null,
                  onPick: null,
                  trailing: IconButton(
                    tooltip: 'Remove',
                    icon: const Icon(Icons.delete_outline),
                    onPressed: () => _removeVoice(context, app, v.id),
                  ),
                ),
            ],
            const SizedBox(height: 20),
            const _Heading('Characters', 'Ready-made personas used by the styles. Tap ▶ to hear one.'),
            for (final c in app.characters)
              Builder(builder: (context) {
                VoiceSpec? spec;
                try {
                  spec = app.resolve('char:${c.id}');
                } catch (_) {}
                return VoiceTile(title: c.name, subtitle: c.blurb, voice: spec?.voice, speed: spec?.speed ?? 1.0, onPick: null);
              }),
            const SizedBox(height: 12),
            Center(
              child: OutlinedButton.icon(
                onPressed: () => pickVoice(context, title: 'All voices'),
                icon: const Icon(Icons.library_music_outlined),
                label: const Text('Browse all voices'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _removeVoice(BuildContext context, AppState app, String id) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (d) => AlertDialog(
        title: const Text('Remove this voice?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(d, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(d, true), child: const Text('Remove')),
        ],
      ),
    );
    if (ok != true) return;
    if (id.startsWith('mix:')) {
      await app.settings.update((s) => s.mixes.removeWhere((m) => m.voiceId == id));
      await app.rebuildMixes();
    } else if (id.startsWith('clone:')) {
      await app.settings.update((s) => s.clones.removeWhere((c) => c.voiceId == id));
    } else if (id.startsWith('gallery:')) {
      await app.settings.update((s) => s.gallery.removeWhere((g) => g.voiceId == id));
    }
  }
}

class _Heading extends StatelessWidget {
  const _Heading(this.title, this.subtitle);
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleLarge),
            if (subtitle.isNotEmpty) Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      );
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({required this.icon, required this.title, required this.subtitle, required this.onTap});
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Card(
        clipBehavior: Clip.antiAlias,
        child: ListTile(
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          leading: Icon(icon, color: Theme.of(context).colorScheme.primary),
          title: Text(title),
          subtitle: Text(subtitle),
          trailing: const Icon(Icons.chevron_right),
          onTap: onTap,
        ),
      );
}

/// A voice pack with its download / progress / delete button.
class PackCard extends StatelessWidget {
  const PackCard({super.key, required this.packId});
  final String packId;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final pack = voicePacks[packId]!;
    final st = app.models.status(packId);
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    Widget action;
    switch (st.state) {
      case PackState.installed:
        action = Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.check_circle, color: scheme.primary, size: 20),
            const SizedBox(width: 4),
            Text('Ready', style: TextStyle(color: scheme.primary)),
            IconButton(
              tooltip: 'Delete to free space',
              icon: const Icon(Icons.delete_outline),
              onPressed: () => _confirmDelete(context, pack),
            ),
          ],
        );
      case PackState.downloading:
        action = TextButton(onPressed: () => app.models.cancel(packId), child: const Text('Pause'));
      case PackState.failed:
      case PackState.missing:
        action = FilledButton.tonalIcon(
          onPressed: () => app.models.install(packId),
          icon: const Icon(Icons.download, size: 18),
          label: Text(st.received > 0 || st.state == PackState.failed ? 'Resume' : formatBytes(pack.bytes)),
        );
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(child: Text(pack.title, style: text.titleMedium)),
                action,
              ],
            ),
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Text(pack.summary, style: text.bodyMedium),
            ),
            const SizedBox(height: 4),
            Text('License: ${pack.license}', style: text.bodySmall?.copyWith(color: scheme.outline)),
            if (st.state == PackState.downloading) ...[
              const SizedBox(height: 10),
              LinearProgressIndicator(value: st.progress, borderRadius: BorderRadius.circular(4)),
              const SizedBox(height: 4),
              Text('${formatBytes(st.received)} of ${formatBytes(st.total)} · ${st.file ?? ''}', style: text.bodySmall),
            ],
            if (st.state == PackState.failed && st.error != null) ...[
              const SizedBox(height: 6),
              Text(st.error!, style: text.bodySmall?.copyWith(color: scheme.error)),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _confirmDelete(BuildContext context, VoicePack pack) async {
    final app = context.app;
    final player = context.scope.player;
    final ok = await showDialog<bool>(
      context: context,
      builder: (d) => AlertDialog(
        title: Text('Delete ${pack.title}?'),
        content: Text('Frees ${formatBytes(app.models.diskBytes(pack.id))}. You can download it again any time.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(d, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(d, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;
    await player.stop();
    await app.synth.unload();
    await app.models.delete(pack.id);
  }
}

class _CompactKokoro extends StatelessWidget {
  const _CompactKokoro({required this.app});
  final AppState app;

  @override
  Widget build(BuildContext context) {
    final lite = app.models.status('kokoro-lite');
    final full = app.models.isInstalled('kokoro');
    if (full && lite.state != PackState.installed) return const SizedBox.shrink();
    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: Text('Short on space?', style: Theme.of(context).textTheme.titleSmall),
      children: const [PackCard(packId: 'kokoro-lite'), SizedBox(height: 8)],
    );
  }
}
