import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/models/catalog.dart';
import '../core/models/model_store.dart';
import '../core/tts/synth.dart';
import '../core/tts/voices.dart';
import 'art.dart';
import 'clone_screen.dart';
import 'gallery_screen.dart';
import 'mix_screen.dart';
import 'scope.dart';
import 'voice_picker.dart';
import 'voice_set.dart';

class VoicesScreen extends StatelessWidget {
  const VoicesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: ListenableBuilder(
          listenable: Listenable.merge([app, context.scope.preview]),
          builder: (context, _) => ListView(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
            children: [
              Text('Voices', style: theme.textTheme.headlineLarge),
              const SizedBox(height: 4),
              Text(
                'Free, open models that run entirely on your phone. Download once, then everything works offline.',
                style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              ),
              const SizedBox(height: 18),
              const SectionLabel('Characters speak with'),
              const VoiceSetSwitch(),
              const SizedBox(height: 20),
              const SectionLabel('Voice packs'),
              for (final id in ['kokoro', 'supertonic', 'pocket', 'clarity']) ...[
                PackCard(packId: id),
                const SizedBox(height: 12),
              ],
              _CompactKokoro(app: app),
              const SizedBox(height: 12),
              const SectionLabel('Make it yours'),
              _ActionCard(
                icon: Icons.groups_rounded,
                seed: 'gallery',
                title: 'Voice Gallery',
                subtitle: 'Hundreds of real voices donated or freely licensed for speech (CC0 / CC-BY): LibriVox narrators, '
                    'voice actors, UK and US accents. Needs the cloning pack.',
                onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const GalleryScreen())),
              ),
              const SizedBox(height: 10),
              _ActionCard(
                icon: Icons.mic_rounded,
                seed: 'clone',
                title: 'Clone a voice',
                subtitle: 'Record 10 seconds of your own voice (or a voice you have permission to use) and narrate with it.',
                onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const CloneScreen())),
              ),
              const SizedBox(height: 10),
              _ActionCard(
                icon: Icons.tune_rounded,
                seed: 'mix',
                title: 'Mix voices',
                subtitle: 'Blend two Kokoro voices into a brand-new one, like 60% George + 40% Onyx.',
                onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const MixScreen())),
              ),
              if (app.settings.clones.isNotEmpty || app.settings.mixes.isNotEmpty || app.settings.gallery.isNotEmpty) ...[
                const SizedBox(height: 20),
                const SectionLabel('Your voices'),
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
              const SectionLabel('Characters'),
              Text(
                'Ready-made personas the styles cast automatically. Tap one to hear it.',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 10),
              LayoutBuilder(
                builder: (context, box) {
                  final columns = box.maxWidth > 560 ? 3 : 2;
                  final w = (box.maxWidth - 10 * (columns - 1)) / columns;
                  return Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    children: [
                      for (final c in app.characters) SizedBox(width: w, child: _CharacterCard(character: c)),
                    ],
                  );
                },
              ),
              const SizedBox(height: 16),
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

class _ActionCard extends StatelessWidget {
  const _ActionCard({required this.icon, required this.seed, required this.title, required this.subtitle, required this.onTap});
  final IconData icon;
  final String seed;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Card(
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(14, 14, 8, 14),
            child: Row(
              children: [
                IconTile(icon: icon, seed: seed),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title, style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: 2),
                      Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right_rounded),
              ],
            ),
          ),
        ),
      );
}

/// A character: portrait, voice, and a tap to hear it.
class _CharacterCard extends StatelessWidget {
  const _CharacterCard({required this.character});
  final Character character;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final preview = context.scope.preview;
    final theme = Theme.of(context);
    VoiceSpec? spec;
    try {
      spec = app.resolve('char:${character.id}');
    } catch (_) {}
    final voice = spec?.voice;
    final busy = voice != null && preview.busy == voice.id;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: voice == null
            ? null
            : () async {
                await preview.play(voice, speed: spec?.speed ?? 1.0);
                if (preview.error != null && context.mounted) context.toast(preview.error!);
              },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  VoiceAvatar(name: character.name, characterId: character.id, size: 40),
                  const Spacer(),
                  if (voice != null)
                    Icon(
                      busy ? Icons.stop_circle_rounded : Icons.play_circle_fill_rounded,
                      color: theme.colorScheme.primary,
                      size: 28,
                      semanticLabel: busy ? 'Stop' : 'Try',
                    ),
                ],
              ),
              const SizedBox(height: 10),
              Text(character.name, maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.titleSmall),
              const SizedBox(height: 2),
              Text(
                voice == null ? 'Download a voice pack' : '${voice.name} · ${voice.group}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: theme.textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ),
    );
  }
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
            Icon(Icons.check_circle_rounded, color: scheme.primary, size: 20),
            const SizedBox(width: 4),
            Text('Ready', style: text.labelLarge?.copyWith(color: scheme.primary)),
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
    final badges = switch (packId) {
      'supertonic' => const ['HD', '44 kHz', 'Best voices'],
      'kokoro' => const ['54 voices', 'Accents'],
      'clarity' => const ['48 kHz', 'For cloned voices'],
      'pocket' => const ['HD', 'Cloning'],
      _ => const <String>[],
    };
    final icon = switch (packId) {
      'supertonic' => Icons.graphic_eq_rounded,
      'kokoro' || 'kokoro-lite' => Icons.record_voice_over_rounded,
      'clarity' => Icons.auto_awesome_rounded,
      'pocket' => Icons.content_copy_rounded,
      _ => Icons.download_rounded,
    };
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 14, 8, 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                IconTile(icon: icon, seed: 'pack-$packId'),
                const SizedBox(width: 14),
                Expanded(child: Text(pack.title, style: text.titleMedium)),
                action,
              ],
            ),
            if (badges.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [for (final b in badges) _Badge(b)],
              ),
            ],
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Text(pack.summary, style: text.bodyMedium),
            ),
            const SizedBox(height: 6),
            Text('${formatBytes(pack.bytes)} · ${pack.license}', style: text.bodySmall?.copyWith(color: scheme.outline)),
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

class _Badge extends StatelessWidget {
  const _Badge(this.label);
  final String label;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(color: scheme.secondaryContainer, borderRadius: BorderRadius.circular(8)),
      child: Text(label, style: Theme.of(context).textTheme.labelSmall?.copyWith(color: scheme.onSecondaryContainer)),
    );
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
