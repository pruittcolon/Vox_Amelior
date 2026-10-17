import 'package:flutter/material.dart';

import '../core/library/settings.dart';
import '../core/models/catalog.dart';
import '../core/tts/voices.dart';
import 'scope.dart';
import 'voices_screen.dart';

/// Hundreds of freely licensed real voices that Pocket TTS can speak with.
class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key});

  @override
  State<GalleryScreen> createState() => _GalleryScreenState();
}

const _sets = {
  'all': 'All',
  'featured': 'Narrators & actors',
  'vctk': 'UK & US accents',
  'voice-donations': 'Donated voices',
};

class _GalleryScreenState extends State<GalleryScreen> {
  List<GalleryVoice>? _voices;
  Object? _error;
  String _set = 'all';
  String? _tone;
  String _query = '';
  String? _loading;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load({bool refresh = false}) async {
    setState(() => _error = null);
    try {
      final v = await context.app.models.gallery(refresh: refresh);
      if (mounted) setState(() => _voices = v);
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  bool _keep(GalleryVoice v) {
    final setOk = switch (_set) {
      'all' => true,
      'featured' => v.set == 'voice-zero' || v.set == 'alba-mackenna' || v.set == 'unmute-prod-website',
      _ => v.set == _set,
    };
    if (!setOk) return false;
    if (_tone != null && v.tone != _tone) return false;
    if (_query.isEmpty) return true;
    final q = _query.toLowerCase();
    return v.name.toLowerCase().contains(q) || v.description.toLowerCase().contains(q);
  }

  Voice _asVoice(GalleryVoice g, String path) => Voice(
        id: g.voiceId,
        name: g.name,
        engine: Engine.pocket,
        blurb: g.description,
        group: 'Gallery',
        refPath: path,
        license: g.license,
      );

  Future<void> _try(GalleryVoice g) async {
    final app = context.app;
    final preview = context.scope.preview;
    setState(() => _loading = g.id);
    try {
      final path = await app.models.fetchGalleryClip(g);
      if (!mounted) return;
      setState(() => _loading = null);
      await preview.play(_asVoice(g, path));
      if (preview.error != null && mounted) context.toast(preview.error!);
    } catch (e) {
      if (mounted) {
        setState(() => _loading = null);
        context.toast('Could not get this voice: $e');
      }
    }
  }

  Future<void> _add(GalleryVoice g) async {
    final app = context.app;
    try {
      await app.models.fetchGalleryClip(g);
      final name = g.name;
      await app.settings.update((s) {
        s.gallery.removeWhere((x) => x.id == g.id);
        s.gallery.add(SavedGalleryVoice(
          id: g.id,
          name: name,
          description: g.description,
          asset: g.asset,
          size: g.size,
          sha256: g.sha256,
          license: g.license,
        ));
      });
      if (mounted) context.toast('$name added to your voices. Pick it for any speaker.');
    } catch (e) {
      if (mounted) context.toast('Could not add it: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final scheme = Theme.of(context).colorScheme;
    final voices = _voices?.where(_keep).toList();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Voice Gallery'),
        actions: [IconButton(tooltip: 'Refresh', icon: const Icon(Icons.refresh), onPressed: () => _load(refresh: true))],
      ),
      body: ListenableBuilder(
        listenable: Listenable.merge([app, context.scope.preview]),
        builder: (context, _) {
          final pocketReady = app.models.isInstalled('pocket');
          return CustomScrollView(
            slivers: [
              if (!pocketReady)
                const SliverPadding(
                  padding: EdgeInsets.fromLTRB(16, 4, 16, 8),
                  sliver: SliverToBoxAdapter(child: PackCard(packId: 'pocket')),
                ),
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
                  child: Text(
                    'Each voice is a short clip from a public-domain (CC0) or Creative Commons (CC-BY) collection '
                    'made for speech technology. Pocket TTS speaks in that voice. Credit: kyutai/tts-voices, '
                    'VCTK (University of Edinburgh), LibriVox / Voice-Zero, Alba MacKenna.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ),
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                  child: TextField(
                    decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search: Scottish, female, deep…'),
                    onChanged: (v) => setState(() => _query = v),
                  ),
                ),
              ),
              SliverToBoxAdapter(
                child: SizedBox(
                  height: 52,
                  child: ListView(
                    scrollDirection: Axis.horizontal,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    children: [
                      for (final e in _sets.entries)
                        Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: ChoiceChip(label: Text(e.value), selected: _set == e.key, onSelected: (_) => setState(() => _set = e.key)),
                        ),
                      for (final t in ['deep', 'low', 'medium', 'high'])
                        Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: FilterChip(
                            label: Text('$t voice'),
                            selected: _tone == t,
                            onSelected: (on) => setState(() => _tone = on ? t : null),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
              if (_error != null)
                SliverFillRemaining(
                  hasScrollBody: false,
                  child: Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text('Could not load the gallery: $_error', textAlign: TextAlign.center),
                          const SizedBox(height: 12),
                          FilledButton(onPressed: _load, child: const Text('Try again')),
                        ],
                      ),
                    ),
                  ),
                )
              else if (voices == null)
                const SliverFillRemaining(child: Center(child: CircularProgressIndicator()))
              else
                SliverList.builder(
                  itemCount: voices.length,
                  itemBuilder: (context, i) {
                    final g = voices[i];
                    final busy = context.scope.preview.busy == g.voiceId;
                    final added = app.settings.gallery.any((x) => x.id == g.id);
                    return ListTile(
                      leading: _loading == g.id
                          ? const SizedBox(width: 40, child: Center(child: SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2))))
                          : IconButton(
                              tooltip: busy ? 'Stop' : 'Try',
                              onPressed: pocketReady ? () => _try(g) : null,
                              icon: Icon(busy ? Icons.stop_circle_outlined : Icons.play_circle_outline),
                            ),
                      title: Text(g.name),
                      subtitle: Text('${g.description} · ${g.license}'),
                      trailing: added
                          ? Icon(Icons.check_circle, color: scheme.primary)
                          : IconButton(tooltip: 'Add to my voices', icon: const Icon(Icons.add_circle_outline), onPressed: () => _add(g)),
                    );
                  },
                ),
              const SliverToBoxAdapter(child: SizedBox(height: 32)),
            ],
          );
        },
      ),
    );
  }
}
