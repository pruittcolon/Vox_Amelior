import 'package:flutter/material.dart';

import '../core/tts/synth.dart';
import '../core/tts/voices.dart';
import 'scope.dart';

/// Pick a character or a voice; returns `char:<id>` or a voice id.
Future<String?> pickVoice(BuildContext context, {required String title, String? current}) {
  return Navigator.of(context).push<String>(
    MaterialPageRoute(builder: (_) => VoicePickerScreen(title: title, current: current)),
  );
}

const voiceGroups = ['All', 'American', 'British', 'Accented', 'Fast', 'Gallery', 'Cloned', 'Mixes'];

class VoicePickerScreen extends StatefulWidget {
  const VoicePickerScreen({super.key, required this.title, this.current});
  final String title;
  final String? current;

  @override
  State<VoicePickerScreen> createState() => _VoicePickerScreenState();
}

class _VoicePickerScreenState extends State<VoicePickerScreen> {
  String _group = 'All';
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final q = _query.toLowerCase();
    bool match(String a, String b) => q.isEmpty || a.toLowerCase().contains(q) || b.toLowerCase().contains(q);
    final characters = app.characters.where((c) => match(c.name, c.blurb)).toList();
    final voices = app.allVoices
        .where((v) => (_group == 'All' || v.group == _group) && match(v.name, v.blurb))
        .toList();
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: ListenableBuilder(
        listenable: context.scope.preview,
        builder: (context, _) => CustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
                child: TextField(
                  decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search voices'),
                  onChanged: (v) => setState(() => _query = v),
                ),
              ),
            ),
            if (characters.isNotEmpty && _group == 'All') ...[
              const _SectionTitle('Characters'),
              SliverList.builder(
                itemCount: characters.length,
                itemBuilder: (context, i) {
                  final c = characters[i];
                  final ref = 'char:${c.id}';
                  final voice = app.voiceById(c.voiceId);
                  final ready = voice != null && app.isAvailable(voice);
                  VoiceSpec? spec;
                  try {
                    spec = app.resolve(ref);
                  } catch (_) {}
                  return VoiceTile(
                    title: c.name,
                    subtitle: ready ? c.blurb : '${c.blurb} (not downloaded: uses a stand-in voice)',
                    selected: widget.current == ref,
                    voice: spec?.voice,
                    speed: spec?.speed ?? 1.0,
                    onPick: () => Navigator.pop(context, ref),
                  );
                },
              ),
            ],
            SliverToBoxAdapter(
              child: SizedBox(
                height: 52,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  children: [
                    for (final g in voiceGroups)
                      Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: ChoiceChip(label: Text(g), selected: _group == g, onSelected: (_) => setState(() => _group = g)),
                      ),
                  ],
                ),
              ),
            ),
            const _SectionTitle('Voices'),
            SliverList.builder(
              itemCount: voices.length,
              itemBuilder: (context, i) {
                final v = voices[i];
                final ready = app.isAvailable(v);
                return VoiceTile(
                  title: v.name,
                  subtitle: ready ? '${v.group} · ${v.blurb}' : '${v.group} · not downloaded yet (Voices tab)',
                  selected: widget.current == v.id,
                  enabled: ready,
                  voice: ready ? v : null,
                  onPick: ready ? () => Navigator.pop(context, v.id) : null,
                );
              },
            ),
            const SliverToBoxAdapter(child: SizedBox(height: 32)),
          ],
        ),
      ),
    );
  }
}

class VoiceTile extends StatelessWidget {
  const VoiceTile({
    super.key,
    required this.title,
    required this.subtitle,
    required this.onPick,
    this.voice,
    this.speed = 1.0,
    this.selected = false,
    this.enabled = true,
    this.trailing,
  });

  final String title;
  final String subtitle;
  final VoidCallback? onPick;

  /// Null hides the Try button (not downloaded).
  final Voice? voice;
  final double speed;
  final bool selected;
  final bool enabled;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final preview = context.scope.preview;
    final v = voice;
    final busy = v != null && preview.busy == v.id;
    return ListTile(
      enabled: enabled,
      selected: selected,
      title: Text(title),
      subtitle: Text(subtitle, maxLines: 2, overflow: TextOverflow.ellipsis),
      leading: v == null
          ? const SizedBox(width: 40, child: Icon(Icons.download_outlined))
          : IconButton(
              tooltip: busy ? 'Stop' : 'Try',
              icon: busy ? const Icon(Icons.stop_circle_outlined) : const Icon(Icons.play_circle_outline),
              onPressed: () async {
                await preview.play(v, speed: speed);
                if (preview.error != null && context.mounted) context.toast(preview.error!);
              },
            ),
      trailing: trailing ?? (selected ? const Icon(Icons.check_circle) : null),
      onTap: onPick,
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => SliverToBoxAdapter(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Text(text, style: Theme.of(context).textTheme.titleMedium),
        ),
      );
}
