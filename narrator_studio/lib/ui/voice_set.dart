import 'package:flutter/material.dart';

import '../core/app_state.dart';
import 'scope.dart';

/// Natural (Kokoro), Studio (Supertonic HD) or Lifelike (VibeVoice): one
/// tap recasts every character. Whatever is playing switches straight away when [reloadPlayer].
class VoiceSetSwitch extends StatelessWidget {
  const VoiceSetSwitch({super.key, this.reloadPlayer = true, this.onChanged});

  final bool reloadPlayer;
  final VoidCallback? onChanged;

  Future<void> _set(BuildContext context, String set) async {
    final app = context.app;
    final player = context.scope.player;
    await app.settings.update((s) => s.voiceSet = set);
    onChanged?.call();
    final s = player.session;
    if (reloadPlayer && s != null) {
      try {
        await player.reload(await app.openSession(s.doc));
      } catch (_) {}
    }
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final set = app.settings.voiceSet;
    final text = Theme.of(context).textTheme;
    final missing = !app.hasVoiceSet(set);
    final active = app.activeVoiceSet;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SegmentedButton<String>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: 'natural', label: Text('Natural')),
            ButtonSegment(value: 'studio', label: Text('Studio')),
            ButtonSegment(value: 'lifelike', label: Text('Lifelike')),
          ],
          selected: {set},
          onSelectionChanged: (v) => _set(context, v.first),
        ),
        const SizedBox(height: 6),
        Text(
          switch (set) {
            'studio' => 'Studio: Supertonic HD. Very crisp and fast, but a flatter, more even delivery.',
            'lifelike' => 'Lifelike: VibeVoice by Microsoft, the most human-sounding voices of all. Heavy work for a '
                'phone: export an episode, then listen.',
            _ => 'Natural: Kokoro. Warm, human-sounding voices that play straight away. Recommended for listening live.',
          },
          style: text.bodySmall,
        ),
        if (missing)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              'Download the ${_packName(set)} in the Voices tab'
              '${active != set && app.hasVoiceSet(active) ? '; until then the ${_packName(active)} read.' : '.'}',
              style: text.bodySmall?.copyWith(color: Theme.of(context).colorScheme.error),
            ),
          ),
      ],
    );
  }
}

String _packName(String set) => switch (set) {
      'studio' => 'Studio voices (Supertonic HD)',
      'lifelike' => 'Lifelike voices (VibeVoice)',
      _ => 'Natural voices (Kokoro)',
    };

/// A short name for the voice set in use.
String voiceSetName(AppState app) => switch (app.activeVoiceSet) {
      'studio' => 'Studio voices',
      'lifelike' => 'Lifelike voices',
      _ => 'Natural voices',
    };
