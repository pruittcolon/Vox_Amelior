import 'package:flutter/material.dart';

import '../core/app_state.dart';
import 'scope.dart';

/// Natural (Kokoro) or Studio (Supertonic HD): one tap recasts every
/// character. Whatever is playing switches straight away when [reloadPlayer].
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
    final missing = set == 'studio' ? !app.models.isInstalled('supertonic') : !app.hasKokoro;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SegmentedButton<String>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: 'natural', icon: Icon(Icons.record_voice_over_rounded), label: Text('Natural')),
            ButtonSegment(value: 'studio', icon: Icon(Icons.graphic_eq_rounded), label: Text('Studio')),
          ],
          selected: {set},
          onSelectionChanged: (v) => _set(context, v.first),
        ),
        const SizedBox(height: 6),
        Text(
          set == 'studio'
              ? 'Studio: Supertonic HD. Very crisp and fast, but a flatter, more even delivery.'
              : 'Natural: Kokoro, the most human-sounding voices, with real warmth and rhythm. Recommended.',
          style: text.bodySmall,
        ),
        if (missing)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              'Download ${set == 'studio' ? 'Studio voices (Supertonic HD)' : 'Kokoro'} in the Voices tab; '
              'until then the other set reads.',
              style: text.bodySmall?.copyWith(color: Theme.of(context).colorScheme.error),
            ),
          ),
      ],
    );
  }
}

/// A short name for the voice set in use.
String voiceSetName(AppState app) => app.settings.voiceSet == 'studio' ? 'Studio voices' : 'Natural voices';
