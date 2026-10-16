import 'package:flutter/material.dart';

import '../core/audio/player.dart';
import 'library_screen.dart';
import 'reader_screen.dart';
import 'scope.dart';
import 'voices_screen.dart';
import 'write_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;

  void goTo(int tab) => setState(() => _tab = tab);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: [
          LibraryScreen(onOpenVoices: () => goTo(1), onOpenWrite: () => goTo(2)),
          const VoicesScreen(),
          const WriteScreen(),
        ],
      ),
      bottomNavigationBar: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const MiniPlayer(),
          NavigationBar(
            selectedIndex: _tab,
            onDestinationSelected: goTo,
            destinations: const [
              NavigationDestination(icon: Icon(Icons.library_books_outlined), selectedIcon: Icon(Icons.library_books), label: 'Library'),
              NavigationDestination(icon: Icon(Icons.record_voice_over_outlined), selectedIcon: Icon(Icons.record_voice_over), label: 'Voices'),
              NavigationDestination(icon: Icon(Icons.edit_note_outlined), selectedIcon: Icon(Icons.edit_note), label: 'Write'),
            ],
          ),
        ],
      ),
    );
  }
}

/// What's playing, from anywhere in the app.
class MiniPlayer extends StatelessWidget {
  const MiniPlayer({super.key});

  @override
  Widget build(BuildContext context) {
    final player = context.scope.player;
    return ListenableBuilder(
      listenable: player,
      builder: (context, _) {
        final s = player.session;
        if (s == null || !player.active) return const SizedBox.shrink();
        final scheme = Theme.of(context).colorScheme;
        final total = s.plan.utterances.length;
        return Material(
          color: scheme.secondaryContainer,
          child: InkWell(
            onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: s.doc))),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 6, 4, 6),
              child: Row(
                children: [
                  Icon(Icons.graphic_eq, color: scheme.onSecondaryContainer),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(s.doc.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: Theme.of(context).textTheme.titleSmall),
                        Text(
                          player.state == PlayState.buffering
                              ? 'Preparing voice…'
                              : 'Sentence ${player.current + 1} of $total',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    tooltip: player.isPlaying ? 'Pause' : 'Play',
                    icon: Icon(player.isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded),
                    onPressed: player.toggle,
                  ),
                  IconButton(tooltip: 'Stop', icon: const Icon(Icons.close), onPressed: player.stop),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
