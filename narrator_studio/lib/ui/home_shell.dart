import 'package:flutter/material.dart';

import '../core/audio/player.dart';
import 'art.dart';
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
              NavigationDestination(icon: Icon(Icons.auto_stories_outlined), selectedIcon: Icon(Icons.auto_stories), label: 'Library'),
              NavigationDestination(icon: Icon(Icons.graphic_eq_outlined), selectedIcon: Icon(Icons.graphic_eq), label: 'Voices'),
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
        final theme = Theme.of(context);
        final scheme = theme.colorScheme;
        final total = s.plan.utterances.length;
        final progress = total == 0 ? 0.0 : (player.current + 1) / total;
        return Padding(
          padding: const EdgeInsets.fromLTRB(10, 0, 10, 8),
          child: Material(
            color: scheme.surfaceContainerHigh,
            elevation: 6,
            shadowColor: Colors.black.withValues(alpha: 0.25),
            borderRadius: BorderRadius.circular(18),
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: s.doc))),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(8, 8, 4, 6),
                    child: Row(
                      children: [
                        CoverArt(seed: s.doc.title, style: s.style.id, size: 44, radius: 12),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(s.doc.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.titleSmall),
                              const SizedBox(height: 2),
                              Row(
                                children: [
                                  Equalizer(playing: player.isPlaying, size: 11, color: scheme.secondary),
                                  const SizedBox(width: 6),
                                  Flexible(
                                    child: Text(
                                      player.state == PlayState.buffering
                                          ? 'Preparing voice…'
                                          : 'Sentence ${player.current + 1} of $total',
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: theme.textTheme.bodySmall,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                        IconButton.filled(
                          tooltip: player.isPlaying ? 'Pause' : 'Play',
                          icon: Icon(player.isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded),
                          onPressed: player.toggle,
                        ),
                        IconButton(tooltip: 'Stop', icon: const Icon(Icons.close_rounded), onPressed: player.stop),
                      ],
                    ),
                  ),
                  LinearProgressIndicator(value: progress, minHeight: 2.5),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
