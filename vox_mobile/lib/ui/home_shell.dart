import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/ui/ask_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/insights_screen.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/now_screen.dart';
import 'package:vox_amelior_mobile/ui/people_screen.dart';
import 'package:vox_amelior_mobile/ui/prompt_editor.dart';
import 'package:vox_amelior_mobile/ui/reviews_screen.dart';
import 'package:vox_amelior_mobile/ui/timeline_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.services});

  final AppServices services;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;

  @override
  void initState() {
    super.initState();
    _colors();
    widget.services.dataVersion.addListener(_colors);
  }

  @override
  void dispose() {
    widget.services.dataVersion.removeListener(_colors);
    super.dispose();
  }

  /// Everyone's colour, kept up to date as people are added or renamed.
  void _colors() => setHouseholdColors(widget.services.speakers.profiles().map((p) => p.name));

  void _openModels() => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ModelsScreen(services: widget.services)));

  @override
  Widget build(BuildContext context) {
    final s = widget.services;
    // Settings (More) open from the gear on Now and People.
    final pages = [
      NowScreen(services: s),
      TimelineScreen(services: s),
      InsightsScreen(services: s),
      ListenableBuilder(
        listenable: Listenable.merge([s.downloads, s.settings, s.listening]),
        builder: (context, _) => AskScreen(
          ask: s.assistant.ask,
          requests: s.requests,
          assistantReady: () => s.assistantReady,
          onOpenModels: _openModels,
          onEditPrompt: () => showPromptEditor(context, s),
          reviewsBuilder: (_) => ReviewsView(services: s),
          onReviewPeriod: (period) => openReviewCreator(context, s, period: period),
        ),
      ),
      PeopleScreen(services: s),
    ];
    return Scaffold(
      body: IndexedStack(
        index: _index,
        // Hidden tabs pause their animations and live readings.
        children: [for (var i = 0; i < pages.length; i++) TickerMode(enabled: i == _index, child: pages[i])],
      ),
      bottomNavigationBar: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _DownloadBar(downloads: s.downloads, onTap: _openModels),
          NavigationBar(
            selectedIndex: _index,
            onDestinationSelected: (i) => setState(() => _index = i),
            destinations: const [
              NavigationDestination(icon: Icon(Icons.graphic_eq_rounded), label: 'Now'),
              NavigationDestination(icon: Icon(Icons.calendar_view_day_rounded), label: 'Timeline'),
              NavigationDestination(icon: Icon(Icons.insights_rounded), label: 'Insights'),
              NavigationDestination(icon: Icon(Icons.auto_awesome_rounded), label: 'Ask'),
              NavigationDestination(icon: Icon(Icons.people_alt_rounded), label: 'People'),
            ],
          ),
        ],
      ),
    );
  }
}

/// App-wide progress for model downloads, visible from every tab.
class _DownloadBar extends StatelessWidget {
  const _DownloadBar({required this.downloads, required this.onTap});

  final ModelDownloads downloads;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: downloads,
      builder: (context, _) {
        final cur = downloads.current;
        if (cur == null) return const SizedBox.shrink();
        final st = cur.state;
        final t = Theme.of(context);
        return Material(
          color: t.colorScheme.secondaryContainer,
          child: InkWell(
            onTap: onTap,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.downloading_rounded, size: 18),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '${st.status == DownloadStatus.unpacking ? 'Setting up' : 'Downloading'} ${cur.asset.title} · '
                          '${describeDownload(st)}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: LinearProgressIndicator(value: st.progress, minHeight: 5),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
