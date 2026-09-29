import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/service/listening_runtime.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/transcript_list.dart';

/// Start/stop listening, see what is being heard, search the past.
class LiveScreen extends StatefulWidget {
  const LiveScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> {
  final _search = TextEditingController();
  StreamSubscription<Map<Object?, Object?>>? _events;
  Timer? _poll;
  List<SegmentView> _segments = const [];
  bool _starting = false;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _load();
    s.dataVersion.addListener(_load);
    _events = s.listening.events.listen((e) {
      if (e['type'] == ServiceEvents.segment) _load();
    });
    // The service writes from another isolate; poll as a fallback.
    _poll = Timer.periodic(const Duration(seconds: 5), (_) => _load());
    unawaited(s.listening.refresh());
  }

  @override
  void dispose() {
    s.dataVersion.removeListener(_load);
    unawaited(_events?.cancel());
    _poll?.cancel();
    _search.dispose();
    super.dispose();
  }

  void _load() {
    if (!mounted) return;
    final q = _search.text.trim();
    setState(() {
      _segments = q.isEmpty
          ? s.transcripts.recent(limit: 200)
          : s.transcripts.search(SegmentQuery(keywords: q.split(RegExp(r'\s+')), limit: 100));
    });
  }

  Future<void> _toggle() async {
    if (s.listening.isRunning) {
      await s.listening.stop();
      return;
    }
    if (!s.writeServiceConfig()) {
      showMessage(context, 'Download the speech models first (Settings → Models).');
      return;
    }
    setState(() => _starting = true);
    final error = await s.listening.start();
    if (!mounted) return;
    setState(() => _starting = false);
    if (error != null) showMessage(context, error);
  }

  Future<void> _onTapSegment(SegmentView seg) async {
    final people = s.speakers.profiles();
    final choice = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (c) => SafeArea(
        child: ListView(
          shrinkWrap: true,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: Text('Who said this?', style: Theme.of(c).textTheme.titleMedium),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text('"${seg.text}"', maxLines: 3, overflow: TextOverflow.ellipsis),
            ),
            for (final p in people)
              ListTile(
                leading: const Icon(Icons.person),
                title: Text(p.name),
                selected: p.id == seg.speakerId,
                onTap: () => Navigator.pop(c, p.id),
              ),
            if (people.isEmpty)
              const ListTile(title: Text('Add people on the People tab to label voices.')),
            ListTile(
              leading: const Icon(Icons.copy),
              title: const Text('Copy text'),
              onTap: () => Navigator.pop(c, '#copy'),
            ),
          ],
        ),
      ),
    );
    if (choice == null || !mounted) return;
    if (choice == '#copy') {
      await Clipboard.setData(ClipboardData(text: seg.text));
      if (mounted) showMessage(context, 'Copied.');
      return;
    }
    s.speakers.assignSegmentToSpeaker(seg.id, choice);
    s.dataChanged();
    showMessage(context, 'Thanks — Vox will recognise this voice better.');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vox')),
      body: Column(
        children: [
          ListenableBuilder(listenable: s.listening, builder: (context, _) => _statusCard(context)),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
            child: SearchBar(
              controller: _search,
              hintText: 'Search what was said',
              leading: const Icon(Icons.search),
              trailing: [
                if (_search.text.isNotEmpty)
                  IconButton(
                    icon: const Icon(Icons.clear),
                    onPressed: () {
                      _search.clear();
                      _load();
                    },
                  ),
              ],
              onChanged: (_) => _load(),
            ),
          ),
          Expanded(
            child: TranscriptList(
              segments: _segments,
              onTap: _onTapSegment,
              emptyText: _search.text.isEmpty
                  ? 'Nothing heard yet. Tap "Start listening" and talk normally.'
                  : 'No matches.',
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusCard(BuildContext context) {
    final l = s.listening;
    final running = l.isRunning;
    final color = !running ? Colors.grey : (l.isPaused ? Colors.orange : Colors.green);
    final label = !running ? 'Not listening' : (l.isPaused ? 'Paused' : 'Listening');
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(running && !l.isPaused ? Icons.hearing : Icons.hearing_disabled, color: color),
                const SizedBox(width: 8),
                Text(label, style: Theme.of(context).textTheme.titleMedium?.copyWith(color: color)),
                const Spacer(),
                if (running) Text('${l.heardToday} today'),
              ],
            ),
            if (l.lastError != null) ...[
              const SizedBox(height: 8),
              Text(l.lastError!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _starting ? null : _toggle,
                    icon: Icon(running ? Icons.stop : Icons.mic),
                    label: Text(_starting ? 'Starting…' : (running ? 'Stop listening' : 'Start listening')),
                  ),
                ),
                if (running) ...[
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: l.isPaused ? l.resume : l.pause,
                    child: Text(l.isPaused ? 'Resume' : 'Pause'),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}
