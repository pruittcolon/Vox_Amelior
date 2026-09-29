import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Browse the past by day and conversation, or search everything.
class TimelineScreen extends StatefulWidget {
  const TimelineScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  final _search = TextEditingController();
  Timer? _debounce;
  List<DaySummary> _days = const [];
  DateTime _selected = _dayOf(DateTime.now());
  List<ConversationSummary> _conversations = const [];
  List<SegmentView> _results = const [];

  AppServices get s => widget.services;
  bool get _searching => _search.text.trim().isNotEmpty;

  static DateTime _dayOf(DateTime d) => DateTime(d.year, d.month, d.day);

  @override
  void initState() {
    super.initState();
    _load();
    s.dataVersion.addListener(_load);
  }

  @override
  void dispose() {
    s.dataVersion.removeListener(_load);
    _debounce?.cancel();
    _search.dispose();
    super.dispose();
  }

  void _load() {
    if (!mounted) return;
    final days = s.transcripts.days();
    final today = _dayOf(DateTime.now());
    // Always offer today, even before anything is recorded.
    final withToday = days.isNotEmpty && days.first.day == today ? days : [DaySummary(day: today, conversations: 0, segments: 0), ...days];
    setState(() {
      _days = withToday;
      _conversations = s.transcripts.conversationsBetween(_selected, _selected.add(const Duration(days: 1)));
      if (_searching) _runSearch();
    });
  }

  void _select(DateTime day) {
    setState(() {
      _selected = day;
      _conversations = s.transcripts.conversationsBetween(day, day.add(const Duration(days: 1)));
    });
  }

  void _runSearch() {
    final words = _search.text.trim().split(RegExp(r'\s+'));
    _results = s.transcripts.search(SegmentQuery(keywords: words, limit: 100));
  }

  Future<void> _pickDate() async {
    final first = _days.isEmpty ? DateTime.now() : _days.last.day;
    final picked = await showDatePicker(
      context: context,
      initialDate: _selected,
      firstDate: first.isAfter(DateTime.now()) ? DateTime.now() : first,
      lastDate: DateTime.now(),
    );
    if (picked != null) _select(_dayOf(picked));
  }

  void _open(int conversationId, {int? highlight}) => Navigator.push(
        context,
        MaterialPageRoute<void>(
          builder: (_) => ConversationScreen(services: s, conversationId: conversationId, highlightSegmentId: highlight),
        ),
      );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Timeline'),
        actions: [IconButton(tooltip: 'Pick a date', icon: const Icon(Icons.calendar_month_rounded), onPressed: _pickDate)],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: TextField(
              controller: _search,
              textInputAction: TextInputAction.search,
              decoration: InputDecoration(
                hintText: 'Search everything that was said',
                prefixIcon: const Icon(Icons.search_rounded),
                suffixIcon: _searching
                    ? IconButton(
                        icon: const Icon(Icons.close_rounded),
                        onPressed: () => setState(() {
                          _search.clear();
                          _results = const [];
                        }),
                      )
                    : null,
              ),
              onChanged: (_) {
                _debounce?.cancel();
                _debounce = Timer(const Duration(milliseconds: 250), () => setState(_runSearch));
              },
            ),
          ),
          if (!_searching) _dayStrip(context),
          Expanded(child: _searching ? _searchResults(context) : _dayView(context)),
        ],
      ),
    );
  }

  Widget _dayStrip(BuildContext context) {
    final t = Theme.of(context);
    return SizedBox(
      height: 84,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        scrollDirection: Axis.horizontal,
        itemCount: _days.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, i) {
          final d = _days[i];
          final selected = d.day == _selected;
          final fg = selected ? t.colorScheme.onPrimary : t.colorScheme.onSurface;
          return InkWell(
            borderRadius: BorderRadius.circular(18),
            onTap: () => _select(d.day),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              width: 64,
              decoration: BoxDecoration(
                color: selected ? t.colorScheme.primary : t.colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(18),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(const ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][d.day.weekday - 1],
                      style: t.textTheme.labelSmall?.copyWith(color: fg, fontWeight: FontWeight.w700)),
                  Text('${d.day.day}', style: t.textTheme.titleLarge?.copyWith(color: fg, fontWeight: FontWeight.w800)),
                  Text(d.conversations == 0 ? '–' : '${d.conversations}',
                      style: t.textTheme.labelSmall?.copyWith(color: fg.withValues(alpha: 0.8))),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _dayView(BuildContext context) {
    final t = Theme.of(context);
    if (_conversations.isEmpty) {
      return EmptyState(
        icon: Icons.event_busy_rounded,
        title: 'No conversations on ${formatDayName(_selected)}',
        message: 'Pick another day above, or search.',
      );
    }
    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      itemCount: _conversations.length + 1,
      separatorBuilder: (_, _) => const SizedBox(height: 10),
      itemBuilder: (context, i) {
        if (i == 0) {
          final lines = _conversations.fold<int>(0, (a, c) => a + c.segmentCount);
          return Padding(
            padding: const EdgeInsets.fromLTRB(4, 4, 4, 2),
            child: Text(
              '${formatDayName(_selected)} · ${_conversations.length} conversation${_conversations.length == 1 ? '' : 's'} · $lines lines',
              style: t.textTheme.titleSmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
            ),
          );
        }
        final c = _conversations[i - 1];
        return VoxCard(
          onTap: () => _open(c.id),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text('${formatTime(c.startedAt)} – ${formatTime(c.endedAt)}',
                      style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(width: 8),
                  Text(formatDuration(c.duration), style: t.textTheme.bodySmall),
                  const Spacer(),
                  AvatarStack(labels: c.participants),
                ],
              ),
              const SizedBox(height: 8),
              Text(c.preview, maxLines: 2, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyLarge),
              const SizedBox(height: 8),
              Text(
                '${c.participants.take(3).join(', ')}${c.participants.length > 3 ? ' +${c.participants.length - 3}' : ''} · ${c.segmentCount} lines',
                style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _searchResults(BuildContext context) {
    final t = Theme.of(context);
    if (_results.isEmpty) {
      return const EmptyState(icon: Icons.search_off_rounded, title: 'No matches', message: 'Try other words.');
    }
    return ListView.builder(
      padding: const EdgeInsets.only(bottom: 24),
      itemCount: _results.length,
      itemBuilder: (context, i) {
        final r = _results[i];
        return ListTile(
          leading: SpeakerAvatar(label: r.speakerLabel, known: r.isKnownSpeaker),
          title: Text(r.text),
          subtitle: Text('${r.speakerLabel} · ${formatDayName(r.startedAt)} ${formatTime(r.startedAt)}',
              style: TextStyle(color: t.colorScheme.onSurfaceVariant)),
          onTap: () => _open(r.conversationId, highlight: r.id),
        );
      },
    );
  }
}
