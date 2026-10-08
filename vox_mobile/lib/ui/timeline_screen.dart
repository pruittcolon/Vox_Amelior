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

  /// People picked to narrow the timeline (conversations where all of them talked).
  final Set<String> _people = {};
  List<SpeakerProfile> _profiles = const [];
  List<ConversationSummary> _withPeople = const [];

  /// Tones picked to narrow the timeline (conversations with lines in any of them).
  final Set<String> _moods = {};
  bool _hasTones = false;

  AppServices get s => widget.services;
  bool get _searching => _search.text.trim().isNotEmpty;
  bool get _filtering => _people.isNotEmpty || _moods.isNotEmpty;

  /// Conversations matching the picked people (all of them talked) and moods.
  List<ConversationSummary> _filtered() {
    if (_moods.isEmpty) return s.transcripts.conversationsWith(_people);
    final toned = s.transcripts.conversationsWithTone(_moods, speakerIds: _people, limit: 300);
    if (_people.length < 2) return toned.take(100).toList();
    final together = {for (final c in s.transcripts.conversationsWith(_people, limit: 100000)) c.id};
    return toned.where((c) => together.contains(c.id)).take(100).toList();
  }

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
    final profiles = s.speakers.profiles();
    _people.removeWhere((id) => !profiles.any((p) => p.id == id));
    setState(() {
      _profiles = profiles;
      _withPeople = _filtered();
      _hasTones = s.transcripts.hasTones;
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
    _results = s.transcripts.search(SegmentQuery(keywords: words, speakerIds: {..._people}, emotions: {..._moods}, limit: 100));
  }

  void _togglePerson(String id) => setState(() {
        if (!_people.remove(id)) _people.add(id);
        _withPeople = _filtered();
        if (_searching) _runSearch();
      });

  void _toggleMood(String tone) => setState(() {
        if (!_moods.remove(tone)) _moods.add(tone);
        _withPeople = _filtered();
        if (_searching) _runSearch();
      });

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
          builder: (_) => ConversationScreen(
            services: s,
            conversationId: conversationId,
            highlightSegmentId: highlight,
            focusSpeakerIds: {..._people},
          ),
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
                hintText: _filtering ? 'Search what they said' : 'Search everything that was said',
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
          if (_profiles.isNotEmpty) _peopleStrip(context),
          if (_hasTones) _moodStrip(context),
          if (!_searching && !_filtering) _dayStrip(context),
          Expanded(
            child: _searching
                ? _searchResults(context)
                : _filtering
                    ? _peopleView(context)
                    : _dayView(context),
          ),
        ],
      ),
    );
  }

  /// Pick people to see only conversations they were in.
  Widget _peopleStrip(BuildContext context) {
    return SizedBox(
      height: 48,
      child: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        children: [
          Padding(
            padding: const EdgeInsets.only(right: 6),
            child: ChoiceChip(
              label: const Text('Everyone'),
              selected: _people.isEmpty,
              onSelected: (_) => setState(() {
                _people.clear();
                _withPeople = _filtered();
                if (_searching) _runSearch();
              }),
            ),
          ),
          for (final p in _profiles)
            Padding(
              padding: const EdgeInsets.only(right: 6),
              child: FilterChip(
                avatar: SpeakerAvatar(label: p.name, radius: 11),
                label: Text(p.name),
                selected: _people.contains(p.id),
                onSelected: (_) => _togglePerson(p.id),
              ),
            ),
        ],
      ),
    );
  }

  /// Pick moods to see only conversations where someone sounded that way.
  Widget _moodStrip(BuildContext context) {
    return SizedBox(
      height: 48,
      child: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        children: [
          Padding(
            padding: const EdgeInsets.only(right: 6),
            child: ChoiceChip(
              label: const Text('Any mood'),
              selected: _moods.isEmpty,
              onSelected: (_) => setState(() {
                _moods.clear();
                _withPeople = _filtered();
                if (_searching) _runSearch();
              }),
            ),
          ),
          for (final tone in [...Tone.names.where((n) => n != 'neutral'), 'laughter'])
            Padding(
              padding: const EdgeInsets.only(right: 6),
              child: FilterChip(
                label: Text('${Tone.emoji(tone)} ${Tone.label(tone)}'),
                selected: _moods.contains(tone),
                onSelected: (_) => _toggleMood(tone),
              ),
            ),
        ],
      ),
    );
  }

  /// Conversations with the picked people and moods, newest first, under day headings.
  Widget _peopleView(BuildContext context) {
    final t = Theme.of(context);
    final names = _profiles.where((p) => _people.contains(p.id)).map((p) => p.name).toList();
    final moods = [for (final m in Tone.names.followedBy(Tone.sounds)) if (_moods.contains(m)) m.toLowerCase()];
    final whoNames = names.length <= 2 ? names.join(' and ') : '${names.take(names.length - 1).join(', ')} and ${names.last}';
    final who = [
      if (names.isNotEmpty) 'with $whoNames',
      if (moods.isNotEmpty) 'that sounded ${moods.join(' or ')}',
    ].join(' ');
    if (_withPeople.isEmpty) {
      return EmptyState(
        icon: moods.isEmpty ? Icons.groups_rounded : Icons.mood_rounded,
        title: 'No conversations $who',
        message: names.length > 1
            ? 'Only conversations where all of them talked are shown.'
            : moods.isNotEmpty
                ? 'Only lines heard after the tone model was installed have a mood.'
                : 'Nothing recorded with them yet.',
      );
    }
    final items = <Widget>[
      Padding(
        padding: const EdgeInsets.fromLTRB(4, 4, 4, 2),
        child: Text(
          '${_withPeople.length}${_withPeople.length == 100 ? '+' : ''} conversation${_withPeople.length == 1 ? '' : 's'} $who',
          style: t.textTheme.titleSmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
        ),
      ),
    ];
    DateTime? day;
    for (final c in _withPeople) {
      final d = _dayOf(c.startedAt);
      if (d != day) {
        day = d;
        items.add(Padding(
          padding: const EdgeInsets.fromLTRB(4, 12, 4, 0),
          child: Text(formatDayName(d), style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        ));
      }
      items.add(_conversationCard(context, c));
    }
    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      itemCount: items.length,
      separatorBuilder: (_, _) => const SizedBox(height: 10),
      itemBuilder: (_, i) => items[i],
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
        return _conversationCard(context, _conversations[i - 1]);
      },
    );
  }

  Widget _conversationCard(BuildContext context, ConversationSummary c) {
    final t = Theme.of(context);
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
                '${c.participants.take(3).join(', ')}${c.participants.length > 3 ? ' +${c.participants.length - 3}' : ''} · ${c.segmentCount} lines'
                '${_moodText(c.id)}',
                style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
            ],
          ),
        );
  }

  /// " · 😠 3 · 😊 2" for the conversation's most common non-neutral tones.
  String _moodText(int conversationId) {
    if (!_hasTones) return '';
    final top = s.transcripts.mood(conversationId).notable.take(3);
    return top.map((e) => ' · ${Tone.emoji(e.key)} ${e.value}').join();
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
          subtitle: Text(
              '${r.speakerLabel}${r.emotion == null || r.emotion == 'neutral' ? '' : ' ${Tone.emoji(r.emotion!)}'} · '
              '${formatDayName(r.startedAt)} ${formatTime(r.startedAt)}',
              style: TextStyle(color: t.colorScheme.onSurfaceVariant)),
          onTap: () => _open(r.conversationId, highlight: r.id),
        );
      },
    );
  }
}
