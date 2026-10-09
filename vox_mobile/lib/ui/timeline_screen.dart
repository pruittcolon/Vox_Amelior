import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/search/hybrid_search.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/search_answer.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Browse the past by day and conversation, or search everything.
class TimelineScreen extends StatefulWidget {
  const TimelineScreen({
    super.key,
    required this.services,
    this.initialPeople = const {},
    this.initialMoods = const {},
    this.initialQuery,
  });

  final AppServices services;

  /// Start filtered (e.g. opened from Insights): people, moods, search words.
  final Set<String> initialPeople;
  final Set<String> initialMoods;
  final String? initialQuery;

  @override
  State<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  final _search = TextEditingController();
  Timer? _debounce;
  List<DaySummary> _days = const [];
  DateTime _selected = _dayOf(DateTime.now());
  List<ConversationSummary> _conversations = const [];
  /// Search results, and how each was found.
  List<SearchHit> _hits = const [];
  SearchMode _mode = SearchMode.smart;

  /// The meaning half of a search is still running.
  bool _meaningPending = false;

  /// Bumped per search, so a slow answer for an old query is dropped.
  int _searchToken = 0;
  StreamSubscription<({int done, int total})>? _indexing;

  /// People picked to narrow the timeline (conversations where all of them talked).
  final Set<String> _people = {};
  List<SpeakerProfile> _profiles = const [];
  List<ConversationSummary> _withPeople = const [];

  /// Tones picked to narrow the timeline (conversations with lines in any of them).
  final Set<String> _moods = {};
  bool _hasTones = false;

  /// Order of the filtered list.
  _Sort _sort = _Sort.newest;

  AppServices get s => widget.services;
  bool get _searching => _search.text.trim().isNotEmpty;
  bool get _filtering => _people.isNotEmpty || _moods.isNotEmpty;

  /// Conversations matching the picked people (all of them talked) and moods.
  List<ConversationSummary> _filtered() => _moods.isEmpty
      ? s.transcripts.conversationsWith(_people)
      : s.transcripts.conversationsWithTone(_moods, speakerIds: _people);

  /// Moods offered as filters.
  static final List<String> _moodChoices = [...Tone.names.where((n) => n != 'neutral'), 'laughter'];

  static DateTime _dayOf(DateTime d) => DateTime(d.year, d.month, d.day);

  @override
  void initState() {
    super.initState();
    _people.addAll(widget.initialPeople);
    _moods.addAll(widget.initialMoods);
    if (widget.initialQuery != null) _search.text = widget.initialQuery!;
    _load();
    s.dataVersion.addListener(_load);
    // Meaning search catching up: refresh the "lines ready" note, and the
    // results once every line is ready.
    _indexing = s.indexer.progress.listen((p) {
      if (!mounted || !_searching) return;
      setState(() {
        if (p.done >= p.total) _runSearch(refresh: true);
      });
    });
  }

  @override
  void dispose() {
    unawaited(_indexing?.cancel());
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
    final hasTones = s.transcripts.hasTones;
    // Without any tones the mood chips are hidden, so a picked mood could not be cleared.
    if (!hasTones) _moods.clear();
    setState(() {
      _profiles = profiles;
      _hasTones = hasTones;
      _withPeople = _filtered();
      _days = withToday;
      _conversations = s.transcripts.conversationsBetween(_selected, _selected.add(const Duration(days: 1)));
      // A search opened with the screen is a first search, not a refresh.
      if (_searching) _runSearch(refresh: _searchToken > 0);
    });
  }

  void _select(DateTime day) {
    setState(() {
      _selected = day;
      _conversations = s.transcripts.conversationsBetween(day, day.add(const Duration(days: 1)));
    });
  }

  SearchFilters get _filters => SearchFilters(speakerIds: {..._people}, emotions: {..._moods});

  /// Word matches show at once; meaning matches (a model run) are merged in
  /// when they arrive. A [refresh] (new lines heard, more lines ready) keeps
  /// the current results on screen until the new ones are in.
  void _runSearch({bool refresh = false}) {
    final query = _search.text.trim();
    final token = ++_searchToken;
    List<SearchHit> byWords() =>
        [for (final seg in s.search.words(query, _filters, limit: 100)) SearchHit(seg, byWords: true, byMeaning: false)];
    if (_mode == SearchMode.words || s.embedder == null || query.isEmpty) {
      _hits = _mode == SearchMode.meaning ? const [] : byWords();
      _meaningPending = false;
      return;
    }
    if (!refresh) {
      _hits = _mode == SearchMode.meaning ? const [] : byWords();
      _meaningPending = true;
      s.indexForSearch();
    }
    unawaited(s.search.search(query, _filters, mode: _mode, limit: 100).then((hits) {
      if (!mounted || token != _searchToken) return;
      setState(() {
        _hits = hits;
        _meaningPending = false;
      });
    }, onError: (Object e) {
      if (!mounted || token != _searchToken) return;
      setState(() => _meaningPending = false);
      if (!refresh) showMessage(context, 'Search by meaning failed: $e');
    }));
  }

  void _setMode(SearchMode m) => setState(() {
        _mode = m;
        _runSearch();
      });

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

  void _clearFilters() => setState(() {
        _people.clear();
        _moods.clear();
        _withPeople = const [];
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
        actions: [
          if (_filtering && !_searching)
            PopupMenuButton<_Sort>(
              tooltip: 'Sort',
              icon: const Icon(Icons.sort_rounded),
              initialValue: _sort,
              onSelected: (v) => setState(() => _sort = v),
              itemBuilder: (_) => [
                for (final v in _Sort.values)
                  if (v != _Sort.heated || _hasTones) PopupMenuItem(value: v, child: Text(v.label)),
              ],
            ),
          IconButton(tooltip: 'Pick a date', icon: const Icon(Icons.calendar_month_rounded), onPressed: _pickDate),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: TextField(
              controller: _search,
              textInputAction: TextInputAction.search,
              decoration: InputDecoration(
                hintText: _people.isNotEmpty
                    ? 'Search what they said'
                    : _moods.isNotEmpty
                        ? 'Search lines with that mood'
                        : 'Search everything that was said',
                prefixIcon: const Icon(Icons.search_rounded),
                suffixIcon: _searching
                    ? IconButton(
                        icon: const Icon(Icons.close_rounded),
                        onPressed: () => setState(() {
                          _search.clear();
                          _hits = const [];
                          _searchToken++;
                          _meaningPending = false;
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
          if (_profiles.isNotEmpty || _hasTones) _filterStrip(context),
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

  /// One row of filters: people (conversations where all of them talked)
  /// and, once lines have a tone, moods. "All" clears both.
  Widget _filterStrip(BuildContext context) {
    final t = Theme.of(context);
    Widget spaced(Widget chip) => Padding(padding: const EdgeInsets.only(right: 6), child: chip);
    return SizedBox(
      height: 48,
      child: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        children: [
          spaced(ChoiceChip(label: const Text('All'), selected: !_filtering, onSelected: (_) => _clearFilters())),
          for (final p in _profiles)
            spaced(FilterChip(
              avatar: SpeakerAvatar(label: p.name, radius: 11),
              label: Text(p.name),
              selected: _people.contains(p.id),
              onSelected: (_) => _togglePerson(p.id),
            )),
          if (_hasTones) ...[
            if (_profiles.isNotEmpty)
              Center(
                child: Container(
                  width: 1,
                  height: 24,
                  margin: const EdgeInsets.only(left: 4, right: 10),
                  color: t.colorScheme.outlineVariant,
                ),
              ),
            for (final tone in _moodChoices)
              spaced(FilterChip(
                label: Text('${Tone.emoji(tone)} ${Tone.label(tone)}'),
                selected: _moods.contains(tone),
                onSelected: (_) => _toggleMood(tone),
              )),
          ],
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
    final list = _sorted(_withPeople);
    final byDay = _sort == _Sort.newest || _sort == _Sort.oldest;
    if (!byDay) {
      items.add(Padding(
        padding: const EdgeInsets.fromLTRB(4, 0, 4, 0),
        child: Text('Sorted by ${_sort.label.toLowerCase()}', style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
      ));
    }
    DateTime? day;
    for (final c in list) {
      final d = _dayOf(c.startedAt);
      if (byDay && d != day) {
        day = d;
        items.add(Padding(
          padding: const EdgeInsets.fromLTRB(4, 12, 4, 0),
          child: Text(formatDayName(d), style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        ));
      }
      items.add(_conversationCard(context, c, showDay: !byDay));
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
              // Shrinks with large system text instead of overflowing the tile.
              child: FittedBox(
                fit: BoxFit.scaleDown,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(const ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][d.day.weekday - 1],
                        style: t.textTheme.labelSmall?.copyWith(color: fg, fontWeight: FontWeight.w700)),
                    Text('${d.day.day}', style: t.textTheme.titleLarge?.copyWith(color: fg, fontWeight: FontWeight.w800)),
                    Text(d.conversations == 0 ? '–' : '${d.conversations}',
                        style: t.textTheme.labelSmall?.copyWith(color: fg.withValues(alpha: 0.8))),
                  ],
                ),
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

  /// [list] in the chosen order (it arrives newest first).
  List<ConversationSummary> _sorted(List<ConversationSummary> list) {
    final out = List.of(list);
    switch (_sort) {
      case _Sort.newest:
        break;
      case _Sort.oldest:
        out.sort((a, b) => a.startedAt.compareTo(b.startedAt));
      case _Sort.longest:
        out.sort((a, b) => b.duration.compareTo(a.duration));
      case _Sort.lines:
        out.sort((a, b) => b.segmentCount.compareTo(a.segmentCount));
      case _Sort.heated:
        int heat(ConversationSummary c) => s.transcripts.mood(c.id).counts.entries
            .where((e) => e.key == 'angry' || e.key == 'disgusted')
            .fold(0, (a, e) => a + e.value);
        final scores = {for (final c in out) c.id: heat(c)};
        out.sort((a, b) => scores[b.id]!.compareTo(scores[a.id]!));
    }
    return out;
  }

  Widget _conversationCard(BuildContext context, ConversationSummary c, {bool showDay = false}) {
    final t = Theme.of(context);
    final mood = _hasTones ? s.transcripts.mood(c.id) : null;
    final top = mood?.notable.take(3) ?? const <MapEntry<String, int>>[];
    return VoxCard(
          onTap: () => _open(c.id),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (showDay)
                Text(formatDayName(c.startedAt), style: t.textTheme.labelMedium?.copyWith(color: t.colorScheme.primary)),
              Row(
                children: [
                  // One text, so the time is never squeezed by the duration; it
                  // only shortens when large system text leaves no room.
                  Expanded(
                    child: Text.rich(
                      TextSpan(
                        text: '${formatTime(c.startedAt)} – ${formatTime(c.endedAt)}',
                        style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                        children: [TextSpan(text: '   ${formatDuration(c.duration)}', style: t.textTheme.bodySmall)],
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  AvatarStack(labels: c.participants),
                ],
              ),
              const SizedBox(height: 8),
              Text(c.preview, maxLines: 2, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyLarge),
              const SizedBox(height: 8),
              Text(
                '${c.participants.take(3).join(', ')}${c.participants.length > 3 ? ' +${c.participants.length - 3}' : ''} · ${c.segmentCount} lines'
                '${top.map((e) => ' · ${Tone.emoji(e.key)} ${e.value}').join()}',
                style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
              if (mood != null && mood.notable.isNotEmpty) ...[
                const SizedBox(height: 8),
                MoodStrip(counts: mood.counts, height: 4),
              ],
            ],
          ),
        );
  }

  /// [text] with the searched words in bold.
  TextSpan _highlight(String text, TextStyle? bold) {
    final words = _search.text.trim().split(RegExp(r'\s+')).where((w) => w.length > 1).map(RegExp.escape).toList();
    if (words.isEmpty) return TextSpan(text: text);
    final re = RegExp('(${words.join('|')})', caseSensitive: false);
    final spans = <TextSpan>[];
    var at = 0;
    for (final m in re.allMatches(text)) {
      if (m.start > at) spans.add(TextSpan(text: text.substring(at, m.start)));
      spans.add(TextSpan(text: m.group(0), style: bold));
      at = m.end;
    }
    if (at < text.length) spans.add(TextSpan(text: text.substring(at)));
    return TextSpan(children: spans);
  }

  Widget _searchResults(BuildContext context) {
    final t = Theme.of(context);
    final query = _search.text.trim();
    final meaningOn = s.embedder != null;
    return ListView(
      padding: const EdgeInsets.only(bottom: 24),
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 4),
          child: Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final (m, label, icon) in const [
                (SearchMode.smart, 'Smart', Icons.auto_awesome_rounded),
                (SearchMode.words, 'Exact words', Icons.short_text_rounded),
                (SearchMode.meaning, 'Meaning', Icons.lightbulb_outline_rounded),
              ])
                if (m == SearchMode.words || meaningOn)
                  ChoiceChip(
                    avatar: Icon(icon, size: 16),
                    label: Text(label),
                    selected: _mode == m || (!meaningOn && m == SearchMode.words),
                    onSelected: (_) => _setMode(m),
                  ),
            ],
          ),
        ),
        _searchNote(context),
        _askCard(context, query),
        if (_hits.isEmpty && !_meaningPending)
          const Padding(
            padding: EdgeInsets.only(top: 24),
            child: EmptyState(icon: Icons.search_off_rounded, title: 'No matches', message: 'Try other words, or search by meaning.'),
          ),
        if (_meaningPending && _hits.isEmpty)
          const Padding(padding: EdgeInsets.all(32), child: Center(child: CircularProgressIndicator())),
        for (final h in _hits) _hitTile(context, h, t),
      ],
    );
  }

  /// Meaning search: how far indexing has got, or how to turn it on.
  Widget _searchNote(BuildContext context) {
    final t = Theme.of(context);
    final style = t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant);
    if (s.embedder == null) {
      if (!s.settings.value.meaningSearch || s.models.isInstalled(ModelCatalog.textEmbedder)) return const SizedBox.shrink();
      final state = s.downloads.stateOf(ModelCatalog.textEmbedder);
      return ListTile(
        dense: true,
        leading: const Icon(Icons.lightbulb_outline_rounded),
        title: Text(state.isBusy ? 'Downloading search by meaning…' : 'Also find what was said in other words'),
        subtitle: Text(state.isBusy
            ? describeDownload(state)
            : 'Download search by meaning (${formatBytes(ModelCatalog.textEmbedder.approxDownloadBytes)})'),
        onTap: state.isBusy ? null : () => s.downloads.download(ModelCatalog.textEmbedder),
      );
    }
    final p = s.vectors.progress(ModelCatalog.textEmbedder.id);
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 2, 20, 4),
      child: Row(
        children: [
          if (_meaningPending) ...[
            const SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2)),
            const SizedBox(width: 8),
          ],
          Expanded(
            child: Text(
              p.done >= p.total
                  ? (_meaningPending ? 'Searching by meaning…' : 'Searched by words and meaning')
                  : 'Meaning search ready for ${formatCount(p.done)} of ${formatCount(p.total)} lines — more in the background',
              style: style,
            ),
          ),
        ],
      ),
    );
  }

  /// RAG: Gemma answers the search as a question, from the best matches.
  Widget _askCard(BuildContext context, String query) {
    if (query.split(RegExp(r'\s+')).length < 2) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
      child: VoxCard(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        onTap: () => showSearchAnswer(context, s, query),
        child: Row(
          children: [
            Icon(Icons.auto_awesome_rounded, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 12),
            Expanded(child: Text('Ask Gemma: "$query"', maxLines: 2, overflow: TextOverflow.ellipsis)),
            const Icon(Icons.chevron_right_rounded),
          ],
        ),
      ),
    );
  }

  Widget _hitTile(BuildContext context, SearchHit h, ThemeData t) {
    final r = h.segment;
    final how = [
      if (h.byWords) 'words',
      if (h.byMeaning) 'meaning${h.similarity == null ? '' : ' ${(h.similarity! * 100).round()}%'}',
    ].join(' + ');
    return ListTile(
      leading: SpeakerAvatar(label: r.speakerLabel, known: r.isKnownSpeaker),
      title: Text.rich(_highlight(r.text, TextStyle(fontWeight: FontWeight.w800, color: t.colorScheme.primary))),
      subtitle: Text(
          '${r.speakerLabel}${r.emotion == null || r.emotion == 'neutral' ? '' : ' ${Tone.emoji(r.emotion!)}'} · '
          '${formatDayName(r.startedAt)} ${formatTime(r.startedAt)}${how.isEmpty ? '' : ' · $how'}',
          style: TextStyle(color: t.colorScheme.onSurfaceVariant)),
      onTap: () => _open(r.conversationId, highlight: r.id),
    );
  }
}

/// Orders for the filtered list of conversations.
enum _Sort {
  newest('Newest'),
  oldest('Oldest'),
  longest('Longest'),
  lines('Most lines'),
  heated('Most heated');

  const _Sort(this.label);
  final String label;
}
