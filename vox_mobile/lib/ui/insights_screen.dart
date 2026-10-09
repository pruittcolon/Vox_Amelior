import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/insights_repository.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/speakers_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/timeline_screen.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Periods offered on the Insights screen: label and length in days (0 = all time).
const List<(String, int)> kInsightPeriods = [('Week', 7), ('Month', 30), ('3 months', 90), ('Year', 365), ('All', 0)];

/// Statistics over everything that was said: headline numbers, mood over
/// time, who talks, when, with whom, standout conversations and common
/// words. Everything can be narrowed to one person and tapped through to
/// the conversations behind it.
class InsightsScreen extends StatefulWidget {
  const InsightsScreen({super.key, required this.services, this.personId, this.now});

  final AppServices services;

  /// Start on this person's statistics (e.g. opened from People).
  final String? personId;

  /// Fixed "now" for tests.
  final DateTime? now;

  @override
  State<InsightsScreen> createState() => _InsightsScreenState();
}

class _InsightsScreenState extends State<InsightsScreen> {
  int _days = 30;
  late String? _person = widget.personId;
  HighlightKind _standout = HighlightKind.heated;
  int? _column;
  int? _hour;

  List<SpeakerProfile> _profiles = const [];
  Overview _overview = const Overview();
  Overview? _before;
  List<MoodBucket> _moods = const [];
  Granularity _granularity = Granularity.day;
  List<PersonStats> _people = const [];
  List<int> _hours = List.filled(7 * 24, 0);
  List<PairStats> _pairs = const [];
  List<Highlight> _standouts = const [];
  List<(String, int)> _words = const [];

  AppServices get s => widget.services;
  DateTime get _now => widget.now ?? DateTime.now();

  InsightsScope get _scope {
    if (_days == 0) return InsightsScope(speakerId: _person);
    final n = _now;
    return InsightsScope(from: DateTime(n.year, n.month, n.day - (_days - 1)), to: DateTime(n.year, n.month, n.day + 1), speakerId: _person);
  }

  String? get _personName => _profiles.where((p) => p.id == _person).firstOrNull?.name;

  @override
  void initState() {
    super.initState();
    _load();
    s.dataVersion.addListener(_load);
  }

  @override
  void dispose() {
    s.dataVersion.removeListener(_load);
    super.dispose();
  }

  void _load() {
    if (!mounted) return;
    final ins = s.insights;
    final profiles = s.speakers.profiles();
    if (_person != null && !profiles.any((p) => p.id == _person)) _person = null;
    setState(() {
      _profiles = profiles;
      final q = _scope;
      _overview = ins.overview(q);
      _before = q.previous == null ? null : ins.overview(q.previous!);
      _moods = ins.moodOverTime(q, now: _now);
      _granularity = _moods.length < 2 ? Granularity.day : _guessGranularity(_moods);
      _people = ins.people(q);
      _hours = ins.weekHours(q);
      _pairs = ins.pairs(q);
      _standouts = ins.highlights(q, _standout);
      _words = ins.topWords(q);
      _column = null;
      _hour = null;
    });
  }

  static Granularity _guessGranularity(List<MoodBucket> b) {
    final gap = b[1].start.difference(b[0].start).inDays;
    return gap <= 1 ? Granularity.day : gap <= 7 ? Granularity.week : Granularity.month;
  }

  void _setPeriod(int days) {
    _days = days;
    _load();
  }

  void _setPerson(String? id) {
    _person = id;
    _load();
  }

  void _setStandout(HighlightKind k) => setState(() {
        _standout = k;
        _standouts = s.insights.highlights(_scope, k);
      });

  void _openTimeline({Set<String>? people, Set<String> moods = const {}, String? query}) => Navigator.push(
        context,
        MaterialPageRoute<void>(
          builder: (_) => TimelineScreen(
            services: s,
            initialPeople: people ?? {?_person},
            initialMoods: moods,
            initialQuery: query,
          ),
        ),
      );

  void _openConversation(int id) =>
      Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ConversationScreen(services: s, conversationId: id)));

  double? _delta(num now, num? before) {
    if (_before == null || before == null || before == 0) return null;
    return (now - before) / before;
  }

  @override
  Widget build(BuildContext context) {
    final name = _personName;
    return Scaffold(
      appBar: AppBar(title: Text(name ?? 'Insights')),
      body: RefreshIndicator(
        onRefresh: () async => _load(),
        child: ListView(
          padding: const EdgeInsets.only(bottom: 32),
          children: [
            _chipRow([
              for (final (label, days) in kInsightPeriods)
                ChoiceChip(label: Text(label), selected: _days == days, onSelected: (_) => _setPeriod(days)),
            ]),
            if (_profiles.isNotEmpty)
              _chipRow([
                ChoiceChip(label: const Text('Everyone'), selected: _person == null, onSelected: (_) => _setPerson(null)),
                for (final p in _profiles)
                  ChoiceChip(
                    avatar: SpeakerAvatar(label: p.name, radius: 11),
                    label: Text(p.name),
                    selected: _person == p.id,
                    onSelected: (_) => _setPerson(_person == p.id ? null : p.id),
                  ),
              ]),
            if (_overview.isEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 48),
                child: EmptyState(
                  icon: Icons.insights_rounded,
                  title: 'Nothing to show yet',
                  message: name == null
                      ? 'Statistics appear once Vox has heard some conversations in this period.'
                      : 'Vox has not heard $name in this period.',
                ),
              )
            else ...[
              _kpis(context),
              _moodCard(context),
              if (_person == null) _peopleCard(context) else _withCard(context),
              _hoursCard(context),
              if (_person == null && _pairs.isNotEmpty) _togetherCard(context),
              _standoutCard(context),
              if (_words.isNotEmpty) _wordsCard(context),
            ],
          ],
        ),
      ),
    );
  }

  Widget _chipRow(List<Widget> chips) => SizedBox(
        height: 48,
        child: ListView.separated(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          scrollDirection: Axis.horizontal,
          itemCount: chips.length,
          separatorBuilder: (_, _) => const SizedBox(width: 6),
          itemBuilder: (_, i) => Center(child: chips[i]),
        ),
      );

  /// A titled card section.
  Widget _section(BuildContext context, String title, {String? subtitle, Widget? trailing, required List<Widget> children}) {
    final t = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(child: Text(title, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700))),
                  ?trailing,
                ],
              ),
              if (subtitle != null) ...[
                const SizedBox(height: 2),
                Text(subtitle, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
              ],
              const SizedBox(height: 12),
              ...children,
            ],
          ),
        ),
      ),
    );
  }

  Widget _kpis(BuildContext context) {
    final o = _overview;
    final b = _before;
    final tiles = [
      StatTile(
        label: 'Talk time',
        icon: Icons.timer_outlined,
        value: formatTalk(o.talk),
        delta: _delta(o.talk.inSeconds, b?.talk.inSeconds),
      ),
      StatTile(
        label: 'Conversations',
        icon: Icons.forum_outlined,
        value: formatCount(o.conversations),
        delta: _delta(o.conversations, b?.conversations),
        onTap: () => _openTimeline(),
      ),
      StatTile(
        label: 'Words',
        icon: Icons.notes_rounded,
        value: formatCount(o.words),
        delta: _delta(o.words, b?.words),
      ),
      StatTile(
        label: 'Laughs',
        icon: Icons.sentiment_very_satisfied_outlined,
        value: formatCount(o.laughs),
        delta: _delta(o.laughs, b?.laughs),
        onTap: o.laughs == 0 ? null : () => _openTimeline(moods: {'laughter'}),
      ),
    ];
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      child: Column(
        children: [
          IntrinsicHeight(
            child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [Expanded(child: tiles[0]), const SizedBox(width: 10), Expanded(child: tiles[1])]),
          ),
          const SizedBox(height: 10),
          IntrinsicHeight(
            child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [Expanded(child: tiles[2]), const SizedBox(width: 10), Expanded(child: tiles[3])]),
          ),
        ],
      ),
    );
  }

  String _bucketLabel(DateTime d, {bool long = false}) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return switch (_granularity) {
      Granularity.month => long ? '${months[d.month - 1]} ${d.year}' : months[d.month - 1],
      Granularity.week => long ? 'Week of ${d.day} ${months[d.month - 1]}' : '${d.day} ${months[d.month - 1]}',
      Granularity.day => long ? formatDayName(d, now: _now) : '${d.day} ${months[d.month - 1]}',
    };
  }

  Widget _moodCard(BuildContext context) {
    final t = Theme.of(context);
    final o = _overview;
    final unit = switch (_granularity) { Granularity.day => 'day', Granularity.week => 'week', Granularity.month => 'month' };
    if (o.moods.isEmpty) {
      return _section(
        context,
        'Mood over time',
        children: [
          Text('Turn on tone of voice to see how conversations felt — angry, sad, happy and more — over time.',
              style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
          const SizedBox(height: 10),
          FilledButton.tonalIcon(
            onPressed: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => SpeakersSettingsScreen(services: s))),
            icon: const Icon(Icons.mood_rounded),
            label: const Text('Set up tone of voice'),
          ),
        ],
      );
    }
    final n = _moods.length;
    final labels = <int, String>{
      0: _bucketLabel(_moods.first.start),
      if (n > 2) n ~/ 2: _bucketLabel(_moods[n ~/ 2].start),
      if (n > 1) n - 1: _bucketLabel(_moods.last.start),
    };
    final sel = _column == null ? null : _moods[_column!];
    final feeling = o.feelingLines;
    final present = [for (final m in MoodColors.order) if ((o.moods[m] ?? 0) > 0) m];
    return _section(
      context,
      'Mood over time',
      subtitle: 'Lines said with feeling, per $unit · tap a column',
      children: [
        StackedColumnChart(
          columns: [for (final b in _moods) b.counts],
          labels: labels,
          selected: _column,
          onSelect: (i) => setState(() => _column = i),
        ),
        const SizedBox(height: 10),
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 150),
          child: Text(
            key: ValueKey(_column),
            sel == null
                ? '$feeling of ${formatCount(o.lines)} lines had a clear feeling (${o.lines == 0 ? 0 : (feeling * 100 / o.lines).round()}%).'
                : '${_bucketLabel(sel.start, long: true)}: ${sel.feeling == 0 ? 'no strong feelings' : [
                    for (final m in MoodColors.order)
                      if ((sel.counts[m] ?? 0) > 0) '${sel.counts[m]} ${m.toLowerCase()}',
                  ].join(' · ')} · ${sel.lines} lines',
            style: t.textTheme.bodyMedium,
          ),
        ),
        const SizedBox(height: 10),
        // Legend: always present, with counts; tap one to see those lines.
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final m in present)
              ActionChip(
                avatar: Swatch(MoodColors.of(context, m)),
                label: Text('${Tone.label(m)} ${o.moods[m]}'),
                onPressed: () => _openTimeline(moods: {m}),
              ),
          ],
        ),
      ],
    );
  }

  Widget _peopleCard(BuildContext context) {
    if (_people.isEmpty) return const SizedBox.shrink();
    final total = _people.fold<int>(0, (a, p) => a + p.talk.inSeconds);
    final top = _people.first.talk.inSeconds;
    return _section(
      context,
      'Who talks',
      subtitle: 'Share of talk time among people you added · tap for their statistics',
      children: [
        for (final p in _people)
          RankBar(
            leading: SpeakerAvatar(label: p.name),
            label: p.name,
            value: '${formatTalk(p.talk)} · ${total == 0 ? 0 : (p.talk.inSeconds * 100 / total).round()}%',
            fraction: top == 0 ? 0 : p.talk.inSeconds / top,
            below: MoodStrip(counts: p.moods),
            onTap: () => _setPerson(p.id),
          ),
      ],
    );
  }

  Widget _withCard(BuildContext context) {
    final t = Theme.of(context);
    final me = _people.where((p) => p.id == _person).firstOrNull;
    final pairs = _pairs;
    return _section(
      context,
      'Talks most with',
      subtitle: me == null ? null : '${me.conversations} conversations · ${formatCount(me.lines)} lines · ${formatCount(me.words)} words',
      children: [
        if (pairs.isEmpty)
          Text('No conversations with other people you added in this period.',
              style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant))
        else
          for (final pair in pairs)
            () {
              final otherId = pair.a == _person ? pair.b : pair.a;
              final other = pair.a == _person ? pair.bName : pair.aName;
              return RankBar(
                leading: SpeakerAvatar(label: other),
                label: other,
                value: '${pair.conversations} conversation${pair.conversations == 1 ? '' : 's'}',
                fraction: pair.conversations / pairs.first.conversations,
                onTap: () => _openTimeline(people: {_person!, otherId}),
              );
            }(),
      ],
    );
  }

  Widget _hoursCard(BuildContext context) {
    final t = Theme.of(context);
    const days = ['Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays'];
    var peak = 0;
    for (var i = 1; i < _hours.length; i++) {
      if (_hours[i] > _hours[peak]) peak = i;
    }
    String slot(int i) => '${days[i ~/ 24]} ${two(i % 24)}:00–${two((i % 24 + 1) % 24)}:00';
    final sel = _hour;
    return _section(
      context,
      'When you talk',
      subtitle: 'Talk time by weekday and hour · darker is more',
      children: [
        WeekHourHeatmap(seconds: _hours, selected: sel, onSelect: (i) => setState(() => _hour = i)),
        const SizedBox(height: 8),
        Text(
          sel != null
              ? '${slot(sel)}: ${formatTalk(Duration(seconds: _hours[sel]))} in total'
              : _hours[peak] == 0
                  ? 'Not enough talk yet.'
                  : 'Busiest: ${slot(peak)}',
          style: t.textTheme.bodyMedium,
        ),
      ],
    );
  }

  Widget _togetherCard(BuildContext context) {
    final most = _pairs.first.conversations;
    return _section(
      context,
      'Together',
      subtitle: 'Conversations where both talked · tap to read them',
      children: [
        for (final p in _pairs)
          RankBar(
            leading: SizedBox(width: 40, child: AvatarStack(labels: [p.aName, p.bName])),
            label: '${p.aName} & ${p.bName}',
            value: '${p.conversations}',
            fraction: p.conversations / most,
            onTap: () => _openTimeline(people: {p.a, p.b}),
          ),
      ],
    );
  }

  Widget _standoutCard(BuildContext context) {
    final t = Theme.of(context);
    String score(Highlight h) => switch (_standout) {
          HighlightKind.heated => '😠 ${h.value}',
          HighlightKind.laughter => '😂 ${h.value}',
          HighlightKind.longest => formatTalk(Duration(minutes: h.value)),
        };
    return _section(
      context,
      'Standout conversations',
      children: [
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final (k, label) in const [
              (HighlightKind.heated, 'Most heated'),
              (HighlightKind.laughter, 'Most laughter'),
              (HighlightKind.longest, 'Longest'),
            ])
              ChoiceChip(label: Text(label), selected: _standout == k, onSelected: (_) => _setStandout(k)),
          ],
        ),
        const SizedBox(height: 8),
        if (_standouts.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              switch (_standout) {
                HighlightKind.heated => 'No heated moments in this period.',
                HighlightKind.laughter => 'No laughter heard in this period.',
                HighlightKind.longest => 'No conversations longer than a minute.',
              },
              style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
            ),
          )
        else
          for (final h in _standouts)
            ListTile(
              contentPadding: EdgeInsets.zero,
              onTap: () => _openConversation(h.conversation.id),
              title: Text(h.conversation.preview, maxLines: 1, overflow: TextOverflow.ellipsis),
              subtitle: Text(
                '${formatDayName(h.conversation.startedAt, now: _now)} ${formatTime(h.conversation.startedAt)} · '
                '${h.conversation.participants.take(3).join(', ')}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              trailing: Text(score(h), style: t.textTheme.labelLarge),
            ),
      ],
    );
  }

  Widget _wordsCard(BuildContext context) {
    return _section(
      context,
      'Most said',
      subtitle: 'Common words, little ones left out · tap to search',
      children: [
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final (w, n) in _words) ActionChip(label: Text('$w · $n'), onPressed: () => _openTimeline(query: w)),
          ],
        ),
      ],
    );
  }
}
