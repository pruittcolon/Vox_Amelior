import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/assistant/time_window.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/capacity_screen.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Periods offered as one-tap choices.
const List<(String, String)> kPeriodChoices = [
  ('Today', 'today'),
  ('Yesterday', 'yesterday'),
  ('This week', 'this week'),
  ('Last week', 'last week'),
  ('This month', 'this month'),
  ('Last month', 'last month'),
];

TimeWindow? windowFor(String phrase, {DateTime? now}) => const TimeWindowParser().parse(phrase, now ?? DateTime.now())?.$1;

// ---- list ---------------------------------------------------------------------

/// Reviews ("go through everything"): newest first, with live progress.
class ReviewsView extends StatefulWidget {
  const ReviewsView({super.key, required this.services});

  final AppServices services;

  @override
  State<ReviewsView> createState() => _ReviewsViewState();
}

class _ReviewsViewState extends State<ReviewsView> {
  Timer? _poll;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    // Progress also arrives as events; polling covers reviews run elsewhere.
    _poll = Timer.periodic(const Duration(seconds: 4), (_) {
      if (mounted && s.reviews.hasActive) setState(() {});
    });
  }

  @override
  void dispose() {
    _poll?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([s.reviewVersion, s.dataVersion]),
      builder: (context, _) {
        final runs = s.reviews.runs();
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
          children: [
            FilledButton.icon(
              onPressed: () => openReviewCreator(context, s),
              icon: const Icon(Icons.manage_search_rounded),
              label: const Text('New review'),
            ),
            const SizedBox(height: 6),
            Text(
              'Gemma reads a whole period part by part, answers your question for each part, and combines the results. '
              'Runs in the background.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            if (s.settings.value.contextTested == 0 && s.assistantReady)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: VoxCard(
                  color: Theme.of(context).colorScheme.tertiaryContainer,
                  onTap: () => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => CapacityScreen(services: s))),
                  child: Row(
                    children: [
                      Icon(Icons.speed_rounded, color: Theme.of(context).colorScheme.onTertiaryContainer),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'Tip: test this phone once, so reviews use the biggest parts it can handle (half of what Gemma can read).',
                          style: TextStyle(color: Theme.of(context).colorScheme.onTertiaryContainer),
                        ),
                      ),
                      Icon(Icons.chevron_right_rounded, color: Theme.of(context).colorScheme.onTertiaryContainer),
                    ],
                  ),
                ),
              ),
            if (runs.isEmpty)
              const Padding(
                padding: EdgeInsets.only(top: 40),
                child: EmptyState(
                  icon: Icons.fact_check_rounded,
                  title: 'No reviews yet',
                  message: 'Try "Logical fallacies" for last week, or "Promises & to-dos" for this month.',
                ),
              ),
            for (final r in runs)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: _ReviewCard(
                  run: r,
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute<void>(builder: (_) => ReviewDetailScreen(services: s, runId: r.id)),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _ReviewCard extends StatelessWidget {
  const _ReviewCard({required this.run, required this.onTap});

  final ReviewRun run;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return VoxCard(
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(run.title, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
              ),
              StatusPill(run: run),
            ],
          ),
          const SizedBox(height: 2),
          Text(run.periodLabel, style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
          if (run.isActive || run.status == ReviewStatus.paused) ...[
            const SizedBox(height: 10),
            LinearProgressIndicator(value: run.progress, minHeight: 6),
            const SizedBox(height: 4),
            Text(progressText(run), style: t.textTheme.labelMedium),
          ] else if (run.finalAnswer != null) ...[
            const SizedBox(height: 8),
            Text(run.finalAnswer!, maxLines: 3, overflow: TextOverflow.ellipsis),
          ],
        ],
      ),
    );
  }
}

String progressText(ReviewRun r) {
  if (r.doneChunks >= r.totalChunks) return 'Combining the results…';
  return 'Part ${r.doneChunks + (r.status == ReviewStatus.running ? 1 : 0)} of ${r.totalChunks}';
}

class StatusPill extends StatelessWidget {
  const StatusPill({super.key, required this.run});

  final ReviewRun run;

  @override
  Widget build(BuildContext context) {
    final c = Theme.of(context).colorScheme;
    final (String text, IconData icon, Color color) = switch (run.status) {
      ReviewStatus.queued => ('Waiting', Icons.schedule_rounded, c.secondary),
      ReviewStatus.running => ('Running', Icons.autorenew_rounded, c.primary),
      ReviewStatus.paused => ('Paused', Icons.pause_circle_rounded, c.tertiary),
      ReviewStatus.done => ('Done', Icons.check_circle_rounded, const Color(0xFF2B8A3E)),
      ReviewStatus.failed => ('Failed', Icons.error_rounded, c.error),
      ReviewStatus.cancelled => ('Stopped', Icons.stop_circle_rounded, c.outline),
    };
    return Pill(text, icon: icon, color: color);
  }
}

// ---- create -------------------------------------------------------------------

Future<void> openReviewCreator(BuildContext context, AppServices services, {String? period}) =>
    Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ReviewCreateScreen(services: services, initialPeriod: period)));

/// Pick what to look for, when, and whose words; then start.
class ReviewCreateScreen extends StatefulWidget {
  const ReviewCreateScreen({super.key, required this.services, this.initialPeriod});

  final AppServices services;
  final String? initialPeriod;

  @override
  State<ReviewCreateScreen> createState() => _ReviewCreateScreenState();
}

class _ReviewCreateScreenState extends State<ReviewCreateScreen> {
  late ReviewTemplate _template = ReviewTemplate.builtIns.first;
  final _prompt = TextEditingController();
  final _format = TextEditingController();
  late String _period = widget.initialPeriod ?? 'last week';
  DateTimeRange? _custom;
  final Set<String> _focus = {};

  /// Read the newest [_count] lines of the chosen people instead of a whole period.
  bool _byLines = false;
  int _count = 100;

  /// Only lines said in these tones (by-lines mode).
  final Set<String> _moods = {};

  static const List<int> countChoices = [50, 100, 200, 500];

  AppServices get s => widget.services;

  List<ReviewTemplate> get _templates => [...ReviewTemplate.builtIns, ...s.settings.value.customTemplates, ReviewTemplate.blank];

  @override
  void initState() {
    super.initState();
    _use(_template);
  }

  @override
  void dispose() {
    _prompt.dispose();
    _format.dispose();
    super.dispose();
  }

  void _use(ReviewTemplate t) {
    setState(() {
      _template = t;
      _prompt.text = t.prompt;
      _format.text = t.format;
    });
  }

  TimeWindow? get _window {
    if (_period == 'custom' && _custom != null) {
      final from = DateTime(_custom!.start.year, _custom!.start.month, _custom!.start.day);
      final to = DateTime(_custom!.end.year, _custom!.end.month, _custom!.end.day).add(const Duration(days: 1));
      return TimeWindow(from, to, '${formatDay(from)} – ${formatDay(to.subtract(const Duration(days: 1)))}');
    }
    return windowFor(_period);
  }

  Future<void> _pickDates() async {
    final now = DateTime.now();
    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(now.year - 3),
      lastDate: now,
      initialDateRange: _custom ?? DateTimeRange(start: now.subtract(const Duration(days: 6)), end: now),
    );
    if (picked != null) {
      setState(() {
        _custom = picked;
        _period = 'custom';
      });
    }
  }

  Future<void> _saveTemplate() async {
    final name = await askText(context, 'Save as my review', initial: _template.builtIn ? '' : _template.name, hint: 'Name');
    if (name == null || name.isEmpty) return;
    final st = s.settings.value;
    final t = ReviewTemplate(
      id: 'my-${DateTime.now().millisecondsSinceEpoch}',
      name: name,
      prompt: _prompt.text.trim(),
      format: _format.text.trim(),
      kind: _template.kind,
    );
    await s.updateSettings(st.copyWith(customTemplates: [...st.customTemplates, t]));
    if (mounted) {
      setState(() => _template = t);
      showMessage(context, 'Saved "$name".');
    }
  }

  Future<void> _deleteTemplate(ReviewTemplate t) async {
    if (!await confirm(context, 'Delete "${t.name}"?', 'Only the saved review is removed.')) return;
    final st = s.settings.value;
    await s.updateSettings(st.copyWith(customTemplates: st.customTemplates.where((x) => x.id != t.id).toList()));
    _use(ReviewTemplate.builtIns.first);
  }

  /// "Last 100 lines · Pruitt, Ericah · angry, sad".
  String _linesLabel(int found, List<SpeakerProfile> people) {
    final names = [for (final p in people) if (_focus.contains(p.id)) p.name];
    return [
      'Last $found line${found == 1 ? '' : 's'}',
      if (names.isNotEmpty) names.join(', '),
      if (_moods.isNotEmpty) _moods.join(', '),
    ].join(' · ');
  }

  void _start() {
    if (_prompt.text.trim().isEmpty) {
      showMessage(context, 'Say what Gemma should look for.');
      return;
    }
    if (_byLines) {
      final found = s.transcripts.lastLines(count: _count, speakerIds: _focus, emotions: _moods).length;
      final id = s.startReviewOfLines(
        template: _template.copyWith(prompt: _prompt.text.trim(), format: _format.text.trim()),
        count: _count,
        speakerIds: {..._focus},
        emotions: {..._moods},
        label: _linesLabel(found, s.speakers.profiles()),
      );
      if (id == null) {
        showMessage(context, 'No lines match.');
        return;
      }
      Navigator.pushReplacement(context, MaterialPageRoute<void>(builder: (_) => ReviewDetailScreen(services: s, runId: id)));
      return;
    }
    final window = _window;
    if (window == null) return;
    final id = s.startReview(
      template: _template.copyWith(prompt: _prompt.text.trim(), format: _format.text.trim()),
      window: window,
      focus: _focus.toList(),
    );
    if (id == null) {
      showMessage(context, 'Nothing was said in that period.');
      return;
    }
    Navigator.pushReplacement(context, MaterialPageRoute<void>(builder: (_) => ReviewDetailScreen(services: s, runId: id)));
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final window = _window;
    final people = s.speakers.profiles();
    final budget = s.budget;
    final picked = _byLines ? s.transcripts.lastLines(count: _count, speakerIds: _focus, emotions: _moods) : null;
    final size = picked != null
        ? (lines: picked.length, chars: picked.fold<int>(0, (a, l) => a + l.text.length))
        : window == null
            ? null
            : s.transcripts.sizeBetween(window.from, window.to);
    final hasTones = s.transcripts.hasTones;
    final parts = size == null || size.lines == 0
        ? 0
        : (((size.chars / ContextBudget.charsPerToken) + size.lines * 8) / budget.chunkTokens).ceil().clamp(1, 1 << 30);
    return Scaffold(
      appBar: AppBar(
        title: const Text('New review'),
        actions: [
          IconButton(tooltip: 'Save as my review', icon: const Icon(Icons.bookmark_add_rounded), onPressed: _saveTemplate),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 120),
        children: [
          const SectionHeader('Look for', padding: EdgeInsets.fromLTRB(4, 8, 4, 10)),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final tp in _templates)
                GestureDetector(
                  onLongPress: tp.builtIn || tp.id == 'custom' ? null : () => _deleteTemplate(tp),
                  child: ChoiceChip(
                    label: Text(tp.name),
                    selected: _template.id == tp.id,
                    onSelected: (_) => _use(tp),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _prompt,
            minLines: 2,
            maxLines: 6,
            decoration: const InputDecoration(labelText: 'What should Gemma find?', alignLabelWithHint: true),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _format,
            minLines: 3,
            maxLines: 8,
            style: t.textTheme.bodyMedium?.copyWith(fontFamily: 'monospace'),
            decoration: InputDecoration(
              labelText: 'Answer format',
              alignLabelWithHint: true,
              helperMaxLines: 3,
              helperText: _template.kind == ReviewKind.list
                  ? 'Keep the "- [line number] "quote" — X — Y" shape: Vox counts and links every finding from it.'
                  : 'Free text. The parts are combined at the end.',
            ),
          ),
          if (_template.id == 'custom') ...[
            const SizedBox(height: 8),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Count findings'),
              subtitle: const Text('Off = a written summary instead of a list'),
              value: _template.kind == ReviewKind.list,
              onChanged: (v) => setState(() {
                _template = ReviewTemplate(
                  id: 'custom',
                  name: 'Your own',
                  prompt: _prompt.text,
                  format: v ? ReviewTemplate.listFormat('Type', 'short explanation') : 'Short bullet points.',
                  kind: v ? ReviewKind.list : ReviewKind.summary,
                );
                _format.text = _template.format;
              }),
            ),
          ],
          const SectionHeader('Read', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
          SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: false, icon: Icon(Icons.date_range_rounded), label: Text('A period')),
              ButtonSegment(value: true, icon: Icon(Icons.format_list_numbered_rounded), label: Text('Last lines')),
            ],
            selected: {_byLines},
            onSelectionChanged: (v) => setState(() => _byLines = v.first),
          ),
          if (_byLines) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final n in countChoices)
                  ChoiceChip(label: Text('Last $n'), selected: _count == n, onSelected: (_) => setState(() => _count = n)),
              ],
            ),
            if (hasTones) ...[
              const SectionHeader('Only when they sounded', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  ChoiceChip(label: const Text('Any way'), selected: _moods.isEmpty, onSelected: (_) => setState(_moods.clear)),
                  for (final tone in [...Tone.names.where((n) => n != 'neutral'), 'laughter'])
                    FilterChip(
                      label: Text('${Tone.emoji(tone)} ${Tone.label(tone)}'),
                      selected: _moods.contains(tone),
                      onSelected: (v) => setState(() => v ? _moods.add(tone) : _moods.remove(tone)),
                    ),
                ],
              ),
            ],
          ],
          if (!_byLines) const SectionHeader('When', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
          if (!_byLines)
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final (label, phrase) in kPeriodChoices)
                  ChoiceChip(label: Text(label), selected: _period == phrase, onSelected: (_) => setState(() => _period = phrase)),
                ChoiceChip(
                  avatar: const Icon(Icons.date_range_rounded, size: 18),
                  label: Text(_custom == null ? 'Pick dates…' : window?.label ?? 'Pick dates…'),
                  selected: _period == 'custom',
                  onSelected: (_) => _pickDates(),
                ),
              ],
            ),
          if (people.isNotEmpty) ...[
            const SectionHeader('Whose words', padding: EdgeInsets.fromLTRB(4, 20, 4, 10)),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ChoiceChip(label: const Text('Everyone'), selected: _focus.isEmpty, onSelected: (_) => setState(_focus.clear)),
                for (final p in people)
                  FilterChip(
                    avatar: SpeakerAvatar(label: p.name, radius: 12),
                    label: Text(p.name),
                    selected: _focus.contains(p.id),
                    onSelected: (v) => setState(() => v ? _focus.add(p.id) : _focus.remove(p.id)),
                  ),
              ],
            ),
            if (_focus.isNotEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(4, 8, 4, 0),
                child: Text(
                    _byLines
                        ? 'Gemma reads only the chosen people\'s lines, in the order they were said.'
                        : 'Gemma still reads everyone for context, but only reports what the chosen people said.',
                    style: t.textTheme.bodySmall),
              ),
          ],
          const SizedBox(height: 20),
          VoxCard(
            color: t.colorScheme.secondaryContainer,
            child: Row(
              children: [
                Icon(Icons.info_outline_rounded, color: t.colorScheme.onSecondaryContainer),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    size == null
                        ? 'Pick a period.'
                        : size.lines == 0
                            ? (_byLines ? 'No lines match.' : 'Nothing was said in ${window!.describe()}.')
                            : '${size.lines} lines${_byLines ? '' : ' in ${window!.describe()}'} → $parts part${parts == 1 ? '' : 's'} of '
                                '~${ContextBudget.linesFor(budget.chunkTokens)} lines. '
                                'About ${_minutes(parts, low: true)}–${_minutes(parts)} min on a phone.',
                    style: TextStyle(color: t.colorScheme.onSecondaryContainer),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
          child: FilledButton.icon(
            onPressed: size == null || size.lines == 0 ? null : _start,
            icon: const Icon(Icons.play_arrow_rounded),
            label: const Text('Start review'),
          ),
        ),
      ),
    );
  }

  static int _minutes(int parts, {bool low = false}) => ((parts + 1) * (low ? 10 : 30) / 60).ceil().clamp(1, 1 << 20);
}

// ---- detail -------------------------------------------------------------------

/// One review: progress, the combined answer and every finding (tap to see
/// where it was said).
class ReviewDetailScreen extends StatefulWidget {
  const ReviewDetailScreen({super.key, required this.services, required this.runId});

  final AppServices services;
  final int runId;

  @override
  State<ReviewDetailScreen> createState() => _ReviewDetailScreenState();
}

class _ReviewDetailScreenState extends State<ReviewDetailScreen> {
  Timer? _poll;
  String? _category;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _poll = Timer.periodic(const Duration(seconds: 3), (_) {
      final r = s.reviews.run(widget.runId);
      if (mounted && r != null && (r.isActive || r.status == ReviewStatus.paused)) setState(() {});
    });
  }

  @override
  void dispose() {
    _poll?.cancel();
    super.dispose();
  }

  Future<void> _open(ReviewItem it) async {
    final seg = it.segmentId == null ? null : s.transcripts.segment(it.segmentId!);
    if (seg == null) {
      showMessage(context, 'That line is no longer stored.');
      return;
    }
    await Navigator.push(
      context,
      MaterialPageRoute<void>(
        builder: (_) => ConversationScreen(services: s, conversationId: seg.conversationId, highlightSegmentId: seg.id),
      ),
    );
  }

  Future<void> _menu(String v, ReviewRun run, List<ReviewItem> items) async {
    switch (v) {
      case 'copy' || 'select':
        final b = StringBuffer()
          ..writeln('${run.title} — ${run.periodLabel}')
          ..writeln()
          ..writeln(run.finalAnswer ?? '');
        for (final it in items) {
          b.writeln('- ${it.saidAt == null ? '' : '${formatDayName(it.saidAt!)} ${formatTime(it.saidAt!)} '}'
              '${it.speaker ?? ''}: "${it.quote}" — ${it.category}${it.note.isEmpty ? '' : ' — ${it.note}'}');
        }
        final text = b.toString().trim();
        if (v == 'select') {
          await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => SelectTextScreen(title: run.title, text: text)));
          return;
        }
        await Clipboard.setData(ClipboardData(text: text));
        if (mounted) showMessage(context, 'Copied');
      case 'delete':
        if (await confirm(context, 'Delete this review?', 'Only the review is deleted, not the transcripts.')) {
          s.deleteReview(run.id);
          if (mounted) Navigator.pop(context);
        }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<int>(
      valueListenable: s.reviewVersion,
      builder: (context, _, _) {
        final run = s.reviews.run(widget.runId);
        if (run == null) {
          return Scaffold(appBar: AppBar(), body: const EmptyState(icon: Icons.fact_check_rounded, title: 'Review deleted'));
        }
        final t = Theme.of(context);
        final items = s.reviews.items(run.id);
        final chunks = s.reviews.chunks(run.id);
        final failed = chunks.where((c) => c.status == 'failed').length;
        final counts = ReviewEngine.countByCategory(items);
        final shown = _category == null ? items : items.where((i) => i.category.toLowerCase() == _category!.toLowerCase()).toList();
        return Scaffold(
          appBar: AppBar(
            title: Text(run.title),
            actions: [
              PopupMenuButton<String>(
                onSelected: (v) => _menu(v, run, items),
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'copy', child: Text('Copy results')),
                  PopupMenuItem(value: 'select', child: Text('Select text to copy')),
                  PopupMenuItem(value: 'delete', child: Text('Delete review')),
                ],
              ),
            ],
          ),
          body: ListView(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
            children: [
              VoxCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(child: Text(run.periodLabel, style: t.textTheme.titleSmall)),
                        StatusPill(run: run),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(run.prompt, maxLines: 3, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyMedium),
                    if (!run.isFinished) ...[
                      const SizedBox(height: 12),
                      LinearProgressIndicator(value: run.progress, minHeight: 8),
                      const SizedBox(height: 6),
                      Text(progressText(run), style: t.textTheme.labelLarge),
                    ],
                    if (run.error != null) ...[
                      const SizedBox(height: 8),
                      Text(run.error!, style: TextStyle(color: t.colorScheme.error)),
                    ],
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      children: [
                        if (run.isActive)
                          OutlinedButton.icon(
                            onPressed: () => s.pauseReview(run.id),
                            icon: const Icon(Icons.pause_rounded),
                            label: const Text('Pause'),
                          ),
                        if (run.status == ReviewStatus.paused)
                          FilledButton.icon(
                            onPressed: () => s.resumeReview(run.id),
                            icon: const Icon(Icons.play_arrow_rounded),
                            label: const Text('Resume'),
                          ),
                        if (!run.isFinished)
                          TextButton.icon(
                            onPressed: () => s.cancelReview(run.id),
                            icon: const Icon(Icons.stop_rounded),
                            label: const Text('Stop'),
                          ),
                        if (run.isFinished && failed > 0)
                          FilledButton.tonalIcon(
                            onPressed: () => s.retryReview(run.id),
                            icon: const Icon(Icons.refresh_rounded),
                            label: Text('Retry $failed failed part${failed == 1 ? '' : 's'}'),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              if (run.finalAnswer != null) ...[
                const SectionHeader('Result', padding: EdgeInsets.fromLTRB(4, 20, 4, 8)),
                VoxCard(child: SelectableText(run.finalAnswer!, style: t.textTheme.bodyLarge)),
              ],
              if (run.kind == ReviewKind.list && counts.isNotEmpty) ...[
                SectionHeader('Found so far · ${items.length}', padding: const EdgeInsets.fromLTRB(4, 20, 4, 8)),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    ChoiceChip(label: Text('All ${items.length}'), selected: _category == null, onSelected: (_) => setState(() => _category = null)),
                    for (final e in counts.entries)
                      ChoiceChip(
                        label: Text('${e.key} ${e.value}'),
                        selected: _category?.toLowerCase() == e.key.toLowerCase(),
                        onSelected: (_) => setState(() => _category = e.key),
                      ),
                  ],
                ),
                const SizedBox(height: 10),
                for (final it in shown) Padding(padding: const EdgeInsets.only(bottom: 8), child: _itemCard(context, it)),
              ],
              if (chunks.isNotEmpty) ...[
                const SectionHeader('Parts', padding: EdgeInsets.fromLTRB(4, 20, 4, 8)),
                VoxCard(
                  padding: EdgeInsets.zero,
                  child: Theme(
                    data: t.copyWith(dividerColor: Colors.transparent),
                    child: Column(
                      children: [
                        for (final c in chunks)
                          ExpansionTile(
                            leading: Icon(
                              switch (c.status) {
                                'done' => Icons.check_circle_outline_rounded,
                                'failed' => Icons.error_outline_rounded,
                                _ => Icons.radio_button_unchecked_rounded,
                              },
                              color: c.status == 'failed' ? t.colorScheme.error : t.colorScheme.primary,
                            ),
                            title: Text('Part ${c.idx + 1}'),
                            subtitle: Text('${formatDayName(c.firstAt)} ${formatTime(c.firstAt)} – '
                                '${formatTime(c.lastAt)} · ${c.lineCount} lines'),
                            childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                            expandedCrossAxisAlignment: CrossAxisAlignment.start,
                            children: [SelectableText(c.error ?? c.answer ?? 'Not read yet.')],
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _itemCard(BuildContext context, ReviewItem it) {
    final t = Theme.of(context);
    return VoxCard(
      onTap: it.segmentId == null ? null : () => _open(it),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: Align(alignment: Alignment.centerLeft, child: Pill(it.category, icon: Icons.label_rounded))),
              const SizedBox(width: 8),
              if (it.saidAt != null)
                Text('${formatDayName(it.saidAt!)} ${formatTime(it.saidAt!)}', style: t.textTheme.labelMedium),
            ],
          ),
          const SizedBox(height: 8),
          Text('"${it.quote}"', style: t.textTheme.bodyLarge?.copyWith(fontStyle: FontStyle.italic)),
          if (it.note.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(it.note, style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
          ],
          if (it.speaker != null) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                SpeakerAvatar(label: it.speaker!, radius: 10),
                const SizedBox(width: 6),
                Expanded(child: Text(it.speaker!, style: t.textTheme.labelLarge, maxLines: 1, overflow: TextOverflow.ellipsis)),
                if (it.segmentId != null) Icon(Icons.open_in_new_rounded, size: 16, color: t.colorScheme.primary),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
