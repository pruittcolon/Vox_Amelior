import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/ui/charts.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/name_voice.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Opens "Who's this?" for every voice without a name. Its sample lines
/// open in their conversation.
Future<void> openNameVoices(BuildContext context, AppServices s) => Navigator.push(
      context,
      MaterialPageRoute<void>(
        builder: (_) => NameVoicesScreen(
          services: s,
          openLine: (c, line) => Navigator.push(
            c,
            MaterialPageRoute<void>(
              builder: (_) => ConversationScreen(services: s, conversationId: line.conversationId, highlightSegmentId: line.id),
            ),
          ),
        ),
      ),
    );

/// One conversation as a chat, with who said what.
class ConversationScreen extends StatefulWidget {
  const ConversationScreen({
    super.key,
    required this.services,
    required this.conversationId,
    this.highlightSegmentId,
    this.focusSpeakerIds = const {},
  });

  final AppServices services;
  final int conversationId;
  final int? highlightSegmentId;

  /// Start showing only these people's lines (e.g. picked on the Timeline).
  final Set<String> focusSpeakerIds;

  @override
  State<ConversationScreen> createState() => _ConversationScreenState();
}

class _ConversationScreenState extends State<ConversationScreen> {
  List<SegmentView> _lines = const [];

  /// Voices to show ([SegmentView.voiceKey]); null shows everyone.
  Set<String>? _only;

  /// Show lines from voices marked as TV / background.
  bool _showBackground = false;

  /// Tones to show ([SegmentView.emotion] or [SegmentView.sound]); empty shows all.
  final Set<String> _tones = {};

  AppServices get s => widget.services;

  /// Lines after the voice filters.
  List<SegmentView> get _visible => [
        for (final l in _lines)
          if ((_showBackground || !l.background || l.id == widget.highlightSegmentId) &&
              (_only == null || _only!.contains(l.voiceKey) || l.id == widget.highlightSegmentId) &&
              (_tones.isEmpty || _tones.contains(l.emotion) || _tones.contains(l.sound) || l.id == widget.highlightSegmentId))
            l,
      ];

  /// One line as text: "20:31 Ericah [angry]: ...".
  static String lineText(SegmentView l) {
    final tone = [?l.emotion, ?l.sound].join(', ');
    return '${formatTime(l.startedAt)} ${l.speakerLabel}${tone.isEmpty ? '' : ' [$tone]'}: ${l.text}';
  }

  Future<void> _copyLine(SegmentView seg) async {
    await Clipboard.setData(ClipboardData(text: lineText(seg)));
    if (mounted) showMessage(context, 'Line copied');
  }

  void _toggleVoice(String key) => setState(() {
        final next = {...?_only};
        if (!next.remove(key)) next.add(key);
        _only = next.isEmpty ? null : next;
      });

  @override
  void initState() {
    super.initState();
    if (widget.focusSpeakerIds.isNotEmpty) _only = {for (final id in widget.focusSpeakerIds) 'person:$id'};
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
    setState(() => _lines = s.transcripts.conversation(widget.conversationId));
  }

  /// Names a guest voice: every line it said, here and in other conversations.
  Future<void> _nameVoice(String clusterId) async {
    if (await nameVoice(context, s, clusterId)) _load();
  }

  /// One line: who said it (one tap on a person), TV / background, copy.
  Future<void> _lineActions(SegmentView seg) async {
    final people = s.speakers.profiles();
    final current = seg.speakerId == null ? null : seg.speakerName;
    final guest = seg.speakerId == null && !seg.background ? seg.clusterId : null;
    final guestLines = guest == null ? 0 : s.transcripts.voiceToName(guest, samples: 0)?.lines ?? 0;
    final choice = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (c) {
        final t = Theme.of(c);
        return SafeArea(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(c).height * 0.85),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text('Who said this?', style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  Text('"${seg.text}"', maxLines: 3, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyLarge?.copyWith(fontStyle: FontStyle.italic)),
                  const SizedBox(height: 14),
                  if (guest != null) ...[
                    // The quick way: the whole voice at once.
                    Material(
                      color: t.colorScheme.tertiaryContainer,
                      borderRadius: BorderRadius.circular(16),
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                SpeakerAvatar(label: seg.speakerLabel, known: false),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    '${seg.speakerLabel} said ${guestLines == 1 ? 'only this line' : '$guestLines lines in all'}. '
                                    'Name the voice to name every one.',
                                    style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onTertiaryContainer),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            Align(
                              alignment: Alignment.centerRight,
                              child: FilledButton.icon(
                                onPressed: () => Navigator.pop(c, '#voice'),
                                icon: const Icon(Icons.record_voice_over_rounded),
                                label: const Text('Name this voice'),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),
                  ],
                  Text(
                    guest != null
                        ? 'Or just this line is…'
                        : current != null
                            ? 'Said by $current. If not, it\'s…'
                            : 'It\'s…',
                    style: t.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      for (final p in people)
                        if (p.id != seg.speakerId)
                          ActionChip(
                            avatar: SpeakerAvatar(label: p.name, radius: 12),
                            label: Text(p.name),
                            onPressed: () => Navigator.pop(c, p.id),
                          ),
                      ActionChip(
                        avatar: Icon(Icons.person_add_alt_1_rounded, size: 18, color: t.colorScheme.primary),
                        label: const Text('New person…'),
                        onPressed: () => Navigator.pop(c, '#new'),
                      ),
                      if (current != null)
                        ActionChip(
                          avatar: Icon(Icons.person_off_rounded, size: 18, color: t.colorScheme.error),
                          label: Text('Not $current'),
                          onPressed: () => Navigator.pop(c, '#not'),
                        ),
                      if (seg.background)
                        ActionChip(
                          avatar: const Icon(Icons.record_voice_over_rounded, size: 18),
                          label: const Text('Not TV or background'),
                          onPressed: () => Navigator.pop(c, '#unbackground'),
                        )
                      else
                        ActionChip(
                          avatar: const Icon(Icons.tv_rounded, size: 18),
                          label: const Text('TV or background'),
                          onPressed: () => Navigator.pop(c, '#background'),
                        ),
                    ],
                  ),
                  if (current != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        '"Not $current" makes the line a guest; $current\'s voiceprint stays the same.',
                        style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
                      ),
                    ),
                  const Divider(height: 28),
                  Wrap(
                    spacing: 8,
                    children: [
                      TextButton.icon(
                        onPressed: () => Navigator.pop(c, '#copy'),
                        icon: const Icon(Icons.copy_rounded),
                        label: const Text('Copy line'),
                      ),
                      TextButton.icon(
                        onPressed: () => Navigator.pop(c, '#copyFull'),
                        icon: const Icon(Icons.content_copy_rounded),
                        label: const Text('Copy with name and time'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
    if (choice == null || !mounted) return;
    try {
      switch (choice) {
        case '#voice':
          await _nameVoice(guest!);
          return;
        case '#copy':
          await Clipboard.setData(ClipboardData(text: seg.text));
          if (mounted) showMessage(context, 'Copied');
          return;
        case '#copyFull':
          await _copyLine(seg);
          return;
        case '#not':
          final g = s.speakers.markNotSpeaker(seg.id, clusterThreshold: s.settings.value.guestThreshold);
          if (mounted) showMessage(context, g == null ? 'Removed $current from this line.' : 'Not $current — now $g.');
        case '#background':
          final label = s.speakers.markBackground(seg.id, clusterThreshold: s.settings.value.guestThreshold);
          if (mounted) {
            showMessage(context, label == null ? 'This line is too short to recognise a voice from.' : '$label is now hidden as TV / background.');
          }
          if (label == null) return;
        case '#unbackground':
          if (seg.clusterId != null) s.speakers.setClusterBackground(seg.clusterId!, false);
          if (mounted) showMessage(context, '${seg.speakerLabel} is shown again.');
        case '#new':
          final name = await askText(context, 'Who is this?', hint: 'Name');
          if (name == null || name.isEmpty) return;
          s.speakers.createFromSegment(seg.id, name, embeddingModel: SherpaSpeakerEmbedder.modelIdConst);
          if (mounted) showMessage(context, 'Added $name. Vox will recognise this voice from now on.');
        default:
          s.speakers.assignSegmentToSpeaker(seg.id, choice);
          if (mounted) showMessage(context, 'Thanks — Vox will recognise this voice better.');
      }
    } on StateError catch (e) {
      if (mounted) showMessage(context, e.message);
      return;
    }
    s.dataChanged();
    _load();
  }

  Future<void> _menu(String action) async {
    switch (action) {
      case 'copy':
        await Clipboard.setData(ClipboardData(text: _visible.map(lineText).join('\n')));
        if (mounted) showMessage(context, _visible.length == _lines.length ? 'Conversation copied' : 'Copied the ${_visible.length} lines shown');
      case 'select':
        await Navigator.push(
          context,
          MaterialPageRoute<void>(builder: (_) => SelectTextScreen(title: 'Select text', text: _visible.map(lineText).join('\n'))),
        );
      case 'delete':
        if (await confirm(context, 'Delete this conversation?', 'It is removed from this phone for good.')) {
          s.deleteConversation(widget.conversationId);
          if (mounted) Navigator.pop(context);
        }
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final first = _lines.firstOrNull;
    final last = _lines.lastOrNull;
    final visible = _visible;
    final people = <String>{for (final l in _lines) if (!l.background) l.speakerLabel}.toList();
    final mood = <String, int>{};
    for (final l in _lines) {
      if (!l.background && l.emotion != null) mood[l.emotion!] = (mood[l.emotion!] ?? 0) + 1;
    }
    final hasMood = mood.keys.any((k) => k != 'neutral');
    // Everything above the lines, then one row per line; only what is on
    // screen is built, so long conversations open at once.
    final header = <Widget>[
      if (first != null)
        Padding(
          padding: EdgeInsets.only(bottom: hasMood ? 8 : 12),
          child: Row(
            children: [
              AvatarStack(labels: people),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  '${formatTime(first.startedAt)} – ${formatTime(last!.endedAt)} · ${people.join(', ')}',
                  style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
                ),
              ),
            ],
          ),
        ),
      if (hasMood) Padding(padding: const EdgeInsets.only(bottom: 12), child: MoodStrip(counts: mood)),
      _voicesToName(context),
      _voiceFilter(context),
      if (visible.isEmpty)
        const Padding(
          padding: EdgeInsets.symmetric(vertical: 32),
          child: Center(child: Text('No lines from the chosen voices.')),
        ),
    ];
    Widget row(int i) {
      if (i < header.length) return header[i];
      final j = i - header.length;
      return _bubble(context, visible[j], showName: j == 0 || visible[j - 1].speakerLabel != visible[j].speakerLabel);
    }

    final count = header.length + visible.length;
    // A line to show (from search or a review) starts a quarter down the
    // screen: the lines before it are laid out upwards from there.
    final h = visible.indexWhere((l) => l.id == widget.highlightSegmentId);
    final center = h < 0 ? 0 : header.length + h;
    const pad = EdgeInsets.symmetric(horizontal: 16);
    return Scaffold(
      appBar: AppBar(
        title: Text(first == null ? 'Conversation' : formatDayName(first.startedAt)),
        actions: [
          PopupMenuButton<String>(
            onSelected: _menu,
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'copy', child: Text('Copy conversation')),
              PopupMenuItem(value: 'select', child: Text('Select text to copy')),
              PopupMenuItem(value: 'delete', child: Text('Delete conversation')),
            ],
          ),
        ],
      ),
      body: _lines.isEmpty
          ? const EmptyState(icon: Icons.chat_bubble_outline_rounded, title: 'This conversation is empty')
          : CustomScrollView(
              key: const ValueKey('conversation-lines'),
              center: const ValueKey('conversation-center'),
              anchor: center == 0 ? 0.0 : 0.25,
              slivers: [
                SliverPadding(
                  padding: pad,
                  sliver: SliverList.builder(itemCount: center, itemBuilder: (context, i) => row(center - 1 - i)),
                ),
                SliverPadding(
                  key: const ValueKey('conversation-center'),
                  padding: pad.copyWith(bottom: 32),
                  sliver: SliverList.builder(itemCount: count - center, itemBuilder: (context, i) => row(center + i)),
                ),
              ],
            ),
    );
  }

  /// Guest voices in this conversation, each a tap away from a name.
  Widget _voicesToName(BuildContext context) {
    final t = Theme.of(context);
    final guests = <String, ({String label, int n})>{};
    for (final l in _lines) {
      if (l.speakerId != null || l.clusterId == null || l.background) continue;
      final g = guests[l.clusterId!];
      guests[l.clusterId!] = (label: l.speakerLabel, n: (g?.n ?? 0) + 1);
    }
    if (guests.isEmpty) return const SizedBox.shrink();
    final sorted = guests.entries.toList()..sort((a, b) => b.value.n.compareTo(a.value.n));
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Material(
        color: t.colorScheme.tertiaryContainer,
        borderRadius: BorderRadius.circular(18),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 10, 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                sorted.length == 1 ? 'A voice without a name' : '${sorted.length} voices without a name',
                style: t.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800, color: t.colorScheme.onTertiaryContainer),
              ),
              for (final g in sorted)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Row(
                    children: [
                      SpeakerAvatar(label: g.value.label, known: false, radius: 16),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          '${g.value.label} · ${g.value.n == 1 ? '1 line' : '${g.value.n} lines'} here',
                          style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onTertiaryContainer),
                        ),
                      ),
                      FilledButton.tonal(onPressed: () => _nameVoice(g.key), child: const Text("Who's this?")),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  /// Chips to show only some voices, and to show or hide TV / background.
  Widget _voiceFilter(BuildContext context) {
    final t = Theme.of(context);
    final counts = <String, ({String label, bool known, int n})>{};
    var background = 0;
    for (final l in _lines) {
      if (l.background) {
        background++;
        continue;
      }
      final c = counts[l.voiceKey];
      counts[l.voiceKey] = (label: l.speakerLabel, known: l.isKnownSpeaker, n: (c?.n ?? 0) + 1);
    }
    final voices = counts.entries.toList()
      ..sort((a, b) {
        if (a.value.known != b.value.known) return a.value.known ? -1 : 1;
        return b.value.n.compareTo(a.value.n);
      });
    final toneCounts = <String, int>{};
    for (final l in _lines) {
      if (l.background) continue;
      for (final tone in [?l.emotion, ?l.sound]) {
        if (tone != 'neutral') toneCounts[tone] = (toneCounts[tone] ?? 0) + 1;
      }
    }
    // A picked tone stays offered (even at 0) so it can always be un-picked.
    for (final tone in _tones) {
      toneCounts.putIfAbsent(tone, () => 0);
    }
    final tones = toneCounts.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    // One voice and nothing hidden: choosing voices would change nothing.
    final showVoices = voices.length > 1 || background > 0 || _only != null;
    if (!showVoices && tones.isEmpty) return const SizedBox.shrink();
    final shown = _visible.length;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (showVoices)
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                ChoiceChip(
                  label: const Text('Everyone'),
                  selected: _only == null,
                  onSelected: (_) => setState(() => _only = null),
                ),
                for (final v in voices)
                  FilterChip(
                    avatar: v.value.known ? null : const Icon(Icons.person_outline_rounded, size: 16),
                    label: Text('${v.value.label} · ${v.value.n}'),
                    selected: _only?.contains(v.key) ?? false,
                    onSelected: (_) => _toggleVoice(v.key),
                  ),
                if (background > 0)
                  FilterChip(
                    avatar: const Icon(Icons.tv_rounded, size: 16),
                    label: Text('TV / background · $background'),
                    selected: _showBackground,
                    onSelected: (v) => setState(() => _showBackground = v),
                  ),
              ],
            ),
          if (tones.isNotEmpty)
            Padding(
              padding: EdgeInsets.only(top: showVoices ? 6 : 0),
              child: Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (final e in tones)
                    FilterChip(
                      label: Text('${Tone.emoji(e.key)} ${Tone.label(e.key)} · ${e.value}'),
                      selected: _tones.contains(e.key),
                      onSelected: (v) => setState(() => v ? _tones.add(e.key) : _tones.remove(e.key)),
                    ),
                ],
              ),
            ),
          if (shown != _lines.length)
            Padding(
              padding: const EdgeInsets.only(top: 6, left: 4),
              child: Text(
                'Showing $shown of ${_lines.length} lines',
                style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
            ),
        ],
      ),
    );
  }

  Widget _bubble(BuildContext context, SegmentView seg, {required bool showName}) {
    final t = Theme.of(context);
    final color = speakerColor(seg.speakerLabel, known: seg.isKnownSpeaker);
    final highlighted = seg.id == widget.highlightSegmentId;
    final guest = seg.speakerId == null && seg.clusterId != null && !seg.background;
    return Padding(
      padding: EdgeInsets.only(top: showName ? 12 : 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: 40, child: showName ? SpeakerAvatar(label: seg.speakerLabel, known: seg.isKnownSpeaker) : null),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (showName && guest)
                  // A guest's name is a button: naming it names all its lines.
                  InkWell(
                    borderRadius: BorderRadius.circular(8),
                    onTap: () => _nameVoice(seg.clusterId!),
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(4, 2, 4, 4),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(seg.speakerLabel, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
                          const SizedBox(width: 6),
                          Icon(Icons.edit_rounded, size: 14, color: t.colorScheme.primary),
                          const SizedBox(width: 2),
                          Text("Who's this?", style: t.textTheme.labelMedium?.copyWith(color: t.colorScheme.primary, fontWeight: FontWeight.w700)),
                        ],
                      ),
                    ),
                  )
                else if (showName)
                  Padding(
                    padding: const EdgeInsets.only(left: 4, bottom: 4),
                    child: Text(seg.speakerLabel, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
                  ),
                GestureDetector(
                  onTap: () => _lineActions(seg),
                  onLongPress: () => _copyLine(seg),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: highlighted ? t.colorScheme.tertiaryContainer : color.withValues(alpha: 0.10),
                      borderRadius: const BorderRadius.only(
                        topRight: Radius.circular(18),
                        bottomLeft: Radius.circular(18),
                        bottomRight: Radius.circular(18),
                        topLeft: Radius.circular(6),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(seg.text, style: t.textTheme.bodyLarge),
                        const SizedBox(height: 4),
                        // Wraps onto a second line on narrow screens or with large text.
                        Wrap(
                          spacing: 8,
                          runSpacing: 2,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          children: [
                            Text(formatTime(seg.startedAt), style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                            for (final tone in [?seg.emotion, ?seg.sound])
                              if (tone != 'neutral')
                                Text('${Tone.emoji(tone)} ${Tone.label(tone)}',
                                    style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                            if (seg.overlap)
                              Tooltip(
                                message: 'Someone else was talking at the same time',
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.forum_rounded, size: 14, color: t.colorScheme.tertiary),
                                    const SizedBox(width: 3),
                                    Text('talking at once', style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.tertiary)),
                                  ],
                                ),
                              ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Shows text that can be selected and copied in parts (long-press a word,
/// drag the handles, then Copy), with a button to copy all of it.
class SelectTextScreen extends StatelessWidget {
  const SelectTextScreen({super.key, required this.title, required this.text});

  final String title;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        actions: [
          IconButton(
            tooltip: 'Copy all',
            icon: const Icon(Icons.copy_all_rounded),
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: text));
              if (context.mounted) showMessage(context, 'Copied');
            },
          ),
        ],
      ),
      body: SelectionArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
          child: Text(text, style: Theme.of(context).textTheme.bodyLarge),
        ),
      ),
    );
  }
}
