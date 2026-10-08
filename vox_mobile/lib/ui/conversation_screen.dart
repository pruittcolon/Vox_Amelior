import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

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
  final _highlightKey = GlobalKey();

  /// Voices to show ([SegmentView.voiceKey]); null shows everyone.
  Set<String>? _only;

  /// Show lines from voices marked as TV / background.
  bool _showBackground = false;

  AppServices get s => widget.services;

  /// Lines after the voice filters.
  List<SegmentView> get _visible => [
        for (final l in _lines)
          if ((_showBackground || !l.background || l.id == widget.highlightSegmentId) &&
              (_only == null || _only!.contains(l.voiceKey) || l.id == widget.highlightSegmentId))
            l,
      ];

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
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ctx = _highlightKey.currentContext;
      if (ctx != null) Scrollable.ensureVisible(ctx, alignment: 0.3);
    });
  }

  void _load() => setState(() => _lines = s.transcripts.conversation(widget.conversationId));

  Future<void> _relabel(SegmentView seg) async {
    final people = s.speakers.profiles();
    final current = seg.speakerId == null ? null : seg.speakerName;
    final choice = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      builder: (c) {
        final t = Theme.of(c);
        return SafeArea(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(c).height * 0.8),
            child: ListView(
              shrinkWrap: true,
              padding: const EdgeInsets.only(bottom: 8),
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 4),
                  child: Text('Who said this?', style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
                  child: Text('"${seg.text}"',
                      maxLines: 3, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyMedium?.copyWith(fontStyle: FontStyle.italic)),
                ),
                for (final p in people)
                  ListTile(
                    leading: SpeakerAvatar(label: p.name),
                    title: Text(p.name),
                    trailing: p.id == seg.speakerId ? Icon(Icons.check_rounded, color: t.colorScheme.primary) : null,
                    onTap: () => Navigator.pop(c, p.id),
                  ),
                if (people.isEmpty) const ListTile(title: Text('Add people on the People tab to label voices.')),
                const Divider(indent: 16, endIndent: 16),
                if (current != null)
                  ListTile(
                    leading: Icon(Icons.person_off_rounded, color: t.colorScheme.error),
                    title: Text('Not $current'),
                    subtitle: Text('Becomes a guest; similar voices won\'t get this name. $current\'s voiceprint stays the same.'),
                    onTap: () => Navigator.pop(c, '#not'),
                  ),
                if (seg.background)
                  ListTile(
                    leading: const Icon(Icons.record_voice_over_rounded),
                    title: const Text('Not TV or background'),
                    subtitle: Text('Show ${seg.speakerLabel}\'s lines again everywhere.'),
                    onTap: () => Navigator.pop(c, '#unbackground'),
                  )
                else
                  ListTile(
                    leading: const Icon(Icons.tv_rounded),
                    title: const Text('TV or background voice'),
                    subtitle: const Text('Hide this voice\'s lines, now and whenever it is heard again. You can show them any time.'),
                    onTap: () => Navigator.pop(c, '#background'),
                  ),
                ListTile(
                  leading: const Icon(Icons.person_add_rounded),
                  title: const Text('New person…'),
                  subtitle: const Text('Name someone new from this line'),
                  onTap: () => Navigator.pop(c, '#new'),
                ),
                ListTile(
                  leading: const Icon(Icons.copy_rounded),
                  title: const Text('Copy this line'),
                  onTap: () => Navigator.pop(c, '#copy'),
                ),
              ],
            ),
          ),
        );
      },
    );
    if (choice == null || !mounted) return;
    try {
      switch (choice) {
        case '#copy':
          await Clipboard.setData(ClipboardData(text: seg.text));
          if (mounted) showMessage(context, 'Copied');
          return;
        case '#not':
          final guest = s.speakers.markNotSpeaker(seg.id, clusterThreshold: s.settings.value.guestThreshold);
          if (mounted) showMessage(context, guest == null ? 'Removed $current from this line.' : 'Not $current — now $guest.');
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
        final text = _visible.map((l) => '${formatTime(l.startedAt)} ${l.speakerLabel}: ${l.text}').join('\n');
        await Clipboard.setData(ClipboardData(text: text));
        if (mounted) showMessage(context, 'Conversation copied');
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
    return Scaffold(
      appBar: AppBar(
        title: Text(first == null ? 'Conversation' : formatDayName(first.startedAt)),
        actions: [
          PopupMenuButton<String>(
            onSelected: _menu,
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'copy', child: Text('Copy conversation')),
              PopupMenuItem(value: 'delete', child: Text('Delete conversation')),
            ],
          ),
        ],
      ),
      body: _lines.isEmpty
          ? const EmptyState(icon: Icons.chat_bubble_outline_rounded, title: 'This conversation is empty')
          : ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
              children: [
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Row(
                    children: [
                      AvatarStack(labels: people),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          '${formatTime(first!.startedAt)} – ${formatTime(last!.endedAt)} · ${people.join(', ')}',
                          style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
                        ),
                      ),
                    ],
                  ),
                ),
                _voiceFilter(context),
                if (visible.isEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 32),
                    child: Center(child: Text('No lines from the chosen voices.')),
                  ),
                for (var i = 0; i < visible.length; i++)
                  _bubble(context, visible[i], showName: i == 0 || visible[i - 1].speakerLabel != visible[i].speakerLabel),
              ],
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
    if (voices.length < 2 && background == 0) return const SizedBox.shrink();
    final shown = _visible.length;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
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
    return Padding(
      key: highlighted ? _highlightKey : null,
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
                if (showName)
                  Padding(
                    padding: const EdgeInsets.only(left: 4, bottom: 4),
                    child: Text(seg.speakerLabel, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
                  ),
                GestureDetector(
                  onTap: () => _relabel(seg),
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
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(formatTime(seg.startedAt), style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                            if (seg.overlap) ...[
                              const SizedBox(width: 8),
                              Tooltip(
                                message: 'Someone else was talking at the same time',
                                child: Icon(Icons.forum_rounded, size: 14, color: t.colorScheme.tertiary),
                              ),
                              const SizedBox(width: 3),
                              Flexible(
                                child: Text('talking at once',
                                    overflow: TextOverflow.ellipsis,
                                    style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.tertiary)),
                              ),
                            ],
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
