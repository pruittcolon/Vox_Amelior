import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// One conversation as a chat, with who said what.
class ConversationScreen extends StatefulWidget {
  const ConversationScreen({super.key, required this.services, required this.conversationId, this.highlightSegmentId});

  final AppServices services;
  final int conversationId;
  final int? highlightSegmentId;

  @override
  State<ConversationScreen> createState() => _ConversationScreenState();
}

class _ConversationScreenState extends State<ConversationScreen> {
  List<SegmentView> _lines = const [];
  final _highlightKey = GlobalKey();

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _load();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ctx = _highlightKey.currentContext;
      if (ctx != null) Scrollable.ensureVisible(ctx, alignment: 0.3);
    });
  }

  void _load() => setState(() => _lines = s.transcripts.conversation(widget.conversationId));

  Future<void> _relabel(SegmentView seg) async {
    final people = s.speakers.profiles();
    final choice = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (c) => SafeArea(
        child: ListView(
          shrinkWrap: true,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              child: Text('Who said this?', style: Theme.of(c).textTheme.titleMedium),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              child: Text('"${seg.text}"', maxLines: 3, overflow: TextOverflow.ellipsis),
            ),
            for (final p in people)
              ListTile(
                leading: SpeakerAvatar(label: p.name),
                title: Text(p.name),
                trailing: p.id == seg.speakerId ? const Icon(Icons.check_rounded) : null,
                onTap: () => Navigator.pop(c, p.id),
              ),
            if (people.isEmpty) const ListTile(title: Text('Add people on the People tab to label voices.')),
            ListTile(
              leading: const Icon(Icons.copy_rounded),
              title: const Text('Copy this line'),
              onTap: () => Navigator.pop(c, '#copy'),
            ),
          ],
        ),
      ),
    );
    if (choice == null || !mounted) return;
    if (choice == '#copy') {
      await Clipboard.setData(ClipboardData(text: seg.text));
      if (mounted) showMessage(context, 'Copied');
      return;
    }
    s.speakers.assignSegmentToSpeaker(seg.id, choice);
    s.dataChanged();
    _load();
    if (mounted) showMessage(context, 'Thanks — Vox will recognise this voice better.');
  }

  Future<void> _menu(String action) async {
    switch (action) {
      case 'copy':
        final text = _lines.map((l) => '${formatTime(l.startedAt)} ${l.speakerLabel}: ${l.text}').join('\n');
        await Clipboard.setData(ClipboardData(text: text));
        if (mounted) showMessage(context, 'Conversation copied');
      case 'delete':
        if (await confirm(context, 'Delete this conversation?', 'It is removed from this phone for good.')) {
          s.transcripts.deleteConversation(widget.conversationId);
          s.dataChanged();
          if (mounted) Navigator.pop(context);
        }
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final first = _lines.firstOrNull;
    final last = _lines.lastOrNull;
    final people = <String>{for (final l in _lines) l.speakerLabel}.toList();
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
                for (var i = 0; i < _lines.length; i++) _bubble(context, _lines[i], showName: i == 0 || _lines[i - 1].speakerLabel != _lines[i].speakerLabel),
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
                        Text(formatTime(seg.startedAt), style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
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
