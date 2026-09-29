import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

class _Turn {
  _Turn(this.question);

  final String question;
  final StringBuffer answer = StringBuffer();
  final List<String> tools = [];
  List<SegmentView> sources = const [];
  String? error;
  bool done = false;
}

const Map<String, String> _toolLabels = {
  'search_conversations': 'Searched conversations',
  'get_timeline': 'Read the timeline',
  'save_note': 'Saved a note',
  'list_notes': 'Checked notes',
  'set_reminder': 'Set a reminder',
  'run_automation': 'Ran an automation',
  'pause_listening': 'Paused listening',
};

/// Chat with the assistant about past conversations.
class AskScreen extends StatefulWidget {
  const AskScreen({super.key, required this.ask, required this.requests, required this.assistantReady, this.onOpenModels});

  final Stream<AnswerEvent> Function(String question) ask;
  final AssistantRequestRepository requests;
  final bool Function() assistantReady;
  final VoidCallback? onOpenModels;

  @override
  State<AskScreen> createState() => _AskScreenState();
}

class _AskScreenState extends State<AskScreen> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final List<_Turn> _turns = [];
  StreamSubscription<AnswerEvent>? _sub;

  bool get _busy => _turns.isNotEmpty && !_turns.last.done;

  @override
  void dispose() {
    unawaited(_sub?.cancel());
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _ask(String question) {
    final q = question.trim();
    if (q.isEmpty || _busy) return;
    _input.clear();
    FocusScope.of(context).unfocus();
    final turn = _Turn(q);
    setState(() => _turns.add(turn));
    _scrollDown();
    _sub = widget.ask(q).listen(
      (e) => setState(() {
        if (e.sources != null) turn.sources = e.sources!;
        if (e.token != null) turn.answer.write(e.token);
        if (e.toolName != null) turn.tools.add(_toolLabels[e.toolName] ?? e.toolName!);
        _scrollDown();
      }),
      onError: (Object e) => setState(() {
        turn
          ..error = '$e'
          ..done = true;
      }),
      onDone: () => setState(() => turn.done = true),
    );
  }

  void _scrollDown() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        unawaited(_scroll.animateTo(_scroll.position.maxScrollExtent, duration: const Duration(milliseconds: 200), curve: Curves.easeOut));
      }
    });
  }

  void _showVoiceHistory() {
    final items = widget.requests.recent(source: RequestSource.voice);
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (c) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.6,
        builder: (context, controller) => items.isEmpty
            ? const EmptyState(
                icon: Icons.record_voice_over_rounded,
                title: 'No spoken questions yet',
                message: 'Say "Hey Vox, …" while Vox is listening.',
              )
            : ListView(
                controller: controller,
                children: [
                  const SectionHeader('Asked out loud', padding: EdgeInsets.fromLTRB(20, 0, 16, 8)),
                  for (final r in items)
                    ListTile(
                      title: Text(r.text, style: const TextStyle(fontWeight: FontWeight.w600)),
                      subtitle: Text(r.answer ?? (r.status == RequestStatus.pending ? 'Waiting…' : '')),
                      trailing: Text('${formatDayName(r.createdAt)}\n${formatTime(r.createdAt)}', textAlign: TextAlign.end),
                    ),
                ],
              ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final ready = widget.assistantReady();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ask Vox'),
        actions: [
          IconButton(tooltip: 'Spoken questions', icon: const Icon(Icons.history_rounded), onPressed: _showVoiceHistory),
          if (_turns.isNotEmpty)
            IconButton(tooltip: 'New chat', icon: const Icon(Icons.add_comment_rounded), onPressed: _busy ? null : () => setState(_turns.clear)),
        ],
      ),
      body: Column(
        children: [
          if (!ready)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: VoxCard(
                color: Theme.of(context).colorScheme.secondaryContainer,
                padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
                child: Row(
                  children: [
                    const Icon(Icons.auto_awesome_rounded),
                    const SizedBox(width: 12),
                    const Expanded(child: Text('Download Gemma 4 to get answers. Search works on the Timeline tab.')),
                    TextButton(onPressed: widget.onOpenModels, child: const Text('Models')),
                  ],
                ),
              ),
            ),
          Expanded(
            child: _turns.isEmpty
                ? _suggestions(context)
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _turnView(context, _turns[i]),
                  ),
          ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 6, 8, 10),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      minLines: 1,
                      maxLines: 4,
                      textInputAction: TextInputAction.send,
                      decoration: const InputDecoration(hintText: 'What did we say about…'),
                      onSubmitted: _ask,
                    ),
                  ),
                  const SizedBox(width: 6),
                  IconButton.filled(
                    tooltip: 'Ask',
                    iconSize: 22,
                    onPressed: _busy ? null : () => _ask(_input.text),
                    icon: const Icon(Icons.arrow_upward_rounded),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _suggestions(BuildContext context) {
    final t = Theme.of(context);
    const prompts = [
      ('Summarise today', Icons.today_rounded),
      ('What did we talk about this week?', Icons.date_range_rounded),
      ('Recap last week', Icons.history_edu_rounded),
      ('What needs to be bought?', Icons.shopping_cart_rounded),
      ('What did we plan for the weekend?', Icons.event_rounded),
      ('Remind me in 30 minutes to check the oven', Icons.alarm_rounded),
    ];
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
      children: [
        Text('Ask about anything said at home', style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        Text('Vox reads your transcripts on this phone. It understands times like "yesterday", "last week" or "on Monday".',
            style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
        const SizedBox(height: 20),
        for (final (text, icon) in prompts)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: VoxCard(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              onTap: () => _ask(text),
              child: Row(children: [Icon(icon, color: t.colorScheme.primary), const SizedBox(width: 12), Expanded(child: Text(text))]),
            ),
          ),
      ],
    );
  }

  Widget _turnView(BuildContext context, _Turn turn) {
    final t = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: Container(
            constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.8),
            margin: const EdgeInsets.only(top: 12, bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: t.colorScheme.primary,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(20),
                topRight: Radius.circular(20),
                bottomLeft: Radius.circular(20),
                bottomRight: Radius.circular(6),
              ),
            ),
            child: Text(turn.question, style: TextStyle(color: t.colorScheme.onPrimary, fontSize: 16)),
          ),
        ),
        VoxCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (turn.tools.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: [for (final tool in turn.tools) Pill(tool, icon: Icons.bolt_rounded, color: const Color(0xFF9C36B5))],
                  ),
                ),
              if (turn.error != null)
                Text(turn.error!, style: TextStyle(color: t.colorScheme.error))
              else if (turn.answer.isEmpty && !turn.done)
                const Row(children: [
                  SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                  SizedBox(width: 10),
                  Text('Thinking…'),
                ])
              else
                SelectableText(turn.answer.toString().trim(), style: t.textTheme.bodyLarge),
              if (turn.sources.isNotEmpty)
                Theme(
                  data: t.copyWith(dividerColor: Colors.transparent),
                  child: ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    childrenPadding: EdgeInsets.zero,
                    title: Text('Based on ${turn.sources.length} thing${turn.sources.length == 1 ? '' : 's'} said',
                        style: t.textTheme.labelLarge?.copyWith(color: t.colorScheme.primary)),
                    children: [
                      for (final s in turn.sources)
                        ListTile(
                          dense: true,
                          contentPadding: EdgeInsets.zero,
                          leading: SpeakerAvatar(label: s.speakerLabel, known: s.isKnownSpeaker, radius: 14),
                          title: Text(s.text),
                          subtitle: Text('${s.speakerLabel} · ${formatDayName(s.startedAt)} ${formatTime(s.startedAt)}'),
                        ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}
