import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

class _Turn {
  _Turn(this.question);

  final String question;
  final StringBuffer answer = StringBuffer();
  List<SegmentView> sources = const [];
  String? error;
  bool done = false;
}

/// Ask questions about past conversations; answered by Gemma on the phone.
class AskScreen extends StatefulWidget {
  const AskScreen({
    super.key,
    required this.assistant,
    required this.requests,
    required this.assistantReady,
    this.onOpenSetup,
  });

  final AssistantService assistant;
  final AssistantRequestRepository requests;
  final bool Function() assistantReady;
  final VoidCallback? onOpenSetup;

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
    final turn = _Turn(q);
    setState(() => _turns.add(turn));
    _sub = widget.assistant.ask(q).listen(
      (e) => setState(() {
        if (e.sources != null) turn.sources = e.sources!;
        if (e.token != null) turn.answer.write(e.token);
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

  void _summarizeToday() {
    if (_busy) return;
    final now = DateTime.now();
    final turn = _Turn('Summarise today');
    setState(() => _turns.add(turn));
    final start = DateTime(now.year, now.month, now.day);
    widget.assistant.summarize(start, start.add(const Duration(days: 1)), label: 'today').listen(
          (t) => setState(() {
            turn.answer.write(t);
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
      if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  void _showVoiceHistory() {
    final items = widget.requests.recent();
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (c) => items.isEmpty
          ? const Padding(
              padding: EdgeInsets.all(24),
              child: Text('No voice questions yet. Say "Hey Vox, …" while Vox is listening.'),
            )
          : ListView(
              children: [
                for (final r in items)
                  ListTile(
                    title: Text(r.text),
                    subtitle: Text(r.answer ?? (r.status == RequestStatus.pending ? 'Waiting…' : '')),
                    trailing: Text(formatTime(r.createdAt)),
                  ),
              ],
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
          IconButton(tooltip: 'Voice questions', icon: const Icon(Icons.record_voice_over), onPressed: _showVoiceHistory),
        ],
      ),
      body: Column(
        children: [
          if (!ready)
            MaterialBanner(
              content: const Text('Download Gemma to get answers. Search still works on the Live tab.'),
              actions: [TextButton(onPressed: widget.onOpenSetup, child: const Text('Set up'))],
            ),
          Expanded(
            child: _turns.isEmpty
                ? _suggestions(context)
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(16),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _turnView(context, _turns[i]),
                  ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      minLines: 1,
                      maxLines: 4,
                      textInputAction: TextInputAction.send,
                      decoration: const InputDecoration(
                        hintText: 'What did we say about…',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: _ask,
                    ),
                  ),
                  IconButton.filled(
                    tooltip: 'Ask',
                    onPressed: _busy ? null : () => _ask(_input.text),
                    icon: const Icon(Icons.send),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _suggestions(BuildContext context) => ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Text('Ask about anything said at home.', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 16),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ActionChip(label: const Text('Summarise today'), onPressed: _summarizeToday),
              for (final q in const [
                'What did we plan for this week?',
                'What needs to be bought?',
                'What did we decide yesterday?',
              ])
                ActionChip(label: Text(q), onPressed: () => _ask(q)),
            ],
          ),
        ],
      );

  Widget _turnView(BuildContext context, _Turn t) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: Card(
            color: theme.colorScheme.primaryContainer,
            child: Padding(padding: const EdgeInsets.all(12), child: Text(t.question)),
          ),
        ),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (t.error != null)
                  Text(t.error!, style: TextStyle(color: theme.colorScheme.error))
                else if (t.answer.isEmpty && !t.done)
                  const Row(children: [
                    SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                    SizedBox(width: 8),
                    Text('Thinking…'),
                  ])
                else
                  SelectableText(t.answer.toString().trim()),
                if (t.sources.isNotEmpty)
                  ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    title: Text('Based on ${t.sources.length} things said', style: theme.textTheme.bodySmall),
                    children: [
                      for (final s in t.sources)
                        ListTile(
                          dense: true,
                          title: Text(s.text),
                          subtitle: Text('${s.speakerLabel} · ${formatDay(s.startedAt)} ${formatTime(s.startedAt)}'),
                        ),
                    ],
                  ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}
