import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Asks Gemma [question] from a search and shows the answer in a sheet, with
/// the lines it read (found by words and by meaning). Kept in Saved answers.
Future<void> showSearchAnswer(BuildContext context, AppServices s, String question) {
  if (!s.assistantReady) {
    showMessage(context, 'Download Gemma (More → Models) to get answers.');
    return Future.value();
  }
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    builder: (_) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.7,
      maxChildSize: 0.95,
      builder: (context, controller) => SearchAnswer(services: s, question: question, controller: controller),
    ),
  );
}

class SearchAnswer extends StatefulWidget {
  const SearchAnswer({super.key, required this.services, required this.question, this.controller});

  final AppServices services;
  final String question;
  final ScrollController? controller;

  @override
  State<SearchAnswer> createState() => _SearchAnswerState();
}

class _SearchAnswerState extends State<SearchAnswer> {
  final _answer = StringBuffer();
  List<SegmentView> _sources = const [];
  StreamSubscription<AnswerEvent>? _sub;
  String? _error;
  bool _done = false;

  @override
  void initState() {
    super.initState();
    _sub = widget.services.assistant.ask(widget.question).listen(
          (e) => setState(() {
            if (e.sources != null) _sources = e.sources!;
            if (e.token != null) _answer.write(e.token);
          }),
          onError: (Object e) => setState(() {
            _error = friendlyLlmError(e);
            _done = true;
          }),
          onDone: () => setState(() => _done = true),
        );
  }

  @override
  void dispose() {
    unawaited(_sub?.cancel());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final text = _answer.toString().trim();
    return ListView(
      controller: widget.controller,
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      children: [
        Text(widget.question, style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 12),
        if (_error != null)
          Text(_error!, style: TextStyle(color: t.colorScheme.error))
        else if (text.isEmpty)
          const Row(children: [SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)), SizedBox(width: 10), Text('Reading…')])
        else
          SelectableText(text, style: t.textTheme.bodyLarge),
        if (_done && text.isNotEmpty)
          Align(
            alignment: Alignment.centerRight,
            child: TextButton.icon(
              onPressed: () async {
                await Clipboard.setData(ClipboardData(text: 'Q: ${widget.question}\nA: $text'));
                if (context.mounted) showMessage(context, 'Copied');
              },
              icon: const Icon(Icons.copy_rounded, size: 18),
              label: const Text('Copy'),
            ),
          ),
        if (_sources.isNotEmpty) ...[
          SectionHeader('Based on ${_sources.length} line${_sources.length == 1 ? '' : 's'}', padding: const EdgeInsets.fromLTRB(0, 16, 0, 4)),
          for (final seg in _sources)
            ListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              leading: SpeakerAvatar(label: seg.speakerLabel, known: seg.isKnownSpeaker, radius: 14),
              title: Text(seg.text, maxLines: 2, overflow: TextOverflow.ellipsis),
              subtitle: Text('${seg.speakerLabel} · ${formatWhen(seg.startedAt)}'),
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute<void>(
                  builder: (_) => ConversationScreen(services: widget.services, conversationId: seg.conversationId, highlightSegmentId: seg.id),
                ),
              ),
            ),
        ],
      ],
    );
  }
}
