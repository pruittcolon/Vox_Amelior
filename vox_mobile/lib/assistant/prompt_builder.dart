import 'package:vox_amelior_mobile/data/models.dart';

/// Builds the instructions and context handed to the language model.
class PromptBuilder {
  const PromptBuilder();

  String system({required DateTime now, required List<String> people, String? assistantName}) {
    final b = StringBuffer()
      ..writeln('You are ${assistantName ?? 'Vox'}, a private assistant living on the user\'s phone.')
      ..writeln('You help the household remember what was said in their own conversations.')
      ..writeln('Rules:')
      ..writeln('- Answer only from the conversation excerpts provided. Never invent details.')
      ..writeln('- If the excerpts do not contain the answer, say you did not hear that.')
      ..writeln('- Say who said it and roughly when, using the speaker labels and times given.')
      ..writeln('- Keep answers short and plain: one to four sentences unless asked for more.')
      ..writeln('- The excerpts come from automatic speech recognition and may contain mistakes.')
      ..writeln('Current date and time: ${_stamp(now)}.');
    if (people.isNotEmpty) b.writeln('People in this household: ${people.join(', ')}.');
    return b.toString().trim();
  }

  String question({
    required String question,
    required List<SegmentView> excerpts,
    required DateTime now,
    String? timeLabel,
  }) {
    final b = StringBuffer();
    if (excerpts.isEmpty) {
      b.writeln('Conversation excerpts: (none found)');
    } else {
      b.writeln('Conversation excerpts${timeLabel == null ? '' : ' from $timeLabel'}:');
      int? lastConversation;
      for (final s in excerpts) {
        if (s.conversationId != lastConversation) {
          b.writeln('--- conversation on ${_stamp(s.startedAt)} ---');
          lastConversation = s.conversationId;
        }
        b.writeln('[${_hm(s.startedAt)}] ${s.speakerLabel}: ${s.text}');
      }
    }
    b
      ..writeln()
      ..writeln('Question: ${question.trim()}');
    return b.toString().trim();
  }

  /// Prompt for "summarise this stretch of conversation".
  String summary({required List<SegmentView> segments, required String label}) {
    final b = StringBuffer('Summarise the conversations from $label in a short bulleted list. '
        'Mention who said what when it matters, and list any tasks, plans or decisions.\n\n');
    int? last;
    for (final s in segments) {
      if (s.conversationId != last) {
        b.writeln('--- conversation on ${_stamp(s.startedAt)} ---');
        last = s.conversationId;
      }
      b.writeln('[${_hm(s.startedAt)}] ${s.speakerLabel}: ${s.text}');
    }
    return b.toString().trim();
  }

  static String _stamp(DateTime d) {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    return '${days[d.weekday - 1]} ${d.year}-${_p(d.month)}-${_p(d.day)} ${_hm(d)}';
  }

  static String _hm(DateTime d) => '${_p(d.hour)}:${_p(d.minute)}';
  static String _p(int n) => n.toString().padLeft(2, '0');
}
