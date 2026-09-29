import 'package:vox_amelior_mobile/assistant/time_window.dart';
import 'package:vox_amelior_mobile/data/models.dart';

/// Builds the instructions and context handed to the language model.
class PromptBuilder {
  const PromptBuilder();

  static const String defaultInstructions =
      'Answer from the conversation excerpts and tool results only; never invent details. '
      'If they do not contain the answer, say you did not hear that. '
      'Say who said it and roughly when. Keep answers short and plain unless asked for more.';

  String system({
    required DateTime now,
    required List<String> people,
    String? instructions,
    bool agent = false,
  }) {
    final custom = instructions?.trim();
    final b = StringBuffer()
      ..writeln("You are Vox, a private assistant living on the user's phone. "
          'You help the household remember and act on what was said in their own conversations.')
      ..writeln(custom == null || custom.isEmpty ? defaultInstructions : custom)
      ..writeln('The excerpts come from automatic speech recognition and may contain mistakes.')
      ..writeln('Current date and time: ${stamp(now)}. Weeks start on Monday.');
    if (people.isNotEmpty) b.writeln('People in this household: ${people.join(', ')}.');
    if (agent) {
      b.writeln('You can call tools: search conversations or read a timeline when the excerpts are not enough, '
          'save notes, set reminders and run automations when the user asks. Do not call a tool when you can already answer.');
    }
    return b.toString().trim();
  }

  String question({
    required String question,
    required List<SegmentView> excerpts,
    required DateTime now,
    TimeWindow? window,
  }) {
    final b = StringBuffer();
    if (window != null) b.writeln('The question is about ${window.describe()}.');
    if (excerpts.isEmpty) {
      b.writeln('Conversation excerpts: (none found${window == null ? '' : ' in that period'})');
    } else {
      b.writeln('Conversation excerpts:');
      int? lastConversation;
      for (final s in excerpts) {
        if (s.conversationId != lastConversation) {
          b.writeln('--- conversation on ${stamp(s.startedAt)} ---');
          lastConversation = s.conversationId;
        }
        b.writeln('[${hm(s.startedAt)}] ${s.speakerLabel}: ${s.text}');
      }
    }
    b
      ..writeln()
      ..writeln('Question: ${question.trim()}');
    return b.toString().trim();
  }

  static String stamp(DateTime d) {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    return '${days[d.weekday - 1]} ${d.year}-${_p(d.month)}-${_p(d.day)} ${hm(d)}';
  }

  static String hm(DateTime d) => '${_p(d.hour)}:${_p(d.minute)}';
  static String _p(int n) => n.toString().padLeft(2, '0');
}
