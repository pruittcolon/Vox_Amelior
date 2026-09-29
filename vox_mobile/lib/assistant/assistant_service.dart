import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/retriever.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

class Answer {
  const Answer({required this.text, required this.sources});

  final String text;

  /// The conversation excerpts the answer was based on.
  final List<SegmentView> sources;
}

/// One step of an answer as it is produced.
class AnswerEvent {
  const AnswerEvent.sources(List<SegmentView> this.sources)
      : token = null,
        toolName = null,
        toolArgs = null;
  const AnswerEvent.token(String this.token)
      : sources = null,
        toolName = null,
        toolArgs = null;
  const AnswerEvent.tool(String this.toolName, Map<String, Object?> this.toolArgs)
      : sources = null,
        token = null;

  final List<SegmentView>? sources;
  final String? token;

  /// The assistant used a tool (search, reminder, ...).
  final String? toolName;
  final Map<String, Object?>? toolArgs;
}

/// Answers questions about past conversations with local retrieval and the
/// on-device model. With a [toolbox] and agent mode on, Gemma may also call
/// tools (search further, read timelines, save notes, set reminders...).
class AssistantService {
  AssistantService({
    required this.llm,
    required this.transcripts,
    required this.speakers,
    this.toolbox,
    this.instructions,
    this.agentEnabled,
    this.clock = systemClock,
    this.prompts = const PromptBuilder(),
    this.maxToolRounds = 4,
  });

  final LlmEngine llm;
  final TranscriptRepository transcripts;
  final SpeakerRepository speakers;
  final AgentToolbox? toolbox;

  /// Custom instructions from settings (read on every question).
  final String? Function()? instructions;
  final bool Function()? agentEnabled;
  final Clock clock;
  final PromptBuilder prompts;
  final int maxToolRounds;

  /// Streams the answer. The first event carries the excerpts used.
  Stream<AnswerEvent> ask(String question) async* {
    final trimmed = question.trim();
    if (trimmed.isEmpty) return;
    final now = clock();
    final people = speakers.profiles();
    final parsed = QueryParser(people: people).parse(trimmed, now: now);
    final excerpts = Retriever(transcripts).retrieve(parsed);
    yield AnswerEvent.sources(excerpts);

    await llm.ensureLoaded();
    final useTools = toolbox != null && (agentEnabled?.call() ?? true) && llm.supportsTools;
    final session = await llm.openSession(
      system: prompts.system(
        now: now,
        people: people.map((p) => p.name).toList(),
        instructions: instructions?.call(),
        agent: useTools,
      ),
      tools: useTools ? toolbox!.specs : const [],
    );
    try {
      var stream = session.send(prompts.question(question: trimmed, excerpts: excerpts, now: now, window: parsed.window));
      for (var round = 0;; round++) {
        LlmToolCall? call;
        await for (final e in stream) {
          if (e is LlmText) {
            if (e.text.isNotEmpty) yield AnswerEvent.token(e.text);
          } else if (e is LlmToolCall) {
            call ??= e;
          }
        }
        if (call == null || !useTools) break;
        yield AnswerEvent.tool(call.name, call.args);
        final result = await toolbox!.call(call.name, call.args);
        if (round >= maxToolRounds) {
          // Stop the model from looping on tools; let it answer with what it has.
          stream = session.sendToolResult(call.name, {...result, 'note': 'No more tool calls. Answer now.'});
          await for (final e in stream) {
            if (e is LlmText && e.text.isNotEmpty) yield AnswerEvent.token(e.text);
          }
          break;
        }
        stream = session.sendToolResult(call.name, result);
      }
    } finally {
      await session.close();
    }
  }

  /// Convenience for callers that don't stream (voice replies).
  Future<Answer> answer(String question) async {
    final buffer = StringBuffer();
    var sources = const <SegmentView>[];
    await for (final e in ask(question)) {
      if (e.sources != null) sources = e.sources!;
      if (e.token != null) buffer.write(e.token);
    }
    return Answer(text: buffer.toString().trim(), sources: sources);
  }
}
