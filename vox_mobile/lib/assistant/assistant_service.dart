import 'dart:convert';

import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/retriever.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/idle_timeout.dart';
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
///
/// Everything handed to the model is sized from [budget] so it fits the
/// phone's context window. A failed attempt is retried without tools, then
/// (if the engine can) in a safer mode, before an error is shown. The model
/// is never waited on for longer than [idleTimeout] without output.
class AssistantService {
  AssistantService({
    required this.llm,
    required this.transcripts,
    required this.speakers,
    this.toolbox,
    this.instructions,
    this.agentEnabled,
    ContextBudget Function()? budget,
    this.clock = systemClock,
    this.prompts = const PromptBuilder(),
    this.maxToolRounds = 4,
    this.idleTimeout = const Duration(minutes: 3),
  }) : budget = budget ?? (() => const ContextBudget(ContextBudget.defaultContext));

  final LlmEngine llm;
  final TranscriptRepository transcripts;
  final SpeakerRepository speakers;
  final AgentToolbox? toolbox;

  /// Custom instructions from settings (read on every question).
  final String? Function()? instructions;
  final bool Function()? agentEnabled;
  final ContextBudget Function() budget;
  final Clock clock;
  final PromptBuilder prompts;
  final int maxToolRounds;
  final Duration idleTimeout;

  /// Streams the answer. The first event carries the excerpts used.
  Stream<AnswerEvent> ask(String question) async* {
    final trimmed = question.trim();
    if (trimmed.isEmpty) return;
    final now = clock();
    final b = budget();
    final people = speakers.profiles();
    final parsed = QueryParser(people: people).parse(trimmed, now: now);
    var useTools = toolbox != null && (agentEnabled?.call() ?? true) && llm.supportsTools;
    final excerpts = Retriever(transcripts, maxChars: ContextBudget.charsFor(b.excerptTokens(agent: useTools))).retrieve(parsed);
    yield AnswerEvent.sources(excerpts);

    await llm.ensureLoaded();
    final prompt = prompts.question(question: trimmed, excerpts: excerpts, now: now, window: parsed.window);
    var recovered = false;
    while (true) {
      var produced = false;
      try {
        await for (final e in _attempt(prompt, people, now, useTools, b)) {
          produced = true;
          yield e;
        }
        return;
      } on Object catch (e) {
        // Never repeat half an answer; only retry when nothing came out yet.
        if (produced || e is LlmUnavailable) rethrow;
        if (useTools) {
          useTools = false; // tool definitions are the most common cause
          continue;
        }
        if (!recovered && await llm.recover()) {
          recovered = true;
          await llm.ensureLoaded();
          continue;
        }
        throw LlmUnavailable(friendlyLlmError(e));
      }
    }
  }

  Stream<AnswerEvent> _attempt(String prompt, List<SpeakerProfile> people, DateTime now, bool useTools, ContextBudget b) async* {
    final system = prompts.system(
      now: now,
      people: people.map((p) => p.name).toList(),
      instructions: instructions?.call(),
      agent: useTools,
    );
    final session = await llm.openSession(
      system: system,
      tools: useTools ? toolbox!.specs : const [],
      maxReplyTokens: b.replyTokens,
    );
    // Running estimate of how full the context is.
    var used = ContextBudget.estimateTokens(system) +
        (useTools ? ContextBudget.toolSpecTokens : 0) +
        ContextBudget.estimateTokens(prompt) +
        b.replyTokens;
    try {
      var stream = session.send(prompt);
      for (var round = 0;; round++) {
        LlmToolCall? call;
        await for (final e in withIdleTimeout(stream, idleTimeout)) {
          if (e is LlmText) {
            if (e.text.isNotEmpty) yield AnswerEvent.token(e.text);
          } else if (e is LlmToolCall) {
            call ??= e;
          }
        }
        if (call == null || !useTools) break;
        yield AnswerEvent.tool(call.name, call.args);
        var result = await toolbox!.call(call.name, call.args);
        final size = ContextBudget.estimateTokens(jsonEncode(result)) + 40;
        final full = used + size > b.contextTokens;
        if (full) {
          result = {'note': 'There is no room to read more. Answer now with what you already have.'};
        } else {
          used += size;
        }
        if (full || round >= maxToolRounds) {
          // Stop the model from looping on tools; let it answer with what it has.
          stream = session.sendToolResult(call.name, {...result, 'note': 'No more tool calls. Answer now.'});
          await for (final e in withIdleTimeout(stream, idleTimeout)) {
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
