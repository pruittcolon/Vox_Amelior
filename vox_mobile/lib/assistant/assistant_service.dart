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

/// Answers questions about past conversations using local retrieval and the
/// on-device language model.
class AssistantService {
  AssistantService({
    required this.llm,
    required this.transcripts,
    required this.speakers,
    this.clock = systemClock,
    this.prompts = const PromptBuilder(),
  });

  final LlmEngine llm;
  final TranscriptRepository transcripts;
  final SpeakerRepository speakers;
  final Clock clock;
  final PromptBuilder prompts;

  /// Streams the answer as it is generated. The first event carries the
  /// excerpts used, so the UI can show sources immediately.
  Stream<AnswerEvent> ask(String question) async* {
    final trimmed = question.trim();
    if (trimmed.isEmpty) return;
    final now = clock();
    final people = speakers.profiles();
    final parsed = QueryParser(people: people).parse(trimmed, now: now);
    final excerpts = Retriever(transcripts).retrieve(parsed);
    yield AnswerEvent.sources(excerpts);

    await llm.ensureLoaded();
    yield* llm
        .generate(
          system: prompts.system(now: now, people: people.map((p) => p.name).toList()),
          prompt: prompts.question(question: trimmed, excerpts: excerpts, now: now, timeLabel: parsed.timeLabel),
        )
        .map(AnswerEvent.token);
  }

  /// Convenience for callers that don't stream (voice replies, webhooks).
  Future<Answer> answer(String question) async {
    final buffer = StringBuffer();
    var sources = const <SegmentView>[];
    await for (final e in ask(question)) {
      if (e.sources != null) sources = e.sources!;
      if (e.token != null) buffer.write(e.token);
    }
    return Answer(text: buffer.toString().trim(), sources: sources);
  }

  /// Streams a summary of everything said between [from] and [to].
  Stream<String> summarize(DateTime from, DateTime to, {required String label}) async* {
    final segments = transcripts.between(from, to, limit: 400);
    if (segments.isEmpty) {
      yield 'I did not hear any conversation $label.';
      return;
    }
    await llm.ensureLoaded();
    yield* llm.generate(
      system: prompts.system(now: clock(), people: speakers.profiles().map((p) => p.name).toList()),
      prompt: prompts.summary(segments: _fit(segments), label: label),
    );
  }

  /// Keeps summaries within the model's small context by sampling evenly.
  List<SegmentView> _fit(List<SegmentView> segments, {int maxChars = 9000}) {
    final total = segments.fold<int>(0, (a, s) => a + s.text.length + 30);
    if (total <= maxChars) return segments;
    final step = (total / maxChars).ceil();
    return [for (var i = 0; i < segments.length; i += step) segments[i]];
  }
}

class AnswerEvent {
  const AnswerEvent.sources(List<SegmentView> this.sources) : token = null;
  const AnswerEvent.token(String this.token) : sources = null;

  final List<SegmentView>? sources;
  final String? token;
}
