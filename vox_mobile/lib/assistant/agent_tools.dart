import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/query_parser.dart';
import 'package:vox_amelior_mobile/assistant/retriever.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';

typedef ToolHandler = Future<Map<String, Object?>> Function(Map<String, Object?> args);

class AgentTool {
  const AgentTool(this.spec, this.run);
  final ToolSpec spec;
  final ToolHandler run;
}

/// Hooks into the running app that some tools need. Null = not available.
class AgentHooks {
  const AgentHooks({this.runAutomation, this.pauseListening});

  /// Runs a named automation rule; returns a short result.
  final Future<String> Function(String ruleName)? runAutomation;

  /// Pauses listening for [minutes] (0 = until resumed).
  final Future<void> Function(int minutes)? pauseListening;
}

/// The actions Gemma can take on its own when agent mode is on.
class AgentToolbox {
  AgentToolbox({
    required this.transcripts,
    required this.speakers,
    required this.notes,
    required this.reminders,
    required this.rules,
    this.hooks = const AgentHooks(),
    this.clock = systemClock,
    ContextBudget Function()? budget,
  }) : budget = budget ?? (() => const ContextBudget(ContextBudget.defaultContext));

  final TranscriptRepository transcripts;
  final SpeakerRepository speakers;
  final NoteRepository notes;
  final ReminderRepository reminders;
  final RuleRepository rules;
  final AgentHooks hooks;
  final Clock clock;

  /// Sizes tool results so they fit the model's context.
  final ContextBudget Function() budget;

  int get _resultChars => ContextBudget.charsFor(budget().toolResultTokens);

  late final List<AgentTool> tools = [
    AgentTool(
      const ToolSpec(
        name: 'search_conversations',
        description: 'Search what was said at home. Use for specific facts, names, plans or topics.',
        parameters: {
          'type': 'object',
          'properties': {
            'query': {'type': 'string', 'description': 'Words to look for, e.g. "plumber appointment".'},
            'person': {'type': 'string', 'description': 'Only lines said by this person (optional).'},
            'period': {'type': 'string', 'description': 'Time phrase such as "last week", "yesterday", "this month" (optional).'},
          },
          'required': ['query'],
        },
      ),
      _search,
    ),
    AgentTool(
      const ToolSpec(
        name: 'get_timeline',
        description: 'Read an overview of all conversations in a period, e.g. to summarise a day or week.',
        parameters: {
          'type': 'object',
          'properties': {
            'period': {'type': 'string', 'description': 'Time phrase such as "today", "last week", "on Monday".'},
          },
          'required': ['period'],
        },
      ),
      _timeline,
    ),
    AgentTool(
      const ToolSpec(
        name: 'save_note',
        description: 'Save a note, to-do or shopping item for the household.',
        parameters: {
          'type': 'object',
          'properties': {'text': {'type': 'string'}},
          'required': ['text'],
        },
      ),
      (a) async {
        final text = '${a['text'] ?? ''}'.trim();
        if (text.isEmpty) return {'ok': false, 'error': 'text is empty'};
        return {'ok': true, 'id': notes.add(text, source: 'assistant')};
      },
    ),
    AgentTool(
      const ToolSpec(
        name: 'list_notes',
        description: 'List the most recent saved notes.',
        parameters: {
          'type': 'object',
          'properties': {
            'limit': {'type': 'integer', 'description': 'How many notes to list (default 20).'},
          },
        },
      ),
      (a) async {
        final limit = (int.tryParse('${a['limit'] ?? 20}') ?? 20).clamp(1, 50);
        return {
          'notes': [for (final n in notes.all(limit: limit)) {'text': n.text, 'saved': n.createdAt.toIso8601String()}],
        };
      },
    ),
    AgentTool(
      const ToolSpec(
        name: 'set_reminder',
        description: 'Remind the user later with a notification.',
        parameters: {
          'type': 'object',
          'properties': {
            'text': {'type': 'string', 'description': 'What to remind about.'},
            'minutes_from_now': {'type': 'integer', 'description': 'Delay in minutes.'},
          },
          'required': ['text', 'minutes_from_now'],
        },
      ),
      (a) async {
        final minutes = int.tryParse('${a['minutes_from_now']}') ?? -1;
        final text = '${a['text'] ?? ''}'.trim();
        if (minutes < 1 || minutes > 60 * 24 * 30 || text.isEmpty) {
          return {'ok': false, 'error': 'need text and 1..43200 minutes'};
        }
        final due = clock().add(Duration(minutes: minutes));
        reminders.add(text, due);
        return {'ok': true, 'due': due.toIso8601String()};
      },
    ),
    if (hooks.runAutomation != null)
      AgentTool(
        ToolSpec(
          name: 'run_automation',
          description: 'Run one of the user\'s automations (smart home webhooks etc.) by name. '
              'Available: ${rules.all().where((r) => r.enabled).map((r) => '"${r.name}"').join(', ')}.',
          parameters: const {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
          },
        ),
        (a) async => {'result': await hooks.runAutomation!('${a['name'] ?? ''}')},
      ),
    if (hooks.pauseListening != null)
      AgentTool(
        const ToolSpec(
          name: 'pause_listening',
          description: 'Stop listening for a while, for privacy.',
          parameters: {
            'type': 'object',
            'properties': {'minutes': {'type': 'integer', 'description': '0 means until resumed.'}},
          },
        ),
        (a) async {
          final m = int.tryParse('${a['minutes'] ?? 0}') ?? 0;
          await hooks.pauseListening!(m.clamp(0, 24 * 60));
          return {'ok': true, 'paused_minutes': m};
        },
      ),
  ];

  List<ToolSpec> get specs => [for (final t in tools) t.spec];

  /// Runs a tool by name. Never throws: errors are returned to the model.
  Future<Map<String, Object?>> call(String name, Map<String, Object?> args) async {
    final tool = tools.where((t) => t.spec.name == name).firstOrNull;
    if (tool == null) return {'error': 'unknown tool $name'};
    try {
      return await tool.run(args);
    } on Object catch (e) {
      return {'error': '$e'};
    }
  }

  Future<Map<String, Object?>> _search(Map<String, Object?> a) async {
    final people = speakers.profiles();
    final period = '${a['period'] ?? ''}';
    final person = '${a['person'] ?? ''}';
    final parsed = QueryParser(people: people).parse('${a['query'] ?? ''} $person $period', now: clock());
    final hits = Retriever(transcripts, maxHits: 10, contextLines: 1, maxChars: _resultChars).retrieve(
      ParsedQuery(keywords: parsed.keywords, speaker: parsed.speaker, window: parsed.window),
    );
    return {
      if (parsed.window != null) 'period': parsed.window!.describe(),
      'results': _lines(hits),
    };
  }

  Future<Map<String, Object?>> _timeline(Map<String, Object?> a) async {
    final window = QueryParser(people: const []).parse('${a['period'] ?? 'today'}', now: clock()).window;
    if (window == null) return {'error': 'could not understand the period'};
    final lines = Retriever(transcripts, maxChars: _resultChars).timeline(window.from, window.to);
    return {'period': window.describe(), 'lines': _lines(lines), if (lines.isEmpty) 'note': 'nothing was recorded'};
  }

  static List<Map<String, Object?>> _lines(List<SegmentView> segments) => [
        for (final s in segments)
          {
            'when': s.startedAt.toIso8601String().substring(0, 16),
            'who': s.speakerLabel,
            'said': s.text,
            if (s.toneNotes.isNotEmpty) 'tone': s.toneNotes.join(', '),
          },
      ];
}
