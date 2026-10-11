/// How a review's answers are combined.
enum ReviewKind {
  /// Each part answers with one line per finding in a fixed format. The app
  /// counts and groups the findings itself, so totals are exact.
  list,

  /// Each part answers in free text; the parts are merged at the end.
  summary,
}

/// A ready-made "go through everything" task. All fields are editable
/// before starting; users can save their own.
class ReviewTemplate {
  const ReviewTemplate({
    required this.id,
    required this.name,
    required this.prompt,
    required this.format,
    this.kind = ReviewKind.list,
    this.builtIn = false,
  });

  final String id;
  final String name;
  final String prompt;
  final String format;
  final ReviewKind kind;
  final bool builtIn;

  /// What the second field of each finding means, for the results screen.
  String get categoryLabel => switch (id) {
        'promises' => 'Who',
        'decisions' || 'disagreements' => 'Topic',
        'facts' => 'Kind',
        'fallacies' => 'Fallacy',
        'fights' => 'Stage',
        'positions' => 'Who',
        'kindness' => 'Kind',
        _ => 'Type',
      };

  ReviewTemplate copyWith({String? name, String? prompt, String? format, ReviewKind? kind}) => ReviewTemplate(
        id: id,
        name: name ?? this.name,
        prompt: prompt ?? this.prompt,
        format: format ?? this.format,
        kind: kind ?? this.kind,
        builtIn: builtIn,
      );

  Map<String, Object?> toJson() => {'id': id, 'name': name, 'prompt': prompt, 'format': format, 'kind': kind.name};

  static ReviewTemplate? fromJson(Object? j) {
    if (j is! Map) return null;
    final id = j['id'];
    final name = j['name'];
    final prompt = j['prompt'];
    final format = j['format'];
    if (id is! String || name is! String || prompt is! String || format is! String) return null;
    return ReviewTemplate(
      id: id,
      name: name,
      prompt: prompt,
      format: format,
      kind: ReviewKind.values.asNameMap()[j['kind']] ?? ReviewKind.list,
    );
  }

  static const String noneLine = 'If there are none in this part, reply only: NONE';

  /// The line format every list review uses (the app parses it).
  static String listFormat(String second, String third) =>
      'One line per finding, exactly like this:\n- [line number] "exact quote" — $second — $third\n$noneLine';

  static final List<ReviewTemplate> builtIns = [
    ReviewTemplate(
      id: 'fallacies',
      name: 'Logical fallacies',
      prompt: 'Find every logical fallacy used in an argument (for example strawman, ad hominem, false dilemma, '
          'slippery slope, hasty generalisation, appeal to emotion, whataboutism, moving the goalposts, '
          'appeal to authority, circular reasoning). Only real fallacies, not jokes or casual remarks.',
      format: listFormat('Fallacy name', 'why it is that fallacy, in one sentence'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'fights',
      name: 'Fights & tension',
      prompt: 'Find every argument, fight or moment of tension. Lines marked (angry), (sad) or similar tell you how '
          'something was said. For each, quote the line where it started, where it got worse, and where it calmed '
          'down or ended. Say who did what, fairly, without taking sides.',
      format: listFormat('Stage (started, escalated, calmed down or unresolved)', 'what happened at that moment, in one sentence'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'positions',
      name: 'Who wanted what',
      prompt: 'For each disagreement, state what each person wanted or believed and the reason they gave. '
          'Be fair to both sides and use their own words where you can.',
      format: listFormat('Who', 'their position and their reason, in one sentence'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'kindness',
      name: 'Kind words',
      prompt: 'Find moments of appreciation, support, affection, apology or humour between the people talking.',
      format: listFormat('Kind (thanks, support, affection, apology or humour)', 'who said it to whom and why it mattered'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'promises',
      name: 'Promises & to-dos',
      prompt: 'List every promise, commitment, plan or to-do that someone agreed to or said they would do.',
      format: listFormat('Who', 'what they will do, and when if it was said'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'decisions',
      name: 'Decisions',
      prompt: 'List every decision that was made or agreed on.',
      format: listFormat('Topic', 'what was decided'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'disagreements',
      name: 'Disagreements',
      prompt: 'List every disagreement or argument, and what it was about.',
      format: listFormat('Topic', 'who disagreed about what, in one sentence'),
      builtIn: true,
    ),
    ReviewTemplate(
      id: 'facts',
      name: 'Things to remember',
      prompt: 'List facts worth remembering: dates, appointments, names, phone numbers, addresses, prices and amounts.',
      format: listFormat('Kind (date, name, number, place or other)', 'the fact'),
      builtIn: true,
    ),
    const ReviewTemplate(
      id: 'summary',
      name: 'Summary',
      prompt: 'Summarise what was talked about: the main topics, decisions and plans.',
      format: 'Short bullet points grouped by topic. Say who said what and roughly when.',
      kind: ReviewKind.summary,
      builtIn: true,
    ),
  ];

  /// Starting point for a user's own review.
  static final ReviewTemplate blank = ReviewTemplate(
    id: 'custom',
    name: 'Your own',
    prompt: '',
    format: listFormat('Type', 'short explanation'),
  );
}
