class VocabularyWord {
  final int? id;
  final String word;
  final String definition;
  final String wordBreakdown;
  final String sameRootWords;
  final bool isMissed;

  const VocabularyWord({
    this.id,
    required this.word,
    required this.definition,
    required this.wordBreakdown,
    required this.sameRootWords,
    this.isMissed = false,
  });

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'word': word,
        'definition': definition,
        'wordBreakdown': wordBreakdown,
        'sameRootWords': sameRootWords,
        'isMissed': isMissed ? 1 : 0,
      };

  factory VocabularyWord.fromMap(Map<String, dynamic> m) => VocabularyWord(
        id: m['id'] as int?,
        word: m['word'] as String,
        definition: m['definition'] as String,
        wordBreakdown: m['wordBreakdown'] as String,
        sameRootWords: m['sameRootWords'] as String,
        isMissed: (m['isMissed'] as int? ?? 0) == 1,
      );
}
