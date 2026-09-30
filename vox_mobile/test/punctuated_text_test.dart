import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';

/// The Parakeet TDT 0.6B v2 model writes punctuation and capitals; the
/// text features must not care.
void main() {
  SegmentView seg(String text) => SegmentView(
        id: 1,
        conversationId: 1,
        startedAt: DateTime(2026, 9, 30, 18),
        duration: const Duration(seconds: 3),
        text: text,
      );

  AutomationRule ruleFor(RuleTrigger t) => AutomationRule(
        id: 'r',
        name: 'r',
        trigger: t,
        actions: [const NotifyAction(title: 't', body: 'b')],
        cooldownSeconds: 0,
      );

  test('phrase rules match whole words with capitals and punctuation around them', () {
    final engine = RuleEngine();
    final rules = [ruleFor(const RuleTrigger(phrases: ['groceries']))];
    for (final text in ['Groceries.', 'we need groceries, please', 'Add it to the "Groceries" list!', 'GROCERIES?']) {
      expect(engine.evaluate(rules, seg(text)), hasLength(1), reason: text);
    }
    expect(engine.evaluate(rules, seg('Grocery store.')), isEmpty);
    expect(engine.evaluate(rules, seg('Ungroceries.')), isEmpty);
    final multi = [ruleFor(const RuleTrigger(phrases: ['shopping list']))];
    expect(engine.evaluate(multi, seg('Put it on the shopping list.')), hasLength(1));
  });

  test('wake phrases work with capitals and commas', () {
    final parser = WakeCommandParser(['hey vox', 'vox']);
    expect(parser.extract('Hey Vox, what did Sam say?'), 'what did Sam say?');
    expect(parser.extract('Vox. Turn on the lights.'), 'Turn on the lights.');
    expect(parser.extract('Hey, Vox! Add milk.'), 'Add milk.');
    expect(parser.extract('Voxel is a word.'), isNull);
  });

  test('full-text search finds punctuated, capitalised text', () {
    final db = AppDatabase.inMemory();
    addTearDown(db.close);
    final repo = TranscriptRepository(db);
    repo.addSegment(
      text: 'Well, I don\'t wish to see it any more. We need Groceries, milk and eggs.',
      startedAt: DateTime(2026, 9, 30, 18),
      duration: const Duration(seconds: 5),
    );
    expect(repo.search(const SegmentQuery(keywords: ['groceries'])), hasLength(1));
    expect(repo.search(const SegmentQuery(keywords: ['Milk'])), hasLength(1));
    expect(repo.search(const SegmentQuery(keywords: ['well'])), hasLength(1));
  });

  test('word times from v2 tokens: punctuation joins the word before it', () {
    final w = wordsFromTokens(
      ['▁Well', ',', '▁I', '▁don', "'", 't', '▁wish', '.', '▁It', '▁is'],
      [0.1, 0.4, 0.6, 0.8, 0.9, 1.0, 1.3, 1.5, 2.0, 2.2],
    );
    expect(w.map((e) => e.text), ['Well,', 'I', "don't", 'wish.', 'It', 'is']);
    expect(w.map((e) => e.start), [0.1, 0.6, 0.8, 1.3, 2.0, 2.2]);
  });
}
