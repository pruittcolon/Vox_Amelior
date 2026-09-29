import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/data/models.dart';

/// A rule that matched an utterance, with the values available to templates.
class RuleFire {
  const RuleFire(this.rule, this.context);

  final AutomationRule rule;
  final Map<String, String> context;
}

/// Decides which rules fire for an utterance, honouring cooldowns.
class RuleEngine {
  RuleEngine({this._clock = systemClock});

  final Clock _clock;
  final Map<String, DateTime> _lastFired = {};

  /// Evaluates [rules] for [segment]. [wakeCommand] is the text after the
  /// wake phrase, or null if the utterance was not addressed to the assistant.
  List<RuleFire> evaluate(
    List<AutomationRule> rules,
    SegmentView segment, {
    String? wakeCommand,
  }) {
    final now = _clock();
    final fires = <RuleFire>[];
    for (final rule in rules) {
      if (!rule.enabled) continue;
      final t = rule.trigger;
      if (!_speakerAllowed(t, segment)) continue;

      final String subject;
      if (t.scope == TriggerScope.wakeCommand) {
        if (wakeCommand == null) continue;
        subject = wakeCommand;
      } else {
        subject = segment.text;
      }

      final matched = _match(t, subject);
      if (matched == null) continue;

      final last = _lastFired[rule.id] ?? rule.lastFiredAt;
      if (last != null && now.difference(last).inSeconds < rule.cooldownSeconds) continue;
      _lastFired[rule.id] = now;

      fires.add(RuleFire(rule, {
        'text': segment.text,
        'speaker': segment.speakerLabel,
        'match': matched,
        'command': wakeCommand ?? '',
        'time_iso': segment.startedAt.toIso8601String(),
        'date': _date(segment.startedAt),
        'time': _time(segment.startedAt),
        'segment_id': '${segment.id}',
        'conversation_id': '${segment.conversationId}',
        'rule': rule.name,
      }));
    }
    return fires;
  }

  bool _speakerAllowed(RuleTrigger t, SegmentView s) {
    final wanted = t.speakerName?.trim();
    if (wanted == null || wanted.isEmpty) return true;
    return s.speakerName != null && s.speakerName!.toLowerCase() == wanted.toLowerCase();
  }

  /// Returns the matched text, '' for a catch-all wake rule, or null for no match.
  String? _match(RuleTrigger t, String subject) {
    if (!t.hasCondition) return t.scope == TriggerScope.wakeCommand ? '' : null;
    final normalized = _normalize(subject);
    for (final phrase in t.phrases) {
      final p = _normalize(phrase);
      if (p.isEmpty) continue;
      if (RegExp('(^| )${RegExp.escape(p)}( |\$)').hasMatch(normalized)) return phrase;
    }
    final pattern = t.pattern;
    if (pattern != null && pattern.isNotEmpty) {
      try {
        final m = RegExp(pattern, caseSensitive: false).firstMatch(subject);
        if (m != null) return m.group(0) ?? '';
      } on FormatException {
        return null;
      }
    }
    return null;
  }

  static String _normalize(String s) =>
      s.toLowerCase().replaceAll(RegExp(r"[^\p{L}\p{N}']+", unicode: true), ' ').trim();

  static String _date(DateTime d) => '${d.year}-${_two(d.month)}-${_two(d.day)}';
  static String _time(DateTime d) => '${_two(d.hour)}:${_two(d.minute)}';
  static String _two(int n) => n.toString().padLeft(2, '0');
}
