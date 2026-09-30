import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/data/models.dart';

class SegmentOutcome {
  const SegmentOutcome({this.firedRules = 0, this.assistantRequestId, this.wakeCommand});

  final int firedRules;

  /// Set when the utterance was a question for the assistant.
  final int? assistantRequestId;
  final String? wakeCommand;
}

/// Everything that happens after an utterance is saved: automation rules and
/// wake-phrase detection.
class SegmentHandler {
  SegmentHandler({
    required this.rules,
    required this.engine,
    required this.executor,
    required this.requests,
    required this.wakeParser,
    this.clock = systemClock,
  });

  final RuleRepository rules;
  final RuleEngine engine;
  final ActionExecutor executor;
  final AssistantRequestRepository requests;

  /// Swappable so a changed wake phrase applies without a restart.
  WakeCommandParser wakeParser;
  final Clock clock;

  Future<SegmentOutcome> handle(SegmentView segment) async {
    final command = wakeParser.extract(segment.text);
    final fires = engine.evaluate(rules.all(), segment, wakeCommand: command);
    for (final fire in fires) {
      rules.markFired(fire.rule.id, clock());
      await executor.execute(fire);
    }

    int? requestId;
    if (command != null && command.isNotEmpty) {
      requestId = requests.add(command);
    }
    return SegmentOutcome(firedRules: fires.length, assistantRequestId: requestId, wakeCommand: command);
  }
}
