import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/bridge/chat_controller.dart';
import 'package:even_suite/core/glasses/claude_panel.dart';
import 'package:even_suite/core/glasses/glasses_text.dart';
import 'package:even_suite/core/settings.dart';
import 'package:even_suite/core/suite.dart';
import 'package:even_suite/core/voice/voice_commands.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

BridgeMessage _msg(String id, String who, String status, {String text = '', String stage = 'done'}) =>
    BridgeMessage(id: id, who: who, text: text, status: status, stage: stage, ts: DateTime.now().millisecondsSinceEpoch / 1000);

void main() {
  group('who answers: Claude or Antigravity', () {
    test('saying it switches (and a question that starts with it does not)', () {
      for (final said in ['use agy', 'Switch to Antigravity.', 'talk to anti gravity', 'use a g y', 'Please use agy']) {
        expect(VoiceCommands.parse(said), VoiceCommand.useAgy, reason: said);
      }
      for (final said in ['use Claude', 'switch back to Claude', 'Back to Claude.']) {
        expect(VoiceCommands.parse(said), VoiceCommand.useClaude, reason: said);
      }
      expect(VoiceCommands.parse('use agy to find my keys'), isNull);
      expect(VoiceCommands.parse('what is antigravity'), isNull);
      // The old way back from the terminal is untouched.
      expect(VoiceCommands.parse('talk to claude'), VoiceCommand.toChat);
    });

    test('the setting is claude unless it says agy, and is kept', () {
      final s = Settings();
      expect(s.agent, 'claude');
      expect(s.agyCommands, isFalse, reason: 'running commands is never on by itself');
      s.set('agent', 'agy');
      expect(s.agent, 'agy');
      s.set('agent', 'nonsense');
      expect(s.agent, 'claude');
      final loaded = Settings()..load({'agent': 'agy', 'agyCommands': true});
      expect(loaded.agent, 'agy');
      expect(loaded.agyCommands, isTrue);
      expect((Settings()..load({'agent': 'gemini'})).agent, 'claude');
    });

    test('the chat listens to the one picked', () {
      final chat = ChatController(FakeBridgeClient());
      expect(chat.agentName, 'Claude');
      chat.agent = 'agy';
      expect(chat.agentName, 'Antigravity');
      chat.apply(BridgeSnapshot(1, [
        _msg('a', 'claude', 'done', text: 'from claude'),
        _msg('b', 'agy', 'pending', stage: 'thinking'),
      ], const ['agy']));
      expect(chat.busy, isTrue, reason: 'Antigravity is working');
      expect(chat.lastReply, isNull, reason: 'Claude\'s answer is not Antigravity\'s');
      chat.apply(BridgeSnapshot(2, [
        _msg('a', 'claude', 'done', text: 'from claude'),
        _msg('b', 'agy', 'done', text: 'from agy'),
      ], const []));
      expect(chat.busy, isFalse);
      expect(chat.lastReply?.text, 'from agy');
      chat.agent = 'claude';
      expect(chat.lastReply?.text, 'from claude');
    });

    test('the glasses rules and the look-up panel say which one is answering', () {
      const size = ScreenSize(47, 5);
      expect(Suite.buildPrimer(size: size), contains('You run as Claude Code'));
      final agy = Suite.buildPrimer(size: size, agent: 'agy');
      expect(agy, contains('Google Antigravity'));
      expect(agy, isNot(contains('You run as Claude Code')));
      final panel = ClaudePanel.build(asking: ('Thinking', 'what is in my notes', ''), agent: 'Antigravity');
      expect(panel.title, contains('Antigravity'));
      expect(ClaudePanel.build(asking: ('Thinking', 'q', '')).title, contains('Claude'));
    });
  });
}
