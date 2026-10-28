import 'package:even_suite/core/bridge/ansi.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/bridge/terminal_controller.dart';
import 'package:even_suite/core/voice/voice_commands.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('terminal colours', () {
    test('runs of text in their colours; a colour carries over lines until changed', () {
      final lines = Ansi.parse('\x1b[38;2;215;119;87m●\x1b[0m Reading\n\x1b[1;32mok\nstill green\x1b[39m plain');
      expect(lines[0].map((s) => s.text).join(), '● Reading');
      expect(lines[0].first.fg, 0xFFD77757);
      expect(lines[0][1].fg, isNull);
      expect(lines[1].single.bold, isTrue);
      expect(lines[1].single.fg, Ansi.palette[2]);
      expect(lines[2].first.fg, Ansi.palette[2], reason: 'tmux writes a colour once');
      expect(lines[2].last.text, ' plain');
      expect(lines[2].last.fg, isNull);
    });

    test('256 colours, inverse, and other escapes dropped', () {
      final l = Ansi.parse('\x1b[38;5;244mgrey\x1b[7m inv\x1b]8;;https://x\x07link\x1b]8;;\x07')[0];
      expect(l.first.fg, Ansi.color256(244));
      expect(l[1].inverse, isTrue);
      expect(l.map((s) => s.text).join(), 'grey invlink');
      expect(Ansi.color256(196), 0xFFFF0000);
      expect(Ansi.strip('\x1b[1m❯\x1b[0m fix it'), '❯ fix it');
      expect(Ansi.strip('plain'), 'plain');
    });

    test('glasses lines never show colour codes', () {
      final lines = TerminalController.glassesLines('\x1b[32m● Done\x1b[0m\n\x1b[2m──────\x1b[0m');
      expect(lines.join('\n'), isNot(contains('\x1b')));
      expect(lines.join('\n'), contains('Done'));
    });
  });

  group('sessions', () {
    test('names people can read', () {
      expect(const TermSession('claude', main: true).label, 'Claude Code');
      expect(TermSession.labelFor('claude-2'), 'Claude 2');
      expect(TermSession.labelFor('shell'), 'Shell');
      expect(TermSession.labelFor('shell-3'), 'Shell 3');
      expect(TermSession.labelFor('agy'), 'Antigravity');
      expect(TermSession.labelFor('work'), 'work');
      final s = TermSession.fromMap(const {'name': 'claude-2', 'kind': 'claude', 'state': 'asking', 'topic': 'deploy'});
      expect((s.isClaude, s.state, s.topic, s.main), (true, 'asking', 'deploy', false));
    });

    test('"new session" and "next session" are commands, not messages', () {
      expect(VoiceCommands.parse('new session'), VoiceCommand.newSession);
      expect(VoiceCommands.parse('new Claude session'), VoiceCommand.newSession);
      expect(VoiceCommands.parse('next session'), VoiceCommand.nextSession);
      expect(VoiceCommands.parse('switch session'), VoiceCommand.nextSession);
      expect(VoiceCommands.parse('start a new session of the dev server and run the tests'), isNull);
    });
  });
}
