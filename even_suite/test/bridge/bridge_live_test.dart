// Talks to a real Agent Hub server (started by CI with a fake `claude`).
// Skipped unless BRIDGE_TEST_URL is set, e.g. BRIDGE_TEST_URL=http://127.0.0.1:8799
import 'dart:async';
import 'dart:io';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/bridge/chat_controller.dart';
import 'package:even_suite/core/bridge/terminal_controller.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  final url = Platform.environment['BRIDGE_TEST_URL'];
  final skip = url == null ? 'BRIDGE_TEST_URL not set (CI starts a test bridge)' : null;

  test('client: status, unreachable bridge gives a friendly error', () async {
    final client = BridgeClient(url!);
    final status = await client.status();
    expect(status.keys, containsAll(['claude', 'shell']));
    final dead = BridgeClient('http://127.0.0.1:1');
    await expectLater(dead.status(), throwsA(isA<BridgeException>()));
    client.close();
    dead.close();
  }, skip: skip);

  test('chat: send a message, the reply arrives through the long poll', () async {
    final client = BridgeClient(url!);
    final chat = ChatController(client);
    final replies = <BridgeMessage>[];
    final got = Completer<void>();
    chat.onReply = (m) {
      replies.add(m);
      if (!got.isCompleted) got.complete();
    };
    final owner = Object();
    chat.watch(owner);
    await chat.send('what is 6 times 7');
    await got.future.timeout(const Duration(seconds: 30));
    expect(replies.single.text, 'echo: what is 6 times 7');
    expect(chat.state, BridgeState.online);
    expect(chat.lastReply?.text, 'echo: what is 6 times 7');
    chat.unwatch(owner);
    client.close();
  }, skip: skip);

  test('terminal: type into the shell, see it, mirror it to the glasses', () async {
    final client = BridgeClient(url!);
    final glasses = FakeGlasses();
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    final display = GlassesDisplay(Proto(ble));
    await control.connectNow();
    final term = TerminalController(client, display)..setAgent('shell');
    final owner = Object();
    term.watch(owner);
    term.setMirror(true);
    await term.sendLine(r'echo glasses-$((6*7))');
    final deadline = DateTime.now().add(const Duration(seconds: 20));
    while (!term.text.contains('glasses-42') && DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }
    expect(term.text, contains('glasses-42'));
    // Give the display a moment to push the mirrored screen.
    await Future<void>.delayed(const Duration(milliseconds: 600));
    expect(display.mode, DisplayMode.mirror);
    expect(glasses.screens().join('\n'), contains('glasses-42'));
    await term.key('C-c');
    term.setMirror(false);
    term.unwatch(owner);
    display.dispose();
    ble.dispose();
    client.close();
  }, skip: skip);
}
