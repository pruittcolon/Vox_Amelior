import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/glasses/display_position.dart';
import 'package:even_suite/core/glasses/height_send.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  test('text at two heights: move to the spot, send its text, then the other spot', () async {
    final glasses = FakeGlasses();
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    await control.connectNow();
    final proto = Proto(ble);
    final send = HeightSend(proto, DisplayPosition(proto))
      ..top.height = 7
      ..top.text = 'UP'
      ..top.holdMs = 0
      ..bottom.height = 3
      ..bottom.text = 'DOWN'
      ..bottom.holdMs = 0;

    // The moves (0x26 state 1 = go there, state 0 = keep it) and the text, in order.
    List<String> order() => [
          for (final p in glasses.sent['L']!)
            if (p[0] == 0x26 && p[5] == 1) 'move${p[6]}' else if (p[0] == 0x4E) 'text',
        ];

    await send.sendOnce(send.top);
    expect(order(), ['move7', 'text']);
    expect(glasses.sent['L']!.where((p) => p[0] == 0x26).last[5], 0, reason: 'the preview is closed again');

    await send.sendOnce(send.bottom);
    expect(order(), ['move7', 'text', 'move3', 'text']);
    expect(send.steps, 2);
    expect(send.error, isNull);

    // Alternating, until stopped.
    await send.start();
    await Future<void>.delayed(const Duration(milliseconds: 2000));
    await send.stop();
    final heights = [
      for (final p in glasses.sent['L']!)
        if (p[0] == 0x26 && p[5] == 1) p[6],
    ];
    expect(heights.length, greaterThanOrEqualTo(6));
    final after = heights.sublist(2); // the two one-offs came first
    for (var i = 0; i < after.length; i++) {
      expect(after[i], i.isEven ? 7 : 3, reason: 'top, bottom, top, bottom ...');
    }
    expect(send.running, isFalse);
    ble.dispose();
  });
}
