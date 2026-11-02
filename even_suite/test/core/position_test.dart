import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/glasses/display_position.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  test('text height: clear, preview at each spot (one move at a time), close when done', () async {
    final glasses = FakeGlasses();
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    await control.connectNow();
    final pos = DisplayPosition(Proto(ble));

    await pos.begin(2, 5);
    List<List<int>> moves(String lr) => [
          for (final p in glasses.sent[lr]!)
            if (p[0] == 0x26) [p[5], p[6], p[7]],
        ];
    expect(glasses.sent['L']!.first, [0x18], reason: 'cleared first: the preview only opens from idle');
    expect(moves('L'), [
      [1, 2, 5]
    ]);
    expect(moves('R'), [
      [1, 2, 5]
    ], reason: 'both arms');

    // Dragging fast: only the newest value goes once the glasses answer.
    pos.set(h: 3);
    pos.set(h: 4);
    pos.set(h: 5);
    await Future<void>.delayed(const Duration(milliseconds: 50));
    expect(moves('L').map((m) => m[1]).toList(), [2, 3, 5]);

    await pos.end();
    expect(moves('R').last, [0, 5, 5], reason: 'state 0 closes the preview, same spot');
    expect(pos.open, isFalse);
    expect(pos.error, isNull);
    ble.dispose();
  });
}
