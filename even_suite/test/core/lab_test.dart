import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/log.dart';
import 'package:even_suite/core/screen/picture_lab.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  test('picture test runs every trial and logs each arm\'s answer', () async {
    EventLog.instance.clear();
    final glasses = FakeGlasses();
    final control = FakeControl();
    final ble = BleManager(glasses, control: control);
    await control.connectNow();
    final lab = PictureLab(Proto(ble));
    await lab.run();
    expect(lab.running, isFalse);
    expect(lab.status, startsWith('Done'));
    final trials = lab.results.map((r) => r.trial).toSet();
    expect(trials.length, 12);
    final crcs = lab.results.where((r) => r.step.startsWith('crc')).toList();
    expect(crcs.every((r) => r.crc == 'c9'), isTrue, reason: 'simulated glasses accept everything');
    expect(crcs.where((r) => r.arm == 'L').length, greaterThan(15));
    final logged = EventLog.instance.recent.where((e) => e.cat == 'lab' && e.name == 'step');
    expect(logged.length, lab.results.length);
    // Half-height pictures really are smaller.
    final sizes = EventLog.instance.recent
        .where((e) => e.cat == 'lab' && '${e.data['step']}'.startsWith('send A68'))
        .map((e) => e.data['bytes']);
    expect(sizes.first, 62 + 72 * 68 + 2);
    ble.dispose();
  }, timeout: const Timeout(Duration(minutes: 1)));
}
