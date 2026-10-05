import 'dart:convert';
import 'dart:typed_data';

import 'package:demo_ai_even/services/efficient_gui.dart';
import 'package:demo_ai_even/services/fast_bmp.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';

/// The G1's image store as decompiled (try_to_save_file / write_font): the
/// upload is written from the start in 4 KB blocks; a block whose content is
/// unchanged is skipped, otherwise it is erased (all 0xFF) and rewritten.
/// Blocks past the end of the upload keep their old content.
void writeToG1Flash(Uint8List flash, Uint8List upload) {
  for (var at = 0; at < upload.length; at += StripHud.blockBytes) {
    final end = at + StripHud.blockBytes < upload.length
        ? at + StripHud.blockBytes
        : upload.length;
    final chunk = upload.sublist(at, end);
    if (listEquals(flash.sublist(at, end), chunk)) continue;
    flash.fillRange(at, at + StripHud.blockBytes, 0xff);
    flash.setRange(at, end, chunk);
  }
}

HudState claudeScreen({required String clock, required double progress}) =>
    HudState(
      title: 'Claude Code',
      clock: clock,
      battery: 82,
      lines: const [
        'Fix the BLE timeout',
        'Read ble_manager.dart',
        'Edit mirror_service.dart',
        'Run tests',
      ],
      selected: 3,
      progress: progress,
    );

String row(String name, int bytes, int packetsPerArm, double ms) =>
    '${name.padRight(30)} ${'$bytes B'.padLeft(8)} '
    '${'$packetsPerArm pkt/arm'.padLeft(12)} '
    '${'~${ms.round()} ms'.padLeft(9)} '
    '${'~${(1000 / ms).toStringAsFixed(1)}/s'.padLeft(8)}';

void main() {
  final report = <String>[];

  tearDownAll(() {
    // ignore: avoid_print
    print('\n=== G1 HUD size per update (estimated time, not measured) ===\n'
        '${report.join('\n')}\n');
  });

  group('TextHud', () {
    test('a full screen fits in one BLE packet per arm', () {
      final screen = TextHud.render(claudeScreen(clock: '12:34', progress: 0.6));
      // ignore: avoid_print
      print('--- text screen ---\n$screen\n-------------------');
      final lines = screen.split('\n');
      expect(lines.length, lessThanOrEqualTo(TextHud.rows));
      expect(lines.every((l) => l.length <= TextHud.cols), isTrue);
      expect(screen.codeUnits.every((c) => c == 10 || (c >= 0x20 && c < 0x7f)),
          isTrue, reason: 'plain ASCII only');

      final bytes = utf8.encode(screen).length;
      final packs = TextHud.packets(screen);
      final ms = G1Cost.textMs(packs);
      report.add(row('Text screen (5 lines)', bytes, packs.length, ms));
      expect(packs.length, 1);
      expect(packs.single.length, lessThanOrEqualTo(200));
      expect(ms, lessThan(150));
    });

    test('keeps the selected line visible and marks it', () {
      final s = HudState(
          title: 'T', clock: '1', battery: 1,
          lines: List.generate(10, (i) => 'item $i'), selected: 7);
      final screen = TextHud.render(s);
      expect(screen, contains('> item 7'));
      expect(screen, isNot(contains('item 0')));
    });

    test('long lines are clipped, not wrapped', () {
      final s = HudState(
          title: 'T', clock: '1', battery: 1, lines: ['x' * 100], selected: 0);
      expect(TextHud.render(s).split('\n')[1].length, TextHud.cols);
    });
  });

  group('StripHud', () {
    final hud = StripHud(stripRows: 20);
    final a = hud.frame(claudeScreen(clock: '12:34', progress: 0.40));
    final b = hud.frame(claudeScreen(clock: '12:35', progress: 0.45));

    test('static content stays out of the first flash block', () {
      expect(StripHud.topEnd, lessThanOrEqualTo(hud.stripTop));
      // Rows between the static area and the strip must be blank in a frame.
      final c = G1Canvas();
      final blank = c.toBmp();
      for (var y = StripHud.topEnd; y < hud.stripTop; y++) {
        final off = 62 + (G1Canvas.height - 1 - y) * G1Canvas.stride;
        expect(a.sublist(off, off + G1Canvas.stride),
            blank.sublist(off, off + G1Canvas.stride),
            reason: 'row $y should be blank');
      }
    });

    test('a clock/progress change sends only the strip', () {
      final d = hud.delta(a, b)!;
      expect(d.length, 62 + 20 * G1Canvas.stride);
      expect(hud.delta(b, b), isNull, reason: 'nothing changed, nothing sent');
      expect(hud.delta(null, b)!.length, b.length, reason: 'first frame is full');
    });

    test('flash ends up holding exactly the new frame after a strip update', () {
      final flash = Uint8List(a.length)..fillRange(0, a.length, 0xff);
      writeToG1Flash(flash, a);
      expect(flash, a);
      writeToG1Flash(flash, hud.delta(a, b)!);
      expect(flash, b);
    });

    test('sizes and estimated update time', () {
      final full = a.length;
      final strip20 = hud.delta(a, b)!.length;
      final hud56 = StripHud(stripRows: 56);
      final strip56 = hud56.stripUpdate(hud56.frame(claudeScreen(clock: '1', progress: 0))).length;

      int ppa(int n) => FastBmp.packets(Uint8List(n)).length;
      report.add(row('Bitmap strip, 20 rows (1 line)', strip20, ppa(strip20), G1Cost.bitmapMs(strip20)));
      report.add(row('Bitmap strip, 56 rows (3 lines)', strip56, ppa(strip56), G1Cost.bitmapMs(strip56)));
      report.add(row('Bitmap full frame', full, ppa(full), G1Cost.bitmapMs(full)));

      expect(full, 9854);
      expect(strip20, 1502);
      expect(ppa(strip20), 8);
      expect(G1Cost.bitmapMs(strip20), lessThan(200));
      expect(G1Cost.bitmapMs(strip20), lessThan(G1Cost.bitmapMs(full) / 2));
    });
  });
}
