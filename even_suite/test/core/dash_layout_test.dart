import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/glasses/claude_panel.dart';
import 'package:even_suite/core/glasses/dashboard.dart';
import 'package:even_suite/core/glasses/g1_font.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

/// Every line of a panel must fit the firmware's boxes, or the glasses cut it
/// letter by letter at the edge.
void expectFits(ClaudePanel p) {
  expect(G1Font.width(p.title), lessThanOrEqualTo(DashLayout.titleMaxPx), reason: p.title);
  expect(p.lines.length, lessThanOrEqualTo(DashLayout.bodyLines), reason: p.lines.join(' | '));
  for (final l in p.lines) {
    expect(G1Font.width(l), lessThanOrEqualTo(DashLayout.bodyMaxPx), reason: l);
    expect(l, isNot(contains('\n')));
  }
  // The note fields have room for 40 + 280 bytes.
  expect(p.title.length, lessThanOrEqualTo(40));
  expect(p.body.length, lessThanOrEqualTo(270));
}

void main() {
  group('G1 font', () {
    test('widths are (glyph + 1) x 2 pixels', () {
      expect(G1Font.width('a'), 12);
      expect(G1Font.width('m'), 16);
      expect(G1Font.width(' '), 6);
      expect(G1Font.width('Hi'), 12 + 4);
      expect(G1Font.width(G1Font.dimMark), 0);
      expect(G1Font.width('é'), 16, reason: 'unknown letters count as the widest');
    });

    test('wraps at spaces, never past the width', () {
      const text = 'Pushed the fix; CI is green and the APK is published to the even-suite-latest release.';
      final lines = G1Font.wrap(text, 372);
      expect(lines.join(' '), text);
      for (final l in lines) {
        expect(G1Font.width(l), lessThanOrEqualTo(372));
      }
      expect(lines.first, 'Pushed the fix; CI is green and the APK is');
    });

    test('a word longer than a line is split, a cut line ends in ...', () {
      final lines = G1Font.wrap('x' * 60, 120);
      expect(lines.every((l) => G1Font.width(l) <= 120), isTrue);
      expect(lines.join(), 'x' * 60);
      expect(G1Font.fit('Running flutter test --coverage on the whole package', 200), endsWith('...'));
      expect(G1Font.width(G1Font.fit('Running flutter test --coverage on the whole package', 200)), lessThanOrEqualTo(200));
      expect(G1Font.wrapMax('one two three four five six seven', 60, 2).last, endsWith('...'));
    });
  });

  group('Claude panel (right side of the look-up screen)', () {
    final now = DateTime(2026, 10, 27, 14, 20);

    test('working: the step bright, what you asked and where dim', () {
      final p = ClaudePanel.build(now: now, jobs: [
        WorkJob(
            session: 's',
            project: 'vox_mobile',
            prompt: 'add mood charts to the Insights tab and make the colors work in dark mode too',
            step: 'Running flutter test --coverage on the whole even_suite package',
            steps: 14,
            started: now.subtract(const Duration(minutes: 4, seconds: 5)).millisecondsSinceEpoch / 1000),
      ]);
      expectFits(p);
      expect(p.title, 'Claude - working 4 min');
      expect(p.bright, ['Running flutter test --coverage on the...']);
      expect(p.dim, ['"add mood charts to the Insights tab and', 'make the colors work in dark mode too"', 'vox_mobile - 14 steps']);
      expect(p.body.split('\n')[1], startsWith(G1Font.dimMark), reason: 'dim from the second line on');
    });

    test('done: the result, what you asked, project, time and steps', () {
      final p = ClaudePanel.build(
          now: now,
          last: WorkItem(
              id: 'w',
              ts: DateTime(2026, 10, 27, 14, 14).millisecondsSinceEpoch / 1000,
              project: 'even-suite',
              summary: 'Done.',
              prompt: 'fix the failing screenshot test in the even suite app please',
              ms: 125000,
              tools: 6));
      expectFits(p);
      expect(p.title, 'Claude - done 2:14pm');
      expect(p.bright, ['Done.']);
      expect(p.dim, ['You asked: fix the failing screenshot test', 'in the even suite app please', 'even-suite - 2 min - 6 steps']);
    });

    test('a long result keeps three lines and the details', () {
      final p = ClaudePanel.build(
          now: now,
          last: WorkItem(
              id: 'w',
              ts: now.millisecondsSinceEpoch / 1000,
              project: 'even-suite',
              summary: 'Pushed the fix; CI is green and the APK is published to the even-suite-latest release '
                  'with every test passing on both text sizes.',
              prompt: 'ship it',
              ms: 61000));
      expectFits(p);
      expect(p.bright.length, 3);
      expect(p.bright.last, endsWith('...'));
      expect(p.dim, ['even-suite - 1 min'], reason: 'no room left for "You asked"');
    });

    test('a message from a Claude Code session (/even)', () {
      final p = ClaudePanel.build(
          now: now,
          last: WorkItem(
              id: 'm',
              ts: now.millisecondsSinceEpoch / 1000,
              kind: 'message',
              project: 'even-suite',
              summary: 'Done: APK built.',
              text: 'Done: APK built. CI is green and the release is up - install it from the link on your phone.'));
      expectFits(p);
      expect(p.title, 'Claude Code - 2:20pm');
      expect(p.bright.first, 'Done: APK built. CI is green and the');
      expect(p.dim, ['even-suite']);
    });

    test('a question from this app, and nothing at all', () {
      final p = ClaudePanel.build(now: now, asking: ('Searching the web: boston weather tomorrow', 'what will the weather be in boston tomorrow and should I bring an umbrella or not'));
      expectFits(p);
      expect(p.title, 'Claude - answering you');
      expect(p.dim.length, 3);
      expectFits(ClaudePanel.build(now: now));
    });

    test('the dashboard sends it as the only note, in the dual layout', () async {
      final glasses = FakeGlasses();
      final control = FakeControl();
      final ble = BleManager(glasses, control: control);
      await control.connectNow();
      final d = Dashboard(Proto(ble))
        ..mode = 0 // ignored: Claude's panel always uses the dual layout
        ..jobs = [WorkJob(session: 's', project: 'p', step: 'Editing a.dart', started: now.millisecondsSinceEpoch / 1000)];
      await d.sync(force: true);
      final sent = glasses.sent['L']!;
      expect(sent.where((p) => p[0] == 0x06 && p[4] == 0x06).single.sublist(5), [1, 0], reason: 'dual, notes pane');
      final notes = sent.where((p) => p[0] == 0x1E).toList();
      expect(notes.map((p) => p[9]), [1, 2, 3, 4]);
      expect(notes.first[10], 1, reason: 'slot 1 set');
      expect(notes.skip(1).map((p) => p[10]), [0, 0, 0], reason: 'slots 2-4 deleted: no "1/3" counter');
      final title = String.fromCharCodes(notes.first.sublist(12, 12 + notes.first[11]));
      expect(title, startsWith('Claude - working'));
      d.dispose();
      ble.dispose();
    });
  });
}
