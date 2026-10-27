import 'package:even_suite/core/glasses/g1_font.dart';
import 'package:even_suite/core/glasses/look_layout.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final now = DateTime(2026, 10, 27, 14, 20);
  final data = LookData.sample(now);

  group('look-up layout', () {
    test('every preset fits the glasses: 5 rows, none too wide', () {
      for (final e in LookLayout.presets.entries) {
        final rows = LookRenderer.render(e.value, data);
        expect(rows.length, 5, reason: e.key);
        for (final r in rows) {
          expect(G1Font.width(r), lessThanOrEqualTo(LookLayout.maxPx), reason: '${e.key}: "$r"');
        }
      }
    });

    test('things land where you put them (spaces are 6 px)', () {
      const layout = LookLayout([
        LookItem(LookKind.text, row: 1, x: 180, w: 300, text: 'Hello'),
      ]);
      final rows = LookRenderer.render(layout, data);
      expect(rows[1], '${' ' * 30}Hello');
      expect(rows[0], isEmpty);
      final piece = LookRenderer.pieces(layout, data).single;
      expect((piece.row, piece.x), (1, 180));
    });

    test('right and centre alignment inside the box', () {
      const layout = LookLayout([
        LookItem(LookKind.text, row: 0, x: 0, w: 570, text: 'end', align: 'r'),
        LookItem(LookKind.text, row: 1, x: 0, w: 570, text: 'mid', align: 'c'),
      ]);
      final pieces = LookRenderer.pieces(layout, data);
      final end = pieces.firstWhere((p) => p.row == 0);
      expect(end.x + G1Font.width('end'), inInclusiveRange(570 - 6, 570));
      final mid = pieces.firstWhere((p) => p.row == 1);
      expect((mid.x + G1Font.width('mid') / 2 - 285).abs(), lessThanOrEqualTo(6));
    });

    test('overlapping things are pushed right, never on top of each other', () {
      const layout = LookLayout([
        LookItem(LookKind.text, row: 0, x: 0, w: 300, text: 'A long first piece'),
        LookItem(LookKind.text, row: 0, x: 12, w: 300, text: 'second'),
      ]);
      final pieces = LookRenderer.pieces(layout, data);
      expect(pieces.length, 2);
      expect(pieces[1].x, greaterThanOrEqualTo(pieces[0].x + G1Font.width(pieces[0].text) + 6));
    });

    test('the time, the date and the weather in each style', () {
      expect(LookRenderer.time(now, 'auto'), '2:20 PM');
      expect(LookRenderer.time(now, 'short'), '2:20');
      expect(LookRenderer.time(now, '24'), '14:20');
      expect(LookRenderer.date(now, 'short'), 'Tue Oct 27');
      expect(LookRenderer.date(now, 'long'), 'Tuesday, Oct 27');
      expect(LookRenderer.date(now, 'num'), '10/27');
      final wx = LookRenderer.content(const LookItem(LookKind.weather, row: 0, x: 0, w: 200), data).single;
      expect(wx, '72F Sunny');
    });

    test('the big clock lines up: every letter 12 px, so the columns stay put', () {
      for (final style in ['slim', 'bold', 'dots']) {
        final rows = LookRenderer.bigClock('12:59', style);
        expect(rows.length, 5);
        for (final r in rows) {
          expect(r.replaceAll(RegExp('[HO ]'), ''), isEmpty, reason: r);
          expect(G1Font.width(r) % 6, 0);
        }
      }
      expect(LookRenderer.bigClockWidth('12:59', 'slim'), lessThan(LookLayout.widthPx ~/ 2));
    });

    test('saved and loaded as is; a broken save falls back to Split', () {
      final layout = LookLayout.presets['busy']!;
      expect(LookLayout.parse(layout.encode()), layout);
      expect(LookLayout.parse('not json'), LookLayout.presets['split']);
      expect(LookLayout.parse(''), LookLayout.presets['split']);
    });

    test('moved or resized past the edge, it snaps back onto the screen', () {
      final item = LookLayout.fit(const LookItem(LookKind.claude, row: 4, x: 1000, w: 9999, h: 3));
      expect(item.row, 2);
      expect(item.x % 6, 0);
      expect(item.x + item.w, lessThanOrEqualTo(LookLayout.widthPx));
      expect(LookLayout.fit(const LookItem(LookKind.time, row: 2, x: 31, w: 100, h: 4)).h, 1, reason: 'one row only');
    });

    test('a new item goes where there is room', () {
      final layout = LookLayout.presets['calm']!;
      final item = layout.placeNew(LookKind.battery);
      expect(item.row, greaterThanOrEqualTo(0));
      final clash = layout.items.any((o) =>
          o.row < item.row + item.h && item.row < o.bottom && o.x < item.x + item.w && item.x < o.x + o.w);
      expect(clash, isFalse);
    });
  });
}
