import 'dart:convert';

import 'dashboard.dart';
import 'g1_font.dart';

/// What can go on your look-up screen.
enum LookKind {
  time('Time', 1),
  bigClock('Big clock', 5),
  date('Date', 1),
  weather('Weather', 1),
  claude('Claude', 3),
  work('Work', 1),
  phone('Notifications', 2),
  battery('Battery', 1),
  text('Text', 1),
  line('Line', 1),
  vline('Divider', 5);

  const LookKind(this.label, this.defaultRows);
  final String label;
  final int defaultRows;

  /// Styles to choose from (first = default), with a label for each.
  List<(String, String)> get styles => switch (this) {
        LookKind.time => const [('auto', '2:20 PM'), ('short', '2:20'), ('24', '14:20')],
        LookKind.bigClock => const [('slim', 'Slim'), ('bold', 'Bold'), ('dots', 'Dots')],
        LookKind.date => const [('short', 'Tue Oct 27'), ('long', 'Tuesday, Oct 27'), ('day', 'Tuesday'), ('num', '10/27'), ('eu', '27 Oct')],
        LookKind.weather => const [('full', '72F Sunny'), ('temp', '72F'), ('word', 'Sunny')],
        LookKind.battery => const [('label', 'Phone 82%'), ('bare', '82%')],
        LookKind.line => const [('-', '------'), ('_', '______'), ('=', '======'), ('.', '......'), ('~', '~~~~~~')],
        _ => const [('plain', 'Plain')],
      };

  /// Can it be more than one row tall (and how tall at most)?
  bool get tall =>
      this == LookKind.claude ||
      this == LookKind.phone ||
      this == LookKind.text ||
      this == LookKind.work ||
      this == LookKind.vline;
}

/// One thing placed on the look-up screen: a box [x, x+w) on rows
/// [row, row+h), drawn the way the glasses draw text.
class LookItem {
  const LookItem(this.kind,
      {required this.row, required this.x, this.w = 0, this.h = 1, String? style, this.align = 'l', this.text = ''})
      : style = style ?? '';

  final LookKind kind;
  final int row;
  final int x;

  /// Width in glasses pixels (0 = as wide as its text).
  final int w;
  final int h;
  final String style;

  /// l, c or r: where the text sits in its box.
  final String align;

  /// Your own words (the Text item).
  final String text;

  String get styleOrDefault => style.isEmpty ? kind.styles.first.$1 : style;
  int get bottom => row + h;

  LookItem copyWith({int? row, int? x, int? w, int? h, String? style, String? align, String? text}) => LookItem(kind,
      row: row ?? this.row,
      x: x ?? this.x,
      w: w ?? this.w,
      h: h ?? this.h,
      style: style ?? this.style,
      align: align ?? this.align,
      text: text ?? this.text);

  Map<String, Object> toJson() => {
        'k': kind.name,
        'r': row,
        'x': x,
        'w': w,
        'h': h,
        if (style.isNotEmpty) 's': style,
        if (align != 'l') 'a': align,
        if (text.isNotEmpty) 't': text,
      };

  static LookItem? fromJson(Object? m) {
    if (m is! Map) return null;
    final kind = LookKind.values.where((k) => k.name == m['k']).firstOrNull;
    if (kind == null) return null;
    int n(Object? v, int d) => v is num ? v.toInt() : d;
    return LookItem(kind,
        row: n(m['r'], 0),
        x: n(m['x'], 0),
        w: n(m['w'], 0),
        h: n(m['h'], 1),
        style: '${m['s'] ?? ''}',
        align: '${m['a'] ?? 'l'}',
        text: '${m['t'] ?? ''}');
  }

  @override
  bool operator ==(Object other) =>
      other is LookItem &&
      other.kind == kind &&
      other.row == row &&
      other.x == x &&
      other.w == w &&
      other.h == h &&
      other.style == style &&
      other.align == align &&
      other.text == text;

  @override
  int get hashCode => Object.hash(kind, row, x, w, h, style, align, text);
}

/// Your look-up screen: what goes where. The glasses' text screen is 5 rows
/// of 576 px; a space is 6 px, so things line up on a 6 px grid.
class LookLayout {
  const LookLayout(this.items);

  final List<LookItem> items;

  static const widthPx = 576;
  static const maxPx = 570; // a little margin: a longer line would wrap
  static const rows = 5;
  static const grid = 6;

  static int snap(num px) => ((px / grid).round() * grid).clamp(0, widthPx - grid);

  /// [item] moved, resized or restyled to fit the screen.
  static LookItem fit(LookItem item) {
    final h = item.kind == LookKind.bigClock ? rows : item.h.clamp(1, item.kind.tall ? rows : 1);
    final row = item.row.clamp(0, rows - h);
    final x = snap(item.x);
    final maxW = widthPx - x;
    final minW = item.kind == LookKind.vline ? grid : 4 * grid;
    final w = item.w == 0 ? 0 : snap(item.w).clamp(minW < maxW ? minW : maxW, maxW);
    return item.copyWith(row: row, x: x, w: w, h: h);
  }

  LookLayout withItem(int i, LookItem item) => LookLayout([for (var j = 0; j < items.length; j++) j == i ? fit(item) : items[j]]);
  LookLayout add(LookItem item) => LookLayout([...items, fit(item)]);
  LookLayout remove(int i) => LookLayout([for (var j = 0; j < items.length; j++) if (j != i) items[j]]);

  /// Where a new [kind] fits: the first spot on the screen with room for it.
  LookItem placeNew(LookKind kind) {
    final w = defaultWidth(kind);
    final h = kind == LookKind.bigClock ? rows : kind.defaultRows;
    for (var row = 0; row + h <= rows; row++) {
      for (var x = 0; x + w <= widthPx; x += grid * 4) {
        final clash = items.any((o) =>
            o.row < row + h && row < o.bottom && o.x < x + w && x < o.x + (o.w == 0 ? defaultWidth(o.kind) : o.w));
        if (!clash) return fit(LookItem(kind, row: row, x: x, w: w, h: h));
      }
    }
    return fit(LookItem(kind, row: 0, x: 0, w: w, h: h));
  }

  static int defaultWidth(LookKind kind) => switch (kind) {
        LookKind.time => 120,
        LookKind.bigClock => 216,
        LookKind.date => 168,
        LookKind.weather => 150,
        LookKind.battery => 126,
        LookKind.vline => 12,
        LookKind.text => 240,
        _ => 372,
      };

  String encode() => jsonEncode([for (final i in items) i.toJson()]);

  /// A saved layout ([fallback] when there is none or it is broken).
  static LookLayout parse(String s, {LookLayout? fallback}) {
    final base = fallback ?? presets['split']!;
    if (s.trim().isEmpty) return base;
    try {
      final list = jsonDecode(s);
      if (list is! List) return base;
      final items = [for (final m in list) LookItem.fromJson(m)].whereType<LookItem>().map(fit).toList();
      return LookLayout(items);
    } catch (_) {
      return base;
    }
  }

  @override
  bool operator ==(Object other) =>
      other is LookLayout && other.items.length == items.length && [for (var i = 0; i < items.length; i++) other.items[i] == items[i]].every((e) => e);

  @override
  int get hashCode => Object.hashAll(items);

  /// Ready-made screens to start from.
  static final presets = <String, LookLayout>{
    'split': const LookLayout([
      LookItem(LookKind.time, row: 0, x: 0, w: 156),
      LookItem(LookKind.date, row: 1, x: 0, w: 156),
      LookItem(LookKind.weather, row: 2, x: 0, w: 156),
      LookItem(LookKind.battery, row: 4, x: 0, w: 156),
      LookItem(LookKind.claude, row: 0, x: 180, w: 390, h: 5),
    ]),
    'clock': const LookLayout([
      LookItem(LookKind.bigClock, row: 0, x: 0, h: 5, style: 'slim'),
      LookItem(LookKind.date, row: 0, x: 240, w: 330),
      LookItem(LookKind.weather, row: 1, x: 240, w: 330),
      LookItem(LookKind.line, row: 2, x: 240, w: 330, style: '.'),
      LookItem(LookKind.claude, row: 3, x: 240, w: 330, h: 2),
    ]),
    'claude': const LookLayout([
      LookItem(LookKind.time, row: 0, x: 0, w: 132),
      LookItem(LookKind.date, row: 0, x: 138, w: 234, style: 'long'),
      LookItem(LookKind.weather, row: 0, x: 378, w: 192, align: 'r', style: 'temp'),
      LookItem(LookKind.line, row: 1, x: 0, w: 570),
      LookItem(LookKind.claude, row: 2, x: 0, w: 570, h: 3),
    ]),
    'calm': const LookLayout([
      LookItem(LookKind.time, row: 1, x: 0, w: 570, align: 'c'),
      LookItem(LookKind.date, row: 2, x: 0, w: 570, align: 'c', style: 'long'),
      LookItem(LookKind.weather, row: 3, x: 0, w: 570, align: 'c'),
    ]),
    'busy': const LookLayout([
      LookItem(LookKind.time, row: 0, x: 0, w: 120),
      LookItem(LookKind.date, row: 0, x: 126, w: 168),
      LookItem(LookKind.weather, row: 0, x: 378, w: 192, align: 'r'),
      LookItem(LookKind.claude, row: 1, x: 0, w: 570, h: 2),
      LookItem(LookKind.work, row: 3, x: 0, w: 570),
      LookItem(LookKind.phone, row: 4, x: 0, w: 570, h: 1),
    ]),
  };

  static const presetNames = {
    'split': 'Split',
    'clock': 'Big clock',
    'claude': 'Claude first',
    'calm': 'Calm',
    'busy': 'Everything',
  };
}

/// What the screen shows right now.
class LookData {
  const LookData({
    required this.now,
    this.h24 = false,
    this.weather,
    this.fahrenheit = true,
    this.claude = const [],
    this.work = const [],
    this.phone = const [],
    this.battery,
  });

  final DateTime now;
  final bool h24;
  final Weather? weather;
  final bool fahrenheit;

  /// Claude's panel: a status line, then the details.
  final List<String> claude;
  final List<String> work;

  /// Phone notifications, newest first.
  final List<String> phone;
  final int? battery;

  /// Sample data for previews.
  static LookData sample(DateTime now) => LookData(
        now: now,
        weather: Weather(22, G1Icon.sunny, 'Here', now),
        claude: const ['Claude · working 42s', 'Running flutter test', '"fix the failing test"'],
        work: const ['Done 5 min ago: pushed the fix'],
        phone: const ['Mom: dinner at 7?', 'Slack: standup moved'],
        battery: 82,
      );
}

/// One piece of a row as it lands on the glasses (for previews).
class LookPiece {
  const LookPiece(this.item, this.row, this.x, this.text);
  final int item;
  final int row;
  final int x;
  final String text;
}

/// Turns a [LookLayout] into the 5 lines of text the glasses draw: each
/// piece is moved to its place with spaces (6 px each), cut to its box, and
/// never pushed past the right edge. Letters are 4-16 px wide, so a piece
/// after other text on its row can land up to 4 px right of its spot.
class LookRenderer {
  LookRenderer._();

  static List<String> render(LookLayout layout, LookData data) => _build(layout, data).$1;

  static List<LookPiece> pieces(LookLayout layout, LookData data) => _build(layout, data).$2;

  static (List<String>, List<LookPiece>) _build(LookLayout layout, LookData data) {
    // Each item's lines for each row it covers.
    final byRow = List.generate(LookLayout.rows, (_) => <(int, LookItem, String)>[]);
    for (var i = 0; i < layout.items.length; i++) {
      final item = layout.items[i];
      final lines = content(item, data);
      for (var k = 0; k < item.h && item.row + k < LookLayout.rows; k++) {
        final text = k < lines.length ? lines[k] : '';
        if (text.trim().isEmpty && item.kind != LookKind.bigClock) continue;
        byRow[item.row + k].add((i, item, text));
      }
    }
    final out = <String>[];
    final pieces = <LookPiece>[];
    for (var r = 0; r < LookLayout.rows; r++) {
      final row = byRow[r]..sort((a, b) => a.$2.x.compareTo(b.$2.x));
      final b = StringBuffer();
      var cur = 0;
      for (final (i, item, raw) in row) {
        final boxW = item.w == 0 ? LookLayout.maxPx - item.x : item.w;
        final boxRight = (item.x + boxW).clamp(0, LookLayout.maxPx);
        final clock = item.kind == LookKind.bigClock;
        var text = clock ? raw : raw.trim();
        var want = item.x;
        final tw = G1Font.width(text);
        if (!clock && tw < boxW) {
          if (item.align == 'r') want = item.x + boxW - tw;
          if (item.align == 'c') want = item.x + (boxW - tw) ~/ 2;
        }
        var start = cur == 0 ? want : (want > cur + LookLayout.grid ? want : cur + LookLayout.grid);
        if (start < cur) start = cur;
        final spaces = ((start - cur) / LookLayout.grid).ceil();
        final at = cur + spaces * LookLayout.grid;
        final room = (clock ? LookLayout.maxPx : boxRight) - at;
        if (room < LookLayout.grid * 2) continue; // pushed out of its box
        text = clock ? _clip(text, room) : G1Font.fit(text, room);
        if (text.trim().isEmpty) continue;
        b.write(' ' * spaces);
        b.write(text);
        pieces.add(LookPiece(i, r, at, text));
        cur = at + G1Font.width(text);
      }
      out.add(b.toString().trimRight());
    }
    return (out, pieces);
  }

  static String _clip(String s, int px) {
    var t = s;
    while (t.isNotEmpty && G1Font.width(t) > px) {
      t = t.substring(0, t.length - 1);
    }
    return t;
  }

  /// The lines an item shows (one per row it covers).
  static List<String> content(LookItem item, LookData d) {
    final w = item.w == 0 ? LookLayout.maxPx - item.x : item.w;
    final style = item.styleOrDefault;
    switch (item.kind) {
      case LookKind.time:
        return [time(d.now, style == 'auto' ? (d.h24 ? '24' : 'auto') : style)];
      case LookKind.bigClock:
        return bigClock(time(d.now, d.h24 ? '24' : 'short'), style);
      case LookKind.date:
        return [date(d.now, style)];
      case LookKind.weather:
        final wx = d.weather;
        if (wx == null) return const [''];
        final t = d.fahrenheit ? (wx.tempC * 9 / 5 + 32).round() : wx.tempC;
        final temp = '$t${d.fahrenheit ? 'F' : 'C'}';
        final word = weatherWord(wx.icon);
        return [
          switch (style) {
            'temp' => temp,
            'word' => word,
            _ => word.isEmpty ? temp : '$temp $word',
          }
        ];
      case LookKind.battery:
        final b = d.battery;
        if (b == null) return const [''];
        return [style == 'bare' ? '$b%' : 'Phone $b%'];
      case LookKind.claude:
        return d.claude.isEmpty ? const ['Claude is ready'] : d.claude;
      case LookKind.work:
        return d.work;
      case LookKind.phone:
        return d.phone;
      case LookKind.text:
        return G1Font.wrapMax(item.text, w, item.h);
      case LookKind.line:
        final ch = style.isEmpty ? '-' : style.substring(0, 1);
        final n = (w / G1Font.width(ch)).floor();
        return [ch * n.clamp(1, 200)];
      case LookKind.vline:
        return List.filled(item.h, '|');
    }
  }

  static const _days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  static const _months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  static String time(DateTime t, String style) {
    final m = t.minute.toString().padLeft(2, '0');
    if (style == '24') return '${t.hour.toString().padLeft(2, '0')}:$m';
    final h = t.hour % 12 == 0 ? 12 : t.hour % 12;
    return style == 'short' ? '$h:$m' : '$h:$m ${t.hour < 12 ? 'AM' : 'PM'}';
  }

  static String date(DateTime t, String style) {
    final day = _days[t.weekday - 1];
    final mon = _months[t.month - 1];
    return switch (style) {
      'long' => '$day, $mon ${t.day}',
      'day' => day,
      'num' => '${t.month}/${t.day}',
      'eu' => '${t.day} $mon',
      _ => '${day.substring(0, 3)} $mon ${t.day}',
    };
  }

  static String weatherWord(int icon) => switch (icon) {
        G1Icon.sunny => 'Sunny',
        G1Icon.night => 'Clear',
        G1Icon.clouds => 'Cloudy',
        G1Icon.drizzle || G1Icon.heavyDrizzle => 'Drizzle',
        G1Icon.rain || G1Icon.heavyRain => 'Rain',
        G1Icon.thunder || G1Icon.thunderstorm => 'Storm',
        G1Icon.snow => 'Snow',
        G1Icon.mist || G1Icon.fog => 'Fog',
        G1Icon.freezingRain => 'Ice',
        _ => '',
      };

  // 3 x 5 pixel digits; every pixel is a 12 px letter so the columns line up.
  static const _font = {
    '0': ['111', '101', '101', '101', '111'],
    '1': ['010', '110', '010', '010', '111'],
    '2': ['111', '001', '111', '100', '111'],
    '3': ['111', '001', '111', '001', '111'],
    '4': ['101', '101', '111', '001', '001'],
    '5': ['111', '100', '111', '001', '111'],
    '6': ['111', '100', '111', '101', '111'],
    '7': ['111', '001', '001', '001', '001'],
    '8': ['111', '101', '111', '101', '111'],
    '9': ['111', '101', '111', '001', '111'],
    ':': ['0', '1', '0', '1', '0'],
  };

  /// [hm] ("2:20") as 5 rows of big pixel digits.
  static List<String> bigClock(String hm, String style) {
    final on = switch (style) { 'bold' => 'HH', 'dots' => 'OO', _ => 'H' };
    final off = ' ' * (on.length * 2); // a 12 px letter = two spaces
    final gap = '  ';
    return [
      for (var r = 0; r < 5; r++)
        [
          for (final ch in hm.split(''))
            if (_font[ch] != null) _font[ch]![r].split('').map((p) => p == '1' ? on : off).join(),
        ].join(gap).trimRight(),
    ];
  }

  /// How wide the big clock is (for the editor).
  static int bigClockWidth(String hm, String style) =>
      bigClock(hm, style).fold(0, (n, l) => G1Font.width(l) > n ? G1Font.width(l) : n);
}
