import 'package:even_suite/core/bridge/key_modifiers.dart';
import 'package:even_suite/features/claude/terminal_pane.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

/// The same rule Agent Hub applies to keys from the app (termux/agent-hub/server.py).
final _hubModifierKey = RegExp(
    r'^(?:(?:[CMS]-){1,3}(?:[A-Za-z0-9]|Enter|Escape|Up|Down|Left|Right|Tab|BTab|Space|BSpace|DC|IC|Home|End|PageUp|PageDown|F[1-9]|F1[0-2])|F[1-9]|F1[0-2]|IC)$');
const _hubPlainKeys = {
  'Enter', 'Escape', 'Up', 'Down', 'Left', 'Right', 'Tab', 'BTab', 'Space', 'BSpace', 'DC', 'Home', 'End', //
  'PageUp', 'PageDown', 'C-c', 'C-d', 'C-l', 'C-o', 'C-r', 'C-t', 'C-u', 'C-a', 'C-e', 'y', 'n',
  '1', '2', '3', '4', '5', '6', '7', '8', '9',
};

void main() {
  group('Termux-style Ctrl, Alt and Shift', () {
    test('tap arms the next key, tap again holds it, a third time lets go', () {
      final m = KeyModifiers();
      expect(m.any, isFalse);
      m.tap(Mod.ctrl);
      expect(m.state(Mod.ctrl), ModState.once);
      m.used();
      expect(m.state(Mod.ctrl), ModState.off, reason: 'armed for one key only');
      m.tap(Mod.ctrl);
      m.tap(Mod.ctrl);
      expect(m.state(Mod.ctrl), ModState.locked);
      m.used();
      m.used();
      expect(m.state(Mod.ctrl), ModState.locked, reason: 'held until tapped again');
      m.tap(Mod.ctrl);
      expect(m.state(Mod.ctrl), ModState.off);
      expect(m.any, isFalse);
    });

    test('used() lets go of the one-shot ones and keeps the held ones', () {
      final m = KeyModifiers()
        ..tap(Mod.ctrl)
        ..tap(Mod.alt)
        ..tap(Mod.alt)
        ..tap(Mod.shift);
      m.used();
      expect(m.state(Mod.ctrl), ModState.off);
      expect(m.state(Mod.shift), ModState.off);
      expect(m.state(Mod.alt), ModState.locked);
      m.clear();
      expect(m.any, isFalse);
    });

    test('they tell their listeners', () {
      final m = KeyModifiers();
      var n = 0;
      m.addListener(() => n++);
      m.tap(Mod.shift);
      m.used();
      m.used(); // nothing left to let go of
      expect(n, 2);
    });

    test('nothing armed: keys go as they are', () {
      final m = KeyModifiers();
      expect(m.named('Up'), const KeyPress('Up'));
      expect(m.char('a'), const KeyPress('a', text: true));
      expect(m.char('/'), const KeyPress('/', text: true));
    });

    test('Ctrl and a letter is a control key, whatever the case', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      expect(m.char('c'), const KeyPress('C-c'));
      expect(m.char('C'), const KeyPress('C-c'));
      expect(m.char('4'), const KeyPress('C-4'));
    });

    test('Alt and a letter; with Shift the letter stays a capital', () {
      final m = KeyModifiers()..tap(Mod.alt);
      expect(m.char('b'), const KeyPress('M-b'));
      m.tap(Mod.shift);
      expect(m.char('b'), const KeyPress('M-B'));
    });

    test('Ctrl and Alt together', () {
      final m = KeyModifiers()
        ..tap(Mod.ctrl)
        ..tap(Mod.alt);
      expect(m.char('x'), const KeyPress('C-M-x'));
      expect(m.named('Left'), const KeyPress('C-M-Left'));
    });

    test('Shift alone types the capital', () {
      final m = KeyModifiers()..tap(Mod.shift);
      expect(m.char('a'), const KeyPress('A', text: true));
      expect(m.char('/'), const KeyPress('/', text: true), reason: 'nothing to shift without a keyboard layout');
    });

    test('Ctrl and space; punctuation with Ctrl is just typed', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      expect(m.char(' '), const KeyPress('C-Space'));
      expect(m.char('/'), const KeyPress('/', text: true));
    });

    test('named keys carry the modifiers; Shift+Tab is its own key', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      expect(m.named('Right'), const KeyPress('C-Right'));
      expect(m.named('PageUp'), const KeyPress('C-PageUp'));
      m.clear();
      m.tap(Mod.shift);
      expect(m.named('Tab'), const KeyPress('BTab'));
      expect(m.named('Up'), const KeyPress('S-Up'));
      m.tap(Mod.ctrl);
      expect(m.named('Tab'), const KeyPress('C-S-Tab'));
    });

    test('everything it can produce is a key Agent Hub accepts', () {
      final names = [
        ...ExtraKeys.gridKeys,
        'Space',
        for (final c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('')) c,
      ];
      for (var bits = 0; bits < 8; bits++) {
        final m = KeyModifiers();
        for (final mod in Mod.values) {
          if (bits & (1 << mod.index) != 0) m.tap(mod);
        }
        for (final name in names) {
          final p = name.length == 1 ? m.char(name) : m.named(name);
          if (p.text) continue; // typed in, not a key name
          expect(_hubModifierKey.hasMatch(p.value) || _hubPlainKeys.contains(p.value), isTrue,
              reason: '$name with modifiers $bits became ${p.value}');
        }
        expect(_hubModifierKey.hasMatch(m.char(' ').value) || m.char(' ').text, isTrue);
      }
    });

    test('the bar\'s own keys are all keys Agent Hub accepts', () {
      for (final k in [...ExtraKeys.gridKeys, ...ExtraKeys.shortcuts.where((s) => !s.$3).map((s) => s.$2)]) {
        expect(_hubPlainKeys.contains(k), isTrue, reason: '$k is not on Agent Hub\'s list');
      }
    });
  });

  group('a letter typed while Ctrl, Alt or Shift is armed', () {
    TextEditingValue v(String text, [int? at]) =>
        TextEditingValue(text: text, selection: TextSelection.collapsed(offset: at ?? text.length));

    test('goes to the terminal as that key and stays out of the box', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      final got = <String>[];
      final f = ModifiedCharFormatter(m, got.add);
      final out = f.formatEditUpdate(v('ab'), v('abc'));
      expect(got, ['c']);
      expect(out.text, 'ab');
    });

    test('works for a box that does not say where the cursor is', () {
      final m = KeyModifiers()..tap(Mod.alt);
      final got = <String>[];
      final out = ModifiedCharFormatter(m, got.add).formatEditUpdate(TextEditingValue.empty, const TextEditingValue(text: 'b'));
      expect(got, ['b']);
      expect(out.text, '');
    });

    test('typed in the middle', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      final got = <String>[];
      final out = ModifiedCharFormatter(m, got.add).formatEditUpdate(v('ac', 1), v('abc', 2));
      expect(got, ['b']);
      expect(out.text, 'ac');
    });

    test('with nothing armed the box works as usual', () {
      final got = <String>[];
      final out = ModifiedCharFormatter(KeyModifiers(), got.add).formatEditUpdate(v('ab'), v('abc'));
      expect(got, isEmpty);
      expect(out.text, 'abc');
    });

    test('a paste, a delete or a replaced word is left alone', () {
      final m = KeyModifiers()..tap(Mod.ctrl);
      final got = <String>[];
      final f = ModifiedCharFormatter(m, got.add);
      expect(f.formatEditUpdate(v('ab'), v('abcd')).text, 'abcd');
      expect(f.formatEditUpdate(v('abc'), v('ab')).text, 'ab');
      expect(f.formatEditUpdate(v('abc'), v('abd')).text, 'abd');
      expect(got, isEmpty);
    });
  });
}
