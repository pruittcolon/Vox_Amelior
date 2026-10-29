import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// The modifier keys of Termux's extra-keys row.
enum Mod { ctrl, alt, shift }

/// Off; armed for the next key only; or held until you tap it again.
enum ModState { off, once, locked }

/// One key to send to a terminal: a tmux key name ("Up", "C-c", "M-b"), or,
/// with [text], characters typed in as they are.
class KeyPress {
  const KeyPress(this.value, {this.text = false});

  final String value;
  final bool text;

  @override
  bool operator ==(Object other) => other is KeyPress && other.value == value && other.text == text;

  @override
  int get hashCode => Object.hash(value, text);

  @override
  String toString() => text ? 'text "$value"' : value;
}

/// Ctrl, Alt and Shift as on Termux's extra keys: tap one and it is armed for
/// the next key; tap again and it stays until you tap it a third time.
/// [named] and [char] turn the next key into what the terminal should get.
class KeyModifiers extends ChangeNotifier {
  final Map<Mod, ModState> _state = {for (final m in Mod.values) m: ModState.off};

  ModState state(Mod m) => _state[m]!;

  bool isOn(Mod m) => _state[m] != ModState.off;

  bool get any => Mod.values.any(isOn);

  /// off -> once -> locked -> off.
  void tap(Mod m) {
    _state[m] = switch (_state[m]!) {
      ModState.off => ModState.once,
      ModState.once => ModState.locked,
      ModState.locked => ModState.off,
    };
    notifyListeners();
  }

  /// A key went out: armed-for-one modifiers drop, held ones stay.
  void used() {
    var changed = false;
    for (final m in Mod.values) {
      if (_state[m] == ModState.once) {
        _state[m] = ModState.off;
        changed = true;
      }
    }
    if (changed) notifyListeners();
  }

  void clear() {
    if (!any) return;
    for (final m in Mod.values) {
      _state[m] = ModState.off;
    }
    notifyListeners();
  }

  String get _prefix => '${isOn(Mod.ctrl) ? 'C-' : ''}${isOn(Mod.alt) ? 'M-' : ''}${isOn(Mod.shift) ? 'S-' : ''}';

  /// A tmux key name (Up, Tab, Enter, PageDown ...) with the armed modifiers.
  KeyPress named(String key) {
    if (!any) return KeyPress(key);
    // Shift+Tab is its own key in tmux.
    if (key == 'Tab' && isOn(Mod.shift) && !isOn(Mod.ctrl) && !isOn(Mod.alt)) return const KeyPress('BTab');
    return KeyPress('$_prefix$key');
  }

  static final _alnum = RegExp(r'^[A-Za-z0-9]$');

  /// One typed character with the armed modifiers: Ctrl+C, Alt+B, Shift+a (= A).
  KeyPress char(String ch) {
    if (!any) return KeyPress(ch, text: true);
    final ctrl = isOn(Mod.ctrl), alt = isOn(Mod.alt), shift = isOn(Mod.shift);
    if (ch == ' ' && (ctrl || alt)) return KeyPress('${ctrl ? 'C-' : ''}${alt ? 'M-' : ''}Space');
    if (!_alnum.hasMatch(ch)) return KeyPress(shift ? ch.toUpperCase() : ch, text: true);
    if (ctrl || alt) {
      // A control character has no case; Alt keeps the letter as Shift makes it.
      final c = ctrl ? ch.toLowerCase() : (shift ? ch.toUpperCase() : ch);
      return KeyPress('${ctrl ? 'C-' : ''}${alt ? 'M-' : ''}$c');
    }
    return KeyPress(ch.toUpperCase(), text: true);
  }
}

/// Keeps a typed character out of a text box while Ctrl, Alt or Shift is armed
/// and hands it to [onChar] instead, to be sent as a key (Ctrl armed, then "c"
/// on the phone keyboard = Ctrl+C in the terminal).
class ModifiedCharFormatter extends TextInputFormatter {
  ModifiedCharFormatter(this.mods, this.onChar);

  final KeyModifiers mods;
  final void Function(String ch) onChar;

  @override
  TextEditingValue formatEditUpdate(TextEditingValue oldValue, TextEditingValue newValue) {
    if (!mods.any || newValue.text.length != oldValue.text.length + 1) return newValue;
    // No cursor reported: it was typed at the end.
    final cursor = newValue.selection.isValid ? newValue.selection.baseOffset : newValue.text.length;
    final at = cursor - 1;
    if (at < 0 || at >= newValue.text.length) return newValue;
    // Exactly one character was added where the cursor is.
    if (newValue.text.replaceRange(at, at + 1, '') != oldValue.text) return newValue;
    onChar(newValue.text[at]);
    return oldValue;
  }
}
