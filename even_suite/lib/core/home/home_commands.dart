import 'voice_monkey.dart';

enum HomeKind {
  /// A Voice Monkey device (an Alexa routine): the lights.
  lights,

  /// Vizio remote keys, in order.
  tvKeys,

  /// Switch the TV to an input (hdmi1, hdmi2, ...).
  tvInput,

  /// Remote mode: from now on "up", "OK", "back" ... go to the TV.
  tvRemote,

  /// Leave remote mode.
  tvDone,
}

/// One thing to do at home, worked out from what was said.
class HomeCommand {
  const HomeCommand(this.kind, {this.device = '', this.label = '', this.keys = const [], this.input = ''});

  final HomeKind kind;

  /// Voice Monkey device id (lights).
  final String device;

  /// Said back on the glasses ("All lights", "TV volume up").
  final String label;

  /// Vizio key names ([VizioTv.keyCodes]).
  final List<String> keys;

  /// Vizio input (tvInput).
  final String input;

  @override
  String toString() => '$kind $device$keys$input';
}

/// Lights and the TV by voice, without Claude - the same idea as the old
/// app's Alexa and Roku modes:
///
/// * "all lights off", "turn on the kitchen light", "lights" - a Voice
///   Monkey device ([VoiceMonkey.parse]); on and off trigger the same
///   routine, as before,
/// * "TV off", "TV volume up" (3 presses), "mute the TV", "TV HDMI 2",
///   "TV home" ...,
/// * "TV remote" - then just "up", "down", "OK", "back", "volume up",
///   "pause" ... until "done".
class HomeCommands {
  HomeCommands._();

  static String norm(String s) =>
      s.toLowerCase().replaceAll(RegExp(r"[^a-z0-9' ]"), ' ').replaceAll(RegExp(r'\s+'), ' ').trim();

  /// "tv", "the tv", "television", "vizio" (also as the recognizer often writes them).
  static const _tv = r"(?:the )?(?:tv|t v|tv's|television|vizio|visio|vizzio)";

  static String _strip(String t) => t.replaceFirst(RegExp(r'^(please |hey |okay |ok )+'), '').replaceFirst(RegExp(r' please$'), '');

  /// Anything for the lights or the TV. [trigger]: heard without the wake
  /// word - then a bare "TV" or "remote" does not start remote mode.
  static HomeCommand? parse(String said, List<LightDevice> lights, {bool trigger = false}) =>
      tv(said, trigger: trigger) ?? light(said, lights);

  /// "all lights off", "turn on the kitchen light": that device.
  static HomeCommand? light(String said, List<LightDevice> devices) {
    var t = _strip(norm(said));
    t = t.replaceFirst(RegExp(r'^(turn|switch|shut) (on|off) '), '').replaceFirst(RegExp(r'^(turn|switch) '), '');
    t = t.replaceFirst(RegExp(r' (on|off)$'), '');
    t = t.replaceAll(RegExp(r'\bthe\b'), '').replaceAll(RegExp(r'\s+'), ' ').trim();
    if (t.isEmpty) return null;
    for (final d in devices) {
      for (final p in d.phrases) {
        if (t == norm(p)) return HomeCommand(HomeKind.lights, device: d.device, label: d.name);
      }
    }
    return null;
  }

  /// Commands that name the TV.
  static HomeCommand? tv(String said, {bool trigger = false}) {
    final t = _strip(norm(said));
    HomeCommand keys(List<String> k, String label) => HomeCommand(HomeKind.tvKeys, keys: k, label: label);
    if (RegExp('^$_tv (remote|mode)\$|^$_tv remote mode\$').hasMatch(t) ||
        (!trigger && RegExp('^$_tv\$|^remote( mode)?\$').hasMatch(t))) {
      return const HomeCommand(HomeKind.tvRemote, label: 'TV remote');
    }
    if (RegExp('^(turn|switch|shut) off $_tv\$|^(turn|switch|shut) $_tv off\$|^$_tv (power )?off\$').hasMatch(t)) {
      return keys(const ['POWER_OFF'], 'TV off');
    }
    if (RegExp('^(turn|switch) on $_tv\$|^(turn|switch) $_tv on\$|^$_tv (power )?on\$').hasMatch(t)) {
      return keys(const ['POWER_ON'], 'TV on');
    }
    if (RegExp('^$_tv (volume )?(up|louder)\$|^turn up $_tv( volume)?\$|^turn $_tv up\$').hasMatch(t)) {
      return keys(const ['VOL_UP', 'VOL_UP', 'VOL_UP'], 'TV volume up');
    }
    if (RegExp('^$_tv (volume )?(down|quieter)\$|^turn down $_tv( volume)?\$|^turn $_tv down\$').hasMatch(t)) {
      return keys(const ['VOL_DOWN', 'VOL_DOWN', 'VOL_DOWN'], 'TV volume down');
    }
    if (RegExp('^(un)?mute $_tv\$|^$_tv (un)?mute\$').hasMatch(t)) return keys(const ['MUTE'], 'TV mute');
    final hdmi = RegExp('^(?:$_tv|switch $_tv to|put $_tv on) (?:input )?(?:hdmi|hd mi|h d m i) ?(one|1|two|2|to|three|3)\$')
        .firstMatch(t);
    if (hdmi != null) {
      final n = switch (hdmi.group(1)) { 'one' || '1' => 1, 'two' || '2' || 'to' => 2, _ => 3 };
      return HomeCommand(HomeKind.tvInput, input: 'hdmi$n', label: 'TV HDMI $n');
    }
    final key = RegExp('^$_tv (home|back|pause|play|input|menu|exit)\$').firstMatch(t);
    if (key != null) {
      final k = key.group(1)!;
      return keys([k.toUpperCase()], 'TV $k');
    }
    return null;
  }

  /// In remote mode: the TV command for what was said, or [HomeKind.tvDone].
  static HomeCommand? remote(String said) {
    final t = _strip(norm(said)).replaceFirst(RegExp('^$_tv '), '').replaceFirst(RegExp(r'^(go|press|hit) '), '');
    HomeCommand keys(List<String> k, String label) => HomeCommand(HomeKind.tvKeys, keys: k, label: label);
    if (RegExp(r"^(done|exit|stop|terminate|quit|close|finish(ed)?|that's (it|all)|leave|remote off)( remote)?$").hasMatch(t)) {
      return const HomeCommand(HomeKind.tvDone, label: 'TV remote off');
    }
    if (RegExp(r'^(volume up|louder|turn it up|vol up)$').hasMatch(t)) {
      return keys(const ['VOL_UP', 'VOL_UP', 'VOL_UP'], 'Volume up');
    }
    if (RegExp(r'^(volume down|quieter|turn it down|vol down)$').hasMatch(t)) {
      return keys(const ['VOL_DOWN', 'VOL_DOWN', 'VOL_DOWN'], 'Volume down');
    }
    final simple = {
      r'^(up|app)$': ('UP', 'Up'),
      r'^(down)$': ('DOWN', 'Down'),
      r'^(left)$': ('LEFT', 'Left'),
      r'^(right|write)$': ('RIGHT', 'Right'),
      r"^(ok|okay|select|enter|choose|click|that one)$": ('OK', 'OK'),
      r'^(back|go back)$': ('BACK', 'Back'),
      r'^(home)$': ('HOME', 'Home'),
      r'^(mute|unmute)$': ('MUTE', 'Mute'),
      r'^(pause)$': ('PAUSE', 'Pause'),
      r'^(play|resume)$': ('PLAY', 'Play'),
      r'^(menu|settings)$': ('MENU', 'Menu'),
      r'^(input|source)$': ('INPUT', 'Input'),
      r'^(off|power off|turn off)$': ('POWER_OFF', 'Off'),
      r'^(on|power on|turn on)$': ('POWER_ON', 'On'),
      r'^(fast forward|forward|skip)$': ('FWD', 'Forward'),
      r'^(rewind)$': ('REW', 'Rewind'),
    };
    for (final e in simple.entries) {
      if (RegExp(e.key).hasMatch(t)) return keys([e.value.$1], e.value.$2);
    }
    // "up up up", "down 3", "right twice"
    final rep = RegExp(r'^(up|down|left|right)(?: (\1)(?: \1)*| (two|2|twice|three|3|four|4|five|5)(?: times)?)$').firstMatch(t);
    if (rep != null) {
      final k = rep.group(1)!;
      final words = t.split(' ').where((w) => w == k).length;
      final n = switch (rep.group(3)) {
        'two' || '2' || 'twice' => 2,
        'three' || '3' => 3,
        'four' || '4' => 4,
        'five' || '5' => 5,
        _ => words,
      };
      return keys(List.filled(n, k.toUpperCase()), '${k[0].toUpperCase()}${k.substring(1)} x$n');
    }
    return tv(said);
  }
}
