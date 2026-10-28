import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../log.dart';

/// A Voice Monkey device (it runs an Alexa routine) and what you say for it.
class LightDevice {
  const LightDevice(this.name, this.device, this.phrases);

  /// Shown on the glasses ("All lights").
  final String name;

  /// The Voice Monkey device id ("alllights").
  final String device;

  /// What you say, without "on"/"off"/"turn" ("all lights", "lights").
  final List<String> phrases;
}

/// Alexa routines through Voice Monkey (api-v2.voicemonkey.io), as the old
/// app did: a GET with your token and the device id. On and off are the same
/// call - the routine decides (a toggle), exactly like before.
class VoiceMonkey {
  VoiceMonkey({this.get});

  /// For tests: answers a GET with (status, body).
  final Future<(int, String)> Function(Uri url)? get;

  /// Your all-lights device, plus the old app's other rooms as examples to
  /// fill in (with the recognizer's usual mistakes: "chicken light",
  /// "leaving room"). Lines starting with # are skipped.
  static const defaultDevices = 'All lights = alllights-di8jv: all lights, lights, all the lights\n'
      '# Bedroom light = YOUR-DEVICE-ID: bedroom light, bedroom lights\n'
      '# Kitchen lights = YOUR-DEVICE-ID: kitchen lights, kitchen light, chicken light, chicken lights\n'
      '# Living room light = YOUR-DEVICE-ID: living room light, living room lights, leaving room light';

  /// One device per line: `Name = device: phrase, phrase`. A line without
  /// phrases uses its name.
  static List<LightDevice> parse(String text) {
    final out = <LightDevice>[];
    for (final raw in text.split('\n')) {
      final line = raw.trim();
      if (line.isEmpty || line.startsWith('#')) continue;
      final eq = line.indexOf('=');
      if (eq <= 0) continue;
      final name = line.substring(0, eq).trim();
      final rest = line.substring(eq + 1);
      final colon = rest.indexOf(':');
      final device = (colon < 0 ? rest : rest.substring(0, colon)).trim();
      if (name.isEmpty || device.isEmpty) continue;
      final phrases = colon < 0
          ? [name]
          : rest.substring(colon + 1).split(',').map((p) => p.trim()).where((p) => p.isNotEmpty).toList();
      out.add(LightDevice(name, device, phrases.isEmpty ? [name] : phrases));
    }
    return out;
  }

  /// The call for [device]: [token] is your Voice Monkey token, or the whole
  /// trigger URL from their dashboard (as the old app kept it).
  static Uri url(String token, String device) {
    var t = token.trim().replaceAll(RegExp(r'\s'), '');
    if (t.startsWith('http')) {
      final u = Uri.parse(t);
      // The old v1 links (api.voicemonkey.io, access_token + secret_token) name the device "monkey".
      final key = u.queryParameters.containsKey('access_token') ? 'monkey' : 'device';
      return u.replace(queryParameters: {...u.queryParameters, key: device});
    }
    t = t.replaceFirst(RegExp(r'^token='), '');
    return Uri.https('api-v2.voicemonkey.io', '/trigger', {'token': t, 'device': device});
  }

  /// Run [d]'s routine. Returns what to show ("All lights ✓" or why not).
  Future<String> trigger(String token, LightDevice d) async {
    if (token.trim().isEmpty) return 'Add your Voice Monkey token: More › Lights and TV';
    try {
      final (status, body) = await (get ?? _get)(url(token, d.device)).timeout(const Duration(seconds: 10));
      if (status != 200) {
        var why = '';
        try {
          final j = jsonDecode(body);
          if (j is Map) why = '${j['error'] ?? j['message'] ?? ''}';
        } catch (_) {}
        EventLog.add('home', 'voiceMonkey', {'device': d.device, 'status': status, 'error': why});
        if (why.contains('user could not be retrieved')) {
          return 'Voice Monkey does not know that token. Copy the API token from voicemonkey.io (Dashboard › API).';
        }
        if (why.toLowerCase().contains('device')) return 'Voice Monkey: $why - check the device id ${d.device}';
        return 'Voice Monkey said $status${why.isEmpty ? '' : ': $why'} (${d.name})';
      }
      // v2 answers {"success": true}; anything else that is 200 counts too.
      try {
        final j = jsonDecode(body);
        if (j is Map && j['success'] == false) return 'Voice Monkey: ${j['message'] ?? 'refused'} (${d.name})';
      } catch (_) {}
      return '${d.name} ✓';
    } on TimeoutException {
      return 'Voice Monkey did not answer (${d.name})';
    } catch (e) {
      return 'Could not reach Voice Monkey: $e';
    }
  }

  static Future<(int, String)> _get(Uri url) async {
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 8);
    try {
      final req = await client.getUrl(url);
      final res = await req.close();
      return (res.statusCode, await res.transform(utf8.decoder).join());
    } finally {
      client.close();
    }
  }
}
