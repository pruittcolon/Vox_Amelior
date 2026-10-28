import 'dart:convert';

import 'package:even_suite/core/home/home_commands.dart';
import 'package:even_suite/core/home/vizio.dart';
import 'package:even_suite/core/home/voice_monkey.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final lights = VoiceMonkey.parse('All lights = alllights-di8jv: all lights, lights, all the lights\n'
      '# Bedroom light = YOUR-DEVICE-ID: bedroom light\n'
      'Kitchen lights = kitchen-x1: kitchen lights, kitchen light, chicken light');

  group('lights', () {
    test('devices: one per line, # lines skipped', () {
      expect(lights.map((d) => d.device), ['alllights-di8jv', 'kitchen-x1']);
      expect(VoiceMonkey.parse(VoiceMonkey.defaultDevices).single.device, 'alllights-di8jv');
      expect(VoiceMonkey.parse('Porch = porch1').single.phrases, ['Porch'], reason: 'no phrases: its name');
    });

    test('what you say: on, off, turn, the - all the same device', () {
      for (final said in [
        'All lights off.',
        'all lights on',
        'Turn off the lights',
        'turn the lights on please',
        'Lights.',
        'switch off all the lights',
      ]) {
        expect(HomeCommands.light(said, lights)?.device, 'alllights-di8jv', reason: said);
      }
      expect(HomeCommands.light('chicken light on', lights)?.label, 'Kitchen lights', reason: 'recognizer mistake');
      expect(HomeCommands.light('what are the lights like in paris', lights), isNull);
      expect(HomeCommands.light('on', lights), isNull);
    });

    test('Voice Monkey call: token, or the whole trigger URL', () {
      expect(VoiceMonkey.url('abc123', 'alllights-di8jv').toString(),
          'https://api-v2.voicemonkey.io/trigger?token=abc123&device=alllights-di8jv');
      expect(VoiceMonkey.url('https://api-v2.voicemonkey.io/trigger?token=abc123', 'alllights-di8jv').queryParameters,
          {'token': 'abc123', 'device': 'alllights-di8jv'});
    });

    test('trigger: done, refused, no token', () async {
      Uri? asked;
      final ok = VoiceMonkey(get: (u) async {
        asked = u;
        return (200, '{"success":true}');
      });
      expect(await ok.trigger('abc', lights.first), 'All lights ✓');
      expect(asked!.queryParameters['device'], 'alllights-di8jv');
      final no = VoiceMonkey(get: (u) async => (200, '{"success":false,"message":"Invalid token"}'));
      expect(await no.trigger('bad', lights.first), contains('Invalid token'));
      expect(await ok.trigger('', lights.first), contains('Add your Voice Monkey token'));
    });
  });

  group('TV by voice', () {
    List<String>? keys(String said) => HomeCommands.tv(said)?.keys;

    test('power, volume (3 presses, like the Roku app), mute, input', () {
      expect(keys('TV off'), ['POWER_OFF']);
      expect(keys('Television off.'), ['POWER_OFF']);
      expect(keys('T.V. off'), ['POWER_OFF']);
      expect(keys('television on'), ['POWER_ON']);
      expect(keys('Turn off the TV.'), ['POWER_OFF']);
      expect(keys('turn the tv on'), ['POWER_ON']);
      expect(keys('TV volume up'), ['VOL_UP', 'VOL_UP', 'VOL_UP']);
      expect(keys('turn down the tv'), ['VOL_DOWN', 'VOL_DOWN', 'VOL_DOWN']);
      expect(keys('mute the TV'), ['MUTE']);
      expect(keys('TV home'), ['HOME']);
      final hdmi = HomeCommands.tv('Vizio HDMI 2')!;
      expect((hdmi.kind, hdmi.input), (HomeKind.tvInput, 'hdmi2'));
      expect(HomeCommands.tv('what is on tv tonight'), isNull);
    });

    test('"TV remote" starts the remote; a bare "TV" only with the wake word', () {
      expect(HomeCommands.tv('TV remote')!.kind, HomeKind.tvRemote);
      expect(HomeCommands.tv('TV')!.kind, HomeKind.tvRemote);
      expect(HomeCommands.tv('TV', trigger: true), isNull, reason: 'heard in passing, no wake word');
    });

    test('remote mode: just the key', () {
      List<String>? r(String said) => HomeCommands.remote(said)?.keys;
      expect(r('Up.'), ['UP']);
      expect(r('okay'), ['OK']);
      expect(r('go back'), ['BACK']);
      expect(r('volume down'), ['VOL_DOWN', 'VOL_DOWN', 'VOL_DOWN']);
      expect(r('down 3'), ['DOWN', 'DOWN', 'DOWN']);
      expect(r('right right'), ['RIGHT', 'RIGHT']);
      expect(r('pause'), ['PAUSE']);
      expect(r('TV off'), ['POWER_OFF']);
      expect(HomeCommands.remote('done')!.kind, HomeKind.tvDone);
      expect(HomeCommands.remote("that's all")!.kind, HomeKind.tvDone);
      expect(HomeCommands.remote('tell me a joke'), isNull);
    });
  });

  group('Vizio SmartCast', () {
    final calls = <(String, String, Map<String, String>, Object?)>[];
    Map<String, dynamic> answer(String path) => switch (path) {
          '/menu_native/dynamic/tv_settings/devices/current_input' => {
              'STATUS': {'RESULT': 'SUCCESS'},
              'ITEMS': [
                {'CNAME': 'current_input', 'HASHVAL': 1234, 'VALUE': 'CAST'}
              ]
            },
          '/pairing/start' => {
              'STATUS': {'RESULT': 'SUCCESS'},
              'ITEM': {'PAIRING_REQ_TOKEN': 555}
            },
          '/pairing/pair' => {
              'STATUS': {'RESULT': 'SUCCESS'},
              'ITEM': {'AUTH_TOKEN': 'Ztoken'}
            },
          '/state/device/deviceinfo' => {
              'STATUS': {'RESULT': 'SUCCESS'},
              'ITEMS': [
                {
                  'VALUE': {'CAST_NAME': 'Wu-Tang-TV'}
                }
              ]
            },
          _ => {
              'STATUS': {'RESULT': 'SUCCESS'}
            },
        };
    final tv = VizioTv(
      call: (method, url, headers, body) async {
        calls.add((method, url.toString(), headers, body));
        if (url.host != '192.168.4.21') throw Exception('not a TV');
        return answer(url.path);
      },
      discoverAddresses: () async => ['192.168.4.1', '192.168.4.21'],
    );

    setUp(calls.clear);

    test('keys: PUT /key_command/ with the token, one per key', () async {
      await tv.keys('192.168.4.21', 'Ztoken', ['VOL_UP', 'VOL_UP']);
      expect(calls, hasLength(2));
      final (method, url, headers, body) = calls.first;
      expect(method, 'PUT');
      expect(url, 'https://192.168.4.21:7345/key_command/');
      expect(headers['AUTH'], 'Ztoken');
      expect(jsonEncode(body), '{"KEYLIST":[{"CODESET":5,"CODE":1,"ACTION":"KEYPRESS"}]}');
    });

    test('input: read its HASHVAL, then change it', () async {
      await tv.setInput('192.168.4.21', 'Ztoken', 'hdmi1');
      expect(calls.map((c) => c.$1), ['GET', 'PUT']);
      expect(calls.last.$4, {'REQUEST': 'MODIFY', 'HASHVAL': 1234, 'VALUE': 'hdmi1'});
    });

    test('find it on the Wi-Fi, pair with the PIN', () async {
      expect(await tv.discover(), ('192.168.4.21', 'Wu-Tang-TV'));
      await tv.startPairing('192.168.4.21', 'even-suite-abc');
      expect(await tv.finishPairing('192.168.4.21', '1234'), 'Ztoken');
      final pair = calls.last.$4 as Map;
      expect(pair['PAIRING_REQ_TOKEN'], 555);
      expect(pair['RESPONSE_VALUE'], '1234');
      expect(pair['DEVICE_ID'], 'even-suite-abc');
    });

    test('a refusal says why', () async {
      final refusing = VizioTv(
          call: (m, u, h, b) async => {
                'STATUS': {'RESULT': 'REQUIRES_PAIRING'}
              });
      await expectLater(refusing.keys('1.2.3.4', '', ['OK']), throwsA(isA<VizioException>()));
    });
  });
}
