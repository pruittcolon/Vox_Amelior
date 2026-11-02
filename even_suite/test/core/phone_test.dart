import 'package:even_suite/core/glasses/glasses_text.dart';
import 'package:even_suite/core/phone_actions.dart';
import 'package:even_suite/core/suite.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  final now = DateTime(2026, 10, 28, 14, 20);

  group('phone actions in Claude\'s reply', () {
    const reply = 'Reminder set for 9 AM.\n'
        '[[phone:{"do":"remind","at":"2026-10-29T09:00","text":"Call the bank"}]]';

    test('found, and taken out of what you see', () {
      final a = PhoneAction.extract(reply).single;
      expect(a.kind, 'remind');
      expect(a.args, {'at': '2026-10-29T09:00', 'text': 'Call the bank'});
      expect(PhoneAction.strip(reply), 'Reminder set for 9 AM.');
      expect(PhoneAction.strip('Opening it.\n[[phone:{"do":"op'), 'Opening it.', reason: 'half-written, still streaming');
      expect(PhoneAction.extract('[[phone:{not json}]]'), isEmpty);
      expect(PhoneAction.strip('Plain answer.'), 'Plain answer.');
    });

    test('Claude is told what it can do (only when allowed)', () {
      const size = ScreenSize(47, 5);
      expect(Suite.buildPrimer(size: size, phone: true), contains('[[phone:'));
      expect(Suite.buildPrimer(size: size), isNot(contains('[[phone:')));
    });
  });

  group('instant commands', () {
    PhoneAction? p(String s) => PhoneAction.parseLocal(s, now);

    test('flashlight, music, volume', () {
      expect(p('Flashlight on')?.args, {'on': true});
      expect(p('turn off the flashlight')?.args, {'on': false});
      expect(p('next song')?.args, {'key': 'next'});
      expect(p('Pause the music.')?.args, {'key': 'pause'});
      expect(p('volume up')?.kind, 'volume');
    });

    test('timers and "remind me in ..."', () {
      final t = p('Set a timer for 5 minutes')!;
      expect(t.kind, 'remind');
      expect(t.args['at'], now.add(const Duration(minutes: 5)).millisecondsSinceEpoch);
      expect(t.args['text'], 'Timer done (5 minutes)');
      expect(p('ten minute timer')!.args['at'], now.add(const Duration(minutes: 10)).millisecondsSinceEpoch);
      final r = p('Remind me in 20 minutes to check the oven')!;
      expect(r.args['text'], 'Check the oven');
      expect(r.args['at'], now.add(const Duration(minutes: 20)).millisecondsSinceEpoch);
    });

    test('everything else goes to Claude', () {
      expect(p('what is the weather'), isNull);
      expect(p('remind me tomorrow at 9 to call the bank'), isNull);
      expect(p('turn on the lights in the kitchen'), isNull);
    });
  });

  group('running them', () {
    test('a reminder: in seconds, or a clock time (already past means tomorrow)', () {
      expect(PhoneActions.whenFor({'in': 600}, now), now.add(const Duration(minutes: 10)));
      expect(PhoneActions.whenFor({'at': '2026-10-28T15:00'}, now), DateTime(2026, 10, 28, 15));
      expect(PhoneActions.whenFor({'at': '2026-10-28T09:00'}, now), DateTime(2026, 10, 29, 9));
      expect(PhoneActions.whenFor({}, now), isNull);
    });

    test('sent to Android with an id, and said back briefly', () async {
      final system = FakeSystem();
      final actions = PhoneActions(system);
      final said = await actions.run(const PhoneAction('remind', {'at': '2026-10-28T15:00', 'text': 'Call'}), now: now);
      expect(said, 'Reminder set · 3:00 PM');
      final sent = system.phoneCalls.single;
      expect(sent['do'], 'remind');
      expect(sent['at'], DateTime(2026, 10, 28, 15).millisecondsSinceEpoch);
      expect(sent['id'], isNotNull);
      system.phoneAnswer = const {'ok': false, 'error': 'No app called Zork'};
      expect(await actions.run(const PhoneAction('open', {'app': 'Zork'})), 'Could not: No app called Zork');
      system.phoneAnswer = const {'ok': true, 'say': 'Opening Spotify', 'note': 'background'};
      expect(await actions.run(const PhoneAction('open', {'app': 'Spotify'})), contains('Display over other apps'));
    });
  });
}
