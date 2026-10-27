import 'package:even_suite/core/voice/voice_commands.dart';
import 'package:even_suite/core/voice/voice_controller.dart';
import 'package:even_suite/core/voice/wake_word.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

void main() {
  group('wake word', () {
    final wake = WakeWord();

    test('Claude and its sound-alikes', () {
      for (final s in ['Claude what time is it', 'clod what time is it', 'Claud, what time is it',
          'Klaud what time is it', 'clawed what time is it', 'hey cloud what time is it',
          'Cloud what time is it', "Claude's what time is it"]) {
        final m = wake.find(s);
        expect(m, isNotNull, reason: s);
        expect(m!.rest, 'what time is it', reason: s);
      }
    });

    test('repeating it ("claude... claude... clod") still works', () {
      final m = wake.find('Claude... Claude... clod open the notes');
      expect(m!.rest, 'open the notes');
    });

    test('everyday "cloud" mid-sentence does not trigger', () {
      expect(wake.find('I saved it to the cloud yesterday'), isNull);
      expect(wake.find('the weather is cloudy'), isNull);
      expect(wake.find('nothing to see here'), isNull);
    });

    test('strictness: mid-sentence mentions only count when relaxed', () {
      const talk = 'I asked Claude to fix the build';
      expect(WakeWord(null, WakeStrictness.relaxed).find(talk)!.rest, 'to fix the build');
      expect(WakeWord(null, WakeStrictness.normal).find(talk), isNull);
      expect(WakeWord(null, WakeStrictness.normal).find('hey Claude what time is it')!.rest, 'what time is it');
      final strict = WakeWord(null, WakeStrictness.strict);
      expect(strict.find('Claude what time is it'), isNull, reason: 'needs "hey"');
      expect(strict.find('OK Claude what time is it')!.rest, 'what time is it');
      expect(strict.find('so hey Claude what'), isNull, reason: 'only at the start');
    });

    test('custom word list', () {
      final w = WakeWord(['jarvis']);
      expect(w.find('Jarvis lights on')!.rest, 'lights on');
      expect(w.find('Claude lights on'), isNull);
    });
  });

  group('commands', () {
    test('whole-utterance commands, polite words ignored', () {
      expect(VoiceCommands.parse('scroll down'), VoiceCommand.scrollDown);
      expect(VoiceCommands.parse('Page up.'), VoiceCommand.scrollUp);
      expect(VoiceCommands.parse('please go to the top'), VoiceCommand.top);
      expect(VoiceCommands.parse('Stop listening'), VoiceCommand.stopListening);
      expect(VoiceCommands.parse('send it'), VoiceCommand.send);
      expect(VoiceCommands.parse('never mind'), VoiceCommand.cancel);
      expect(VoiceCommands.parse('press enter'), VoiceCommand.enter);
      expect(VoiceCommands.parse('what is the weather'), isNull);
    });

    test('"send" at the end of a message', () {
      expect(VoiceCommands.splitEnding('what time is it send').command, VoiceCommand.send);
      expect(VoiceCommands.splitEnding('what time is it send').body, 'what time is it');
      expect(VoiceCommands.splitEnding('write a haiku about rain and send it').body, 'write a haiku about rain');
      expect(VoiceCommands.splitEnding('Summarize my notes. Send.').body, 'Summarize my notes.');
      expect(VoiceCommands.splitEnding('send').body, '');
    });

    test('"send" that is part of the sentence is not a command', () {
      expect(VoiceCommands.splitEnding('what should I send').command, isNull);
      expect(VoiceCommands.splitEnding('how do I send it').command, isNull);
      expect(VoiceCommands.splitEnding('remind me to send').command, isNull);
    });

    test('the end of a back-and-forth', () {
      for (final t in ['Thanks!', 'thank you', "that's all", 'no thanks', 'OK.', 'got it']) {
        expect(VoiceCommands.endsConversation(t), isTrue, reason: t);
      }
      for (final t in ['thanks for the tip, what about tomorrow', 'no I meant the other one']) {
        expect(VoiceCommands.endsConversation(t), isFalse, reason: t);
      }
    });

    test('cancel phrases; a lone "cancel" inside a message is kept', () {
      expect(VoiceCommands.splitEnding('blah blah never mind').command, VoiceCommand.cancel);
      expect(VoiceCommands.splitEnding('cancel').command, VoiceCommand.cancel);
      expect(VoiceCommands.splitEnding('please cancel').command, isNull);
    });
  });

  group('voice controller', () {
    late FakeSpeech speech;
    late VoiceController voice;
    late List<String> sent;
    late List<VoiceCommand> commands;
    late List<String> drafts;

    setUp(() {
      speech = FakeSpeech();
      sent = [];
      commands = [];
      drafts = [];
      voice = VoiceController(
        speech,
        onSubmit: (t) async => sent.add(t),
        onCommand: commands.add,
        onDraft: drafts.add,
      );
    });

    tearDown(() => voice.dispose());

    Future<void> flush() => Future<void>.delayed(Duration.zero);

    test('hands-free: wake word, message, "send" in separate breaths', () async {
      await voice.setHandsFree(true);
      expect(voice.mode, VoiceMode.waiting);
      expect(speech.running, isTrue);

      speech.partial('what is');
      await flush();
      expect(voice.mode, VoiceMode.waiting, reason: 'no wake word yet');

      speech.partial('Claude');
      await flush();
      expect(voice.mode, VoiceMode.dictating, reason: 'switches as soon as it hears "Claude"');
      speech.say('Claude');
      await flush();
      expect(voice.mode, VoiceMode.dictating);

      speech.partial('what is on');
      speech.say('what is on my calendar');
      await flush();
      expect(voice.draft, 'what is on my calendar');
      speech.say('send');
      await flush();
      expect(sent, ['what is on my calendar']);
      expect(voice.mode, VoiceMode.waiting, reason: 'hands-free stays on');
      expect(voice.draft, '');
    });

    test('hands-free: everything in one breath', () async {
      await voice.setHandsFree(true);
      speech.partial('clod tell me');
      speech.say('clod tell me a joke send');
      await flush();
      expect(sent, ['tell me a joke']);
    });

    test('"Claude, scroll down" is a command, not a message', () async {
      await voice.setHandsFree(true);
      speech.partial('Claude scroll');
      speech.say('Claude scroll down');
      await flush();
      expect(commands, [VoiceCommand.scrollDown]);
      expect(sent, isEmpty);
      expect(voice.mode, VoiceMode.waiting);
    });

    test('cancel drops the message', () async {
      await voice.setHandsFree(true);
      speech.say('Claude delete all my files');
      await flush();
      expect(voice.dictating, isTrue);
      speech.say('never mind');
      await flush();
      expect(sent, isEmpty);
      expect(voice.draft, '');
      expect(voice.mode, VoiceMode.waiting);
    });

    test('"Claude, stop listening" turns hands-free off', () async {
      await voice.setHandsFree(true);
      speech.say('Claude stop listening');
      await flush();
      expect(voice.handsFree, isFalse);
      expect(voice.mode, VoiceMode.off);
      expect(speech.running, isFalse);
    });

    test('talk button: dictate, say send, mic stops', () async {
      await voice.startDictation();
      expect(voice.dictating, isTrue);
      expect(speech.running, isTrue);
      speech.say('open the garage');
      speech.say('send');
      await flush();
      expect(sent, ['open the garage']);
      expect(voice.mode, VoiceMode.off);
      expect(speech.running, isFalse, reason: 'not hands-free: the mic goes off after sending');
    });

    test('talk button: stop keeps the text for the Send button', () async {
      voice.autoSend = false;
      await voice.startDictation();
      speech.say('draft this for me');
      await flush();
      await voice.stopDictation();
      expect(voice.draft, 'draft this for me');
      expect(voice.mode, VoiceMode.off);
      await voice.send();
      expect(sent, ['draft this for me']);
    });

    test('talk button: it sends when you pause (no "send" needed)', () async {
      await voice.startDictation();
      speech.partial('what time');
      await flush();
      expect(sent, isEmpty);
      speech.say('what time is it in Tokyo');
      await flush();
      expect(sent, ['what time is it in Tokyo']);
      expect(voice.lastSpoken, isTrue);
    });

    test('typed text is not "spoken" (no follow-up for it)', () async {
      voice.setDraft('typed question');
      await voice.send('typed question');
      expect(sent, ['typed question']);
      expect(voice.lastSpoken, isFalse);
    });

    test('hands-free still needs "send" by default (nothing goes by accident)', () async {
      await voice.setHandsFree(true);
      speech.say('Claude remind me to call mom');
      await flush();
      expect(sent, isEmpty);
      expect(voice.draft, 'remind me to call mom');
      await voice.cancel();
      voice.wakeAutoSend = true;
      speech.say('Claude what is the weather');
      await flush();
      expect(sent, ['what is the weather']);
    });

    testWidgets('follow-up: listens right after, no wake word; silence ends it', (tester) async {
      var ended = 0;
      final v = VoiceController(speech, onSubmit: (t) async => sent.add(t), onCommand: commands.add)
        ..onFollowUpEnd = () => ended++;
      await v.startFollowUp(const Duration(seconds: 8));
      expect(v.followUp, isTrue);
      expect(v.dictating, isTrue);
      expect(speech.running, isTrue);
      await tester.pump(const Duration(seconds: 9));
      expect(v.followUp, isFalse);
      expect(ended, 1);
      expect(speech.running, isFalse);

      // A follow-up question is sent when you pause.
      await v.startFollowUp(const Duration(seconds: 8));
      speech.partial('and tomorrow');
      await tester.pump(const Duration(seconds: 9));
      expect(v.followUp, isTrue, reason: 'talking cancels the window');
      speech.say('and tomorrow');
      await tester.pump();
      expect(sent, ['and tomorrow']);
      expect(v.followUp, isFalse);

      // "Thanks" ends it without sending.
      await v.startFollowUp(const Duration(seconds: 8));
      speech.say('thanks');
      await tester.pump();
      expect(sent, ['and tomorrow']);
      expect(ended, 2);

      // A command works without the wake word.
      await v.startFollowUp(const Duration(seconds: 8));
      speech.say('scroll down');
      await tester.pump();
      expect(commands, [VoiceCommand.scrollDown]);
      await tester.pump(const Duration(seconds: 7));
      expect(v.followUp, isFalse);
      v.dispose();
    });

    test('suspend keeps the mic off while the phone talks', () async {
      await voice.setHandsFree(true);
      expect(speech.running, isTrue);
      await voice.suspend();
      expect(speech.running, isFalse);
      await voice.resume();
      expect(speech.running, isTrue);
    });

    test('glasses hold-to-talk: release sends what was said', () async {
      await voice.startDictation(hold: true);
      speech.partial('turn on');
      speech.say('turn on the lights', finishing: true);
      await flush();
      expect(sent, isEmpty, reason: 'still holding');
      speech.partial('in the kitchen');
      await flush();
      voice.releaseHold();
      expect(speech.finishes, 1);
      speech.say('in the kitchen', finishing: true);
      await flush();
      expect(sent, ['turn on the lights in the kitchen']);
    });

    test('TouchBar hold during hands-free uses the glasses mic, then goes back to the phone', () async {
      await voice.setHandsFree(true);
      expect(speech.lastSource, 'phone');
      await voice.startDictation(hold: true, source: 'glasses');
      expect(speech.lastSource, 'glasses');
      expect(speech.stops, 1, reason: 'the phone session was swapped out');
      voice.releaseHold();
      speech.say('turn off the lights', finishing: true);
      await flush();
      expect(sent, ['turn off the lights']);
      expect(voice.mode, VoiceMode.waiting);
      expect(speech.lastSource, 'phone', reason: 'hands-free listens with the phone again');
      expect(speech.running, isTrue);
    });

    test('teleprompter commands', () {
      expect(VoiceCommands.parse('pause'), VoiceCommand.pause);
      expect(VoiceCommands.parse('keep going'), VoiceCommand.resume);
      expect(VoiceCommands.parse('faster'), VoiceCommand.faster);
      expect(VoiceCommands.parse('slow down'), VoiceCommand.slower);
    });

    test('a fatal error turns everything off', () async {
      await voice.setHandsFree(true);
      speech.error('Microphone permission is off', fatal: true);
      await flush();
      expect(voice.handsFree, isFalse);
      expect(voice.mode, VoiceMode.off);
      expect(voice.error, contains('Microphone'));
    });

    testWidgets('hands-free switches itself off after the auto-off time', (tester) async {
      voice.autoOff = const Duration(minutes: 2);
      var autoOff = false;
      final v = VoiceController(speech, onSubmit: (_) async {}, onCommand: (_) {}, onAutoOff: () => autoOff = true)
        ..autoOff = const Duration(minutes: 2);
      await v.setHandsFree(true);
      expect(v.autoOffAt, isNotNull);
      await tester.pump(const Duration(minutes: 1));
      speech.say('Claude what time is it');
      await tester.pump();
      // Using it pushes the deadline back.
      await tester.pump(const Duration(seconds: 90));
      expect(v.handsFree, isTrue);
      await tester.pump(const Duration(seconds: 40));
      expect(v.handsFree, isFalse);
      expect(autoOff, isTrue);
      v.dispose();
    });

    testWidgets('a paused message goes back to waiting but keeps its text', (tester) async {
      final v = VoiceController(speech, onSubmit: (t) async => sent.add(t), onCommand: (_) {})
        ..dictationIdle = const Duration(seconds: 30);
      await v.setHandsFree(true);
      speech.say('Claude remind me to buy milk');
      await tester.pump();
      expect(v.dictating, isTrue);
      await tester.pump(const Duration(seconds: 31));
      expect(v.mode, VoiceMode.waiting);
      expect(v.draft, 'remind me to buy milk');
      // "Claude, send" sends the paused message.
      speech.say('Claude send');
      await tester.pump();
      expect(sent, ['remind me to buy milk']);
      await v.setHandsFree(false);
      v.dispose();
    });
  });
}
