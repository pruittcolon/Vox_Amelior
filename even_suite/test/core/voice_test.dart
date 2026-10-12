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
      await voice.startDictation();
      speech.say('draft this for me');
      await flush();
      await voice.stopDictation();
      expect(voice.draft, 'draft this for me');
      expect(voice.mode, VoiceMode.off);
      await voice.send();
      expect(sent, ['draft this for me']);
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
