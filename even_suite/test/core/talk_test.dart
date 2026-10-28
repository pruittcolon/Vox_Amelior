import 'dart:typed_data';

import 'package:even_suite/core/ble/ble_manager.dart';
import 'package:even_suite/core/ble/proto.dart';
import 'package:even_suite/core/bridge/bridge_client.dart';
import 'package:even_suite/core/bridge/chat_controller.dart';
import 'package:even_suite/core/bridge/work_controller.dart';
import 'package:even_suite/core/glasses/claude_panel.dart';
import 'package:even_suite/core/glasses/dashboard.dart';
import 'package:even_suite/core/glasses/glasses_display.dart';
import 'package:even_suite/core/glasses/glasses_text.dart';
import 'package:even_suite/core/suite.dart';
import 'package:even_suite/core/talk/conversation.dart';
import 'package:even_suite/core/voice/speaker.dart';
import 'package:even_suite/core/voice/voice_commands.dart';
import 'package:even_suite/core/voice/wake_word.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

BridgeMessage claude(String id,
        {String status = 'pending', String stage = 'thinking', String tool = '', String live = '', int block = 0,
        String text = ''}) =>
    BridgeMessage(
        id: id,
        who: 'claude',
        text: text,
        status: status,
        stage: stage,
        tool: tool,
        live: live,
        block: block,
        ts: DateTime.now().millisecondsSinceEpoch / 1000);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('speech text', () {
    test('clean: no Markdown, links or code read out', () {
      expect(SpeechText.clean('**Done.** See https://x.io/a and `ls -la`.\n- one\n- two'),
          'Done. See a link and ls -la. one two');
      expect(SpeechText.clean('```\ncode\n```\nThat is it'), '(code on screen) That is it');
    });

    test('whole sentences only while streaming; short ones grouped', () {
      const t = 'Sunny and 22 degrees. Light wind from the west, about 10 km/h. Rain later';
      final (pieces, next) = SpeechText.sentences(t, 0);
      expect(pieces, ['Sunny and 22 degrees. Light wind from the west, about 10 km/h.']);
      expect(t.substring(next), 'Rain later');
      final (rest, end) = SpeechText.sentences(t, next, flush: true);
      expect(rest, ['Rain later']);
      expect(end, t.length);
    });

    test('a number still being written is not cut ("3." -> "3.5")', () {
      final (pieces, next) = SpeechText.sentences('It costs 3.', 0, minChars: 1);
      expect(pieces, isEmpty);
      expect(next, 0);
    });
  });

  group('narrator', () {
    test('speaks sentences as they stream, the rest at the end, then done', () async {
      final sp = FakeSpeaker();
      final n = Narrator(sp);
      var done = 0;
      n.onDone = () => done++;
      n.feed('m1', 1, 'The build passed on the first try, which is nice.');
      n.feed('m1', 1, 'The build passed on the first try, which is nice. All 48 tests ran');
      expect(sp.said, ['The build passed on the first try, which is nice.']);
      n.finish('m1', 1, 'The build passed on the first try, which is nice. All 48 tests ran green.');
      expect(sp.said.last, 'All 48 tests ran green.');
      expect(sp.said.length, 2, reason: 'nothing said twice');
      await Future<void>.delayed(Duration.zero);
      expect(done, 1);
      expect(n.speaking, isFalse);
      n.dispose();
    });

    test('an answer that never streamed is said whole', () async {
      final sp = FakeSpeaker();
      final n = Narrator(sp);
      n.finish('m2', 0, 'Yes. Tomorrow at nine.');
      expect(sp.said.join(' '), 'Yes. Tomorrow at nine.');
      n.dispose();
    });

    test('off: nothing is said, done right away', () {
      final sp = FakeSpeaker();
      final n = Narrator(sp)..enabled = false;
      var done = 0;
      n.onDone = () => done++;
      n.feed('m', 1, 'Hello there, this is a sentence.');
      n.finish('m', 1, 'Hello there, this is a sentence.');
      expect(sp.said, isEmpty);
      expect(done, 1);
      n.dispose();
    });
  });

  group('conversation screens', () {
    test('listening shows a voice meter and the last lines said', () {
      final s = Conversation.listeningScreen('what is the weather', 0.5, 1);
      final lines = s.split('\n');
      expect(lines, ['Termux-Claude listening..', 'what is the weather'], reason: 'plain text, no meter');
      expect([for (var f = 0; f < 5; f++) Conversation.listeningScreen('', 0, f).split('\n').first],
          ['Termux-Claude listening...', 'Termux-Claude listening..', 'Termux-Claude listening.',
           'Termux-Claude listening..', 'Termux-Claude listening...']);
      expect(Conversation.listeningScreen('', 0, 0), contains('Say your question'));
    });

    test('terminal mode: "start" opens it, "send" types it, both show on screen', () {
      final wake = WakeWord(const ['start', 'tart', 'starts']);
      final said = Suite.terminalWords('Start, list the files send', wake);
      expect(said.started, isTrue);
      expect(said.text, 'list the files');
      expect(Suite.terminalWords('git status', wake), (started: false, text: 'git status'));
      final screen = Conversation.listeningScreen('list the files', 0, 0,
          header: 'Terminal [start]', footer: 'say "send" to type it');
      expect(screen.split('\n'), ['Terminal [start]...', 'list the files', 'say "send" to type it']);
      expect(Suite.sameAsk('list the files', 'List the files.'), isTrue);
      expect(Suite.sameAsk('list the files', 'fix the build'), isFalse);
      expect(VoiceCommands.parse('terminal'), VoiceCommand.toTerminal);
      expect(VoiceCommands.parse('talk to the terminal'), VoiceCommand.toTerminal);
      expect(VoiceCommands.parse('back to chat'), VoiceCommand.toChat);
    });

    test('a long question is cut to two lines with an ellipsis', () {
      final q = List.filled(40, 'word').join(' ');
      final lines = Conversation.quoteLines(q);
      expect(lines.length, 2);
      expect(lines.last, endsWith('..."'));
      expect(Conversation.quoteLines('short one'), ['"short one"']);
    });

    test('primer: brief by default, read aloud when spoken', () {
      const size = ScreenSize(47, 5);
      final brief = Suite.buildPrimer(size: size);
      expect(brief, contains('about 50 characters'));
      expect(brief, contains('100-150'));
      expect(brief, contains('235'));
      expect(brief, contains('G1'));
      expect(brief, isNot(contains('read aloud')));
      expect(Suite.buildPrimer(size: size, length: 'normal', spoken: true), contains('read aloud'));
      expect(Suite.buildPrimer(size: size, length: 'normal'), contains('about 150 characters'));
      expect(Suite.buildPrimer(size: size, quick: true), contains('[[full]]'));
      expect(Suite.buildPrimer(size: size), isNot(contains('[[full]]')));
      expect(Suite.wantsFull('Think harder about this'), isTrue);
      expect(Suite.wantsFull('what time is it'), isFalse);
    });
  });

  group('glasses flow', () {
    late FakeGlasses glasses;
    late FakeControl control;
    late BleManager ble;
    late GlassesDisplay display;
    late Conversation talk;

    setUp(() async {
      glasses = FakeGlasses();
      control = FakeControl();
      ble = BleManager(glasses, control: control);
      display = GlassesDisplay(Proto(ble), liveGap: Duration.zero);
      talk = Conversation(display);
      await control.connectNow();
    });

    tearDown(() {
      talk.dispose();
      display.dispose();
      ble.dispose();
    });

    Future<void> settle() async {
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(Duration.zero);
      }
    }

    test('got it -> working with the tool step -> streamed answer -> done', () async {
      talk.asked('what is the weather in Boston');
      await settle();
      expect(glasses.screens().last, contains('Got it'));
      expect(glasses.screens().last, contains('"what is the weather'));
      expect(glasses.screens().last, contains('Boston"'));
      expect(talk.stage, TalkStage.sent);

      talk.progress(claude('a', stage: 'tool', tool: 'Searching the web: boston'));
      await settle();
      expect(talk.step, 'Searching the web: boston');
      expect(talk.stage, TalkStage.working);

      talk.progress(claude('a', stage: 'writing', live: 'Sunny and', block: 1));
      await settle();
      expect(talk.stage, TalkStage.answering);
      expect(display.mode, DisplayMode.reply);
      expect(glasses.screens().last, contains('Sunny and'));

      talk.answered(claude('a', status: 'done', stage: 'done', text: 'Sunny and 22 degrees.', block: 1));
      await settle();
      expect(glasses.screens().last, contains('Sunny and 22 degrees.'));
      expect(talk.stage, TalkStage.idle);
    });

    test('double tap while Claude works: its screens stay away until the answer', () async {
      talk.asked('build it');
      await settle();
      display.onGlassesExit();
      talk.dismiss();
      expect(talk.quiet, isTrue);
      final shown = glasses.screens().length;
      talk.progress(claude('q', stage: 'tool', tool: 'Editing a.dart'));
      await settle();
      expect(glasses.screens().length, shown, reason: 'not put back up');
      expect(talk.step, 'Editing a.dart', reason: 'still followed (phone, look-up screen)');
      final shownOnGlasses = talk.answered(claude('q', status: 'done', stage: 'done', text: 'Built.'));
      expect(shownOnGlasses, isFalse);
      expect(talk.stage, TalkStage.idle);
      expect(talk.quiet, isFalse);
    });

    test('an error says so and offers a retry', () async {
      talk.asked('hi');
      talk.answered(claude('b', status: 'error', stage: 'error', text: 'Claude Code stopped without answering.'));
      await settle();
      expect(talk.stage, TalkStage.error);
      expect(glasses.screens().last, contains('Tap right to try again'));
    });

    test('a growing reply keeps your page and the teleprompter going', () async {
      final long = List.generate(12, (i) => 'Sentence number $i is here.').join(' ');
      display.updateReply(long.substring(0, 60));
      display.updateReply(long);
      expect(display.pager!.pageCount, greaterThan(1));
      display.down();
      final page = display.pager!.page;
      display.updateReply('$long And more.', done: true);
      expect(display.pager!.page, page);
    });

    test('the working screen follows your layout', () {
      final rows = Conversation.workingRows(
          layout: PanelLayout.parse('claude,asked:2,meta:0', base: PanelLayout.working),
          status: 'Claude · 3s ...',
          question: 'build it',
          step: 'Editing a.dart',
          steps: const ['Reading a.dart', 'Editing a.dart']);
      expect([for (final r in rows) r.text], ['Editing a.dart', 'Reading a.dart', '"build it"']);
      expect(rows.last.dim, isTrue);
    });

    test('Claude\'s screen closes by itself, but not while someone is talking', () async {
      display.closeAfter = const Duration(milliseconds: 60);
      var talking = true;
      display.holdOpen = () => talking;
      display.showStatus('Claude did not answer');
      await Future<void>.delayed(const Duration(milliseconds: 150));
      expect(display.mode, DisplayMode.status, reason: 'held open');
      talking = false;
      await Future<void>.delayed(const Duration(milliseconds: 150));
      expect(display.mode, DisplayMode.idle, reason: 'closed by itself');
      expect(glasses.sent['L']!.last[0], 0x18, reason: 'exit: back to the dashboard');
    });

    test('work timer: elapsed time on the working screen', () {
      talk.asked('build it');
      talk.progress(claude('c', stage: 'thinking'));
      final later = DateTime.now().add(const Duration(seconds: 5));
      final lines = talk.workingScreen(now: later).split('\n');
      expect(lines.first, startsWith('Claude · 5s'));
      expect(lines[1], '"build it"');
      expect(lines.last, 'Thinking');
      talk.progress(claude('c', stage: 'tool', tool: 'Searching the web: ${'very ' * 20}long query'));
      final cut = talk.workingScreen(now: later).split('\n').last;
      expect(cut, endsWith('...'), reason: 'a long step is cut to one line');
      expect(GlassesText.wrap(cut).length, 1);
    });
  });

  group('chat progress', () {
    test('each step of the answer is reported, then the reply', () {
      final chat = ChatController(FakeBridgeClient());
      final steps = <String>[];
      final replies = <String>[];
      chat.onProgress = (m) => steps.add('${m.stage}:${m.tool}${m.live}');
      chat.onReply = (m) => replies.add(m.text);
      chat.apply(const BridgeSnapshot(1, [], []));
      // Pretend a question went out.
      chat.messages = const [];
      chat.send('x').ignore();
      chat.apply(BridgeSnapshot(2, [claude('r', stage: 'thinking')], const ['claude']));
      chat.apply(BridgeSnapshot(3, [claude('r', stage: 'tool', tool: 'Reading a.dart')], const ['claude']));
      chat.apply(BridgeSnapshot(3, [claude('r', stage: 'tool', tool: 'Reading a.dart')], const ['claude']));
      chat.apply(BridgeSnapshot(4, [claude('r', stage: 'writing', live: 'Fixed', block: 1)], const ['claude']));
      chat.apply(BridgeSnapshot(5, [claude('r', status: 'done', stage: 'done', text: 'Fixed it.')], const []));
      expect(steps, ['thinking:', 'tool:Reading a.dart', 'writing:Fixed']);
      expect(replies, ['Fixed it.']);
      expect(chat.active, isNull);
    });
  });

  group('work', () {
    test('only new jobs that took long enough are announced', () {
      final work = WorkController(FakeBridgeClient())..minDuration = const Duration(seconds: 30);
      final told = <String>[];
      work.onDone = (i) => told.add(i.summary);
      WorkItem job(String id, int ms) => WorkItem(id: id, ts: 1, summary: 'job $id', ms: ms);
      work.apply(WorkSnapshot(1, [job('old', 90000)]));
      expect(told, isEmpty, reason: 'what was there before the app looked is not news');
      work.apply(WorkSnapshot(2, [job('old', 90000), job('quick', 4000), job('long', 75000)]));
      expect(told, ['job long']);
      expect(work.unread, 1);
      work.markRead();
      expect(work.unread, 0);

      final live = <int>[];
      work.onActive = (jobs) => live.add(jobs.length);
      const liveJob = WorkJob(session: 's1', project: 'even-suite', prompt: 'fix the build', step: 'Thinking');
      work.apply(WorkSnapshot(3, work.items, const [liveJob]));
      work.apply(WorkSnapshot(4, work.items, const [liveJob]));
      work.apply(WorkSnapshot(5, work.items, const [
        WorkJob(session: 's1', project: 'even-suite', prompt: 'fix the build', step: 'Editing a.dart', steps: 1),
      ]));
      work.apply(WorkSnapshot(6, work.items));
      expect(live, [1, 1, 0], reason: 'started, took a step, finished (no repeat for no change)');

      // `glasses "..."` from a Claude Code session: always news, however quick.
      work.apply(WorkSnapshot(7, [...work.items, const WorkItem(id: 'm', ts: 1, summary: 'Need your OK', kind: 'message')]));
      expect(told.last, 'Need your OK');
      expect(formatDuration(const Duration(seconds: 75)), '1 min 15 s');
    });
  });

  group('look up screen', () {
    test('clock and weather packet', () {
      final now = DateTime(2026, 10, 27, 14, 5, 30);
      final p = Proto.timeWeatherPacket(7, now, icon: G1Icon.sunny, tempC: -3, fahrenheit: true, h24: false);
      expect(p.length, 21);
      expect(p.sublist(0, 5), [0x06, 0x15, 0x00, 7, 0x01]);
      final b = ByteData.sublistView(p);
      final localSecs = (now.millisecondsSinceEpoch + now.timeZoneOffset.inMilliseconds) ~/ 1000;
      expect(b.getUint32(5, Endian.little), localSecs);
      expect(b.getUint32(9, Endian.little) + b.getUint32(13, Endian.little) * 0x100000000, localSecs * 1000);
      expect(p[17], G1Icon.sunny);
      expect(b.getInt8(18), -3);
      expect(p.sublist(19), [1, 0]);
    });

    test('layout and note packets', () {
      expect(Proto.dashboardModePacket(3, mode: 1, pane: 0), [0x06, 0x07, 0x00, 3, 0x06, 1, 0]);
      final n = Proto.notePacket(1, 2, 'Claude', 'Done.');
      expect(n[0], 0x1E);
      expect(n[1] | n[2] << 8, n.length);
      expect(n.sublist(4, 11), [0x03, 0x01, 0x00, 0x01, 0x00, 2, 0x01]);
      expect(n[11], 6);
      expect(String.fromCharCodes(n.sublist(12, 18)), 'Claude');
      expect(n[18] | n[19] << 8, 5);
      expect(String.fromCharCodes(n.sublist(20)), 'Done.');
      final del = Proto.notePacket(1, 3, '', '');
      expect(del, [0x1E, 11, 0, 1, 0x03, 0x01, 0x00, 0x01, 0x00, 3, 0x00]);
      expect(Proto.notePacket(1, 1, 'T' * 80, 'x' * 500).length, 4 + 7 + 1 + 40 + 2 + 280);
    });

    test('weather codes to G1 icons', () {
      expect(G1Icon.fromWmo(0), G1Icon.sunny);
      expect(G1Icon.fromWmo(0, day: false), G1Icon.night);
      expect(G1Icon.fromWmo(63), G1Icon.rain);
      expect(G1Icon.fromWmo(75), G1Icon.snow);
      expect(G1Icon.fromWmo(99), G1Icon.thunderstorm);
      expect(G1Icon.fromWmo(45), G1Icon.fog);
    });

    test('notes come from the sources you pick', () {
      final d = Dashboard(Proto(BleManager(FakeGlasses())))
        ..claudePanel = false
        ..slots = ['work', 'reply', 'custom', 'off']
        ..custom = 'Groceries\nmilk, eggs'
        ..lastReply = 'Sunny and **22** degrees.'
        ..lastWork = WorkItem(
            id: 'w', ts: DateTime(2026, 10, 27, 9, 30).millisecondsSinceEpoch / 1000, summary: 'Pushed it.', project: 'even-suite');
      final notes = d.buildNotes();
      expect(notes[0].$1, 'Claude done 9:30am');
      expect(notes[0].$2, 'Pushed it.\neven-suite');
      expect(notes[1], ('Last answer', 'Sunny and 22 degrees.'));
      expect(notes[2], ('Groceries', 'milk, eggs'));
      expect(notes[3], ('', ''));
      d.dispose();
    });

    test('Claude\'s work note is live while it runs', () {
      final started = DateTime.now().subtract(const Duration(minutes: 4, seconds: 10));
      final d = Dashboard(Proto(BleManager(FakeGlasses())))
        ..claudePanel = false
        ..slots = ['work', 'off', 'off', 'off']
        ..jobs = [
          WorkJob(
              session: 's1',
              project: 'even-suite',
              prompt: 'fix the build',
              step: 'Running the tests',
              steps: 7,
              started: started.millisecondsSinceEpoch / 1000),
        ];
      final (title, text) = d.buildNotes().first;
      expect(title, 'Claude working 4 min');
      expect(text, 'Running the tests\n"fix the build"\neven-suite - 7 steps');
      d.jobs = const [];
      d.asking = ('Searching the web: boston', 'what is the weather', '');
      expect(d.buildNotes().first, ('Claude working', 'Searching the web: boston\n"what is the weather"'));
      d.dispose();
    });

    test('looking up sends the changed note at once, and keeps it live', () async {
      final glasses = FakeGlasses();
      final control = FakeControl();
      final ble = BleManager(glasses, control: control);
      await control.connectNow();
      final d = Dashboard(Proto(ble))
        ..slots = ['work', 'off', 'off', 'off']
        ..liveEvery = const Duration(milliseconds: 50);
      await d.sync(force: true);
      glasses.clear();
      d.jobs = [WorkJob(session: 's', step: 'Reading a.dart', started: DateTime.now().millisecondsSinceEpoch / 1000)];
      d.onHeadUp();
      await Future<void>.delayed(const Duration(milliseconds: 20));
      expect(glasses.sent['L']!.where((p) => p[0] == 0x1E), isNotEmpty, reason: 'the note, right away');
      expect(glasses.sent['L']!.where((p) => p[0] == 0x06), isEmpty, reason: 'clock was fresh: notes only');
      expect(d.lookingUp, isTrue);
      glasses.clear();
      d.jobs = [WorkJob(session: 's', step: 'Editing a.dart', steps: 1, started: DateTime.now().millisecondsSinceEpoch / 1000)];
      await Future<void>.delayed(const Duration(milliseconds: 150));
      expect(glasses.sent['L']!.where((p) => p[0] == 0x1E), isNotEmpty, reason: 'updates while you look');
      d.onHeadDown();
      expect(d.lookingUp, isFalse);
      glasses.clear();
      d.jobs = [WorkJob(session: 's', step: 'Writing b.dart', steps: 2, started: DateTime.now().millisecondsSinceEpoch / 1000)];
      await Future<void>.delayed(const Duration(milliseconds: 150));
      expect(glasses.sent['L']!.where((p) => p[0] == 0x1E), isEmpty, reason: 'not while you look down');
      d.dispose();
      ble.dispose();
    });

    test('sync sends clock, layout and notes to both arms', () async {
      final glasses = FakeGlasses();
      final control = FakeControl();
      final ble = BleManager(glasses, control: control);
      await control.connectNow();
      final d = Dashboard(Proto(ble))
        ..slots = ['custom', 'off', 'off', 'off']
        ..custom = 'Hi\nthere';
      await d.sync(force: true);
      for (final arm in ['L', 'R']) {
        final cmds = glasses.sent[arm]!.map((p) => p[0]).toList();
        expect(cmds.where((c) => c == 0x06).length, 2, reason: 'clock + layout on $arm');
        expect(cmds.where((c) => c == 0x1E).length, 4, reason: 'four note slots on $arm');
      }
      glasses.clear();
      await d.sync();
      expect(glasses.sent['L']!.where((p) => p[0] == 0x1E), isEmpty, reason: 'unchanged notes are not resent');
      d.dispose();
      ble.dispose();
    });
  });
}
