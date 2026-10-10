import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/main.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';
import 'package:vox_amelior_mobile/ui/conversation_screen.dart';

/// The whole app on an Android emulator, as a person would use it (CI:
/// .github/workflows/app-tour.yml, which also takes the screenshots):
///
/// first-run setup with the real model downloads; switching on tone of
/// voice and search by meaning in Models; listening to a recorded
/// two-person conversation with the real speech, voice and tone models;
/// naming the two voices; the Timeline, a conversation, search by meaning
/// (then with the small search model, switched in Models); Insights, Ask,
/// People and every settings page, also in dark mode; then a year-long
/// archive, scrolled quickly while frame times are measured.
void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('tour of the app', (tester) async {
    final tour = _Tour(tester, binding);
    await tour.run();
  }, timeout: const Timeout(Duration(minutes: 130)));
}

class _Tour {
  _Tour(this.t, this.binding);

  final WidgetTester t;
  final IntegrationTestWidgetsFlutterBinding binding;
  late final AppServices s;

  /// The recorded conversation: who said each line ('ryan' or 'amy').
  final List<({String voice, String text})> _script = [];

  /// The names the tour gives the two voices.
  static const Map<String, String> names = {'ryan': 'Pruitt', 'amy': 'Ericah'};

  /// This machine, as the emulator sees it (tool/run_tour.sh serves the speech there).
  static const String host = 'http://10.0.2.2:8765';

  Future<void> run() async {
    binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;
    if (!t.testTextInput.isRegistered) t.testTextInput.register();
    await prepareApp();
    s = await AppServices.create();
    await t.pumpWidget(VoxApp(services: Future.value(s)));
    await step('first-run setup', _setup);
    await step('tone and search switched on in Models', _optionalModels);
    await step('listening to a conversation', _listen);
    await step('naming the voices', _nameVoices);
    await step('timeline and conversation', _timeline);
    await step('search by meaning', _search);
    await step('switching the search model', _switchSearchModel);
    await step('insights', _insights);
    await step('ask', _ask);
    await step('people', _people);
    await step('settings', _settings);
    await step('dark mode', _dark);
    await step('a large archive, scrolled', _performance);
  }

  // ---- the tour ------------------------------------------------------------

  Future<void> _setup() async {
    final download = find.textContaining('Download speech models');
    await until('the setup screen', () => download.evaluate().isNotEmpty);
    await shot('setup');
    await t.tap(download.first);
    await wait(const Duration(seconds: 10));
    await shot('setup-downloading');
    await until('the speech models', () => s.speechReady, timeout: const Duration(minutes: 30));
    await scrollTo(find.text('Continue'), find.byKey(const ValueKey('models-list')));
    await shot('setup-ready');
    await t.tap(find.text('Continue'));
    await wait(const Duration(seconds: 2));
    await shot('now-first');
  }

  Future<void> _optionalModels() async {
    await openModels();
    await shot('models');
    await tapText('On · SenseVoice Small');
    await tapText('Standard · 8-bit');
    await wait(const Duration(seconds: 6));
    await shot('models-downloading');
    await until(
      'tone and search models',
      () => s.models.isInstalled(ModelCatalog.toneModel) && s.models.isInstalled(ModelCatalog.textEmbedder),
      timeout: const Duration(minutes: 20),
    );
    await shot('models-ready');
    await scrollTo(find.text('Assistant'), find.byKey(const ValueKey('models-list')));
    await shot('models-assistant');
    await back();
    await back();
  }

  Future<void> _listen() async {
    // The recorded conversation goes where the microphone's speech goes;
    // starting to listen picks it up.
    final manifest = jsonDecode(utf8.decode(await fetch('manifest.json'))) as Map<String, Object?>;
    final chunks = (manifest['chunks']! as List).cast<Map<String, Object?>>();
    final queue = ChunkQueue(Directory(p.join(s.supportDir.path, 'speech_queue')));
    final start = DateTime.now().subtract(const Duration(minutes: 20));
    for (final c in chunks) {
      final bytes = await fetch(c['file']! as String);
      queue.push(bytes.buffer.asFloat32List(0, bytes.length ~/ 4), start.add(Duration(milliseconds: c['offset_ms']! as int)));
      for (final l in (c['lines']! as List).cast<Map<String, Object?>>()) {
        _script.add((voice: l['voice']! as String, text: l['text']! as String));
      }
    }
    await tab('Now');
    await tapText('Start listening');
    await wait(const Duration(seconds: 8));
    await shot('now-starting');
    await until('the conversation transcribed', () => s.transcripts.count() >= _script.length - 3, timeout: const Duration(minutes: 20));
    // Second pass: speaker changes, final names, tone.
    await until('the speech queue to empty', () => queue.length == 0, timeout: const Duration(minutes: 5));
    await wait(const Duration(seconds: 25));
    await shot('now-heard');
    await tapText('Turn off');
    await wait(const Duration(seconds: 3));
    final said = s.transcripts.recent(limit: 100).map((l) => l.text.toLowerCase()).join(' ');
    note('transcribed ${s.transcripts.count()} lines; voices: '
        '${s.transcripts.voicesToName(detailed: 0).map((v) => '${v.label} (${v.lines} lines)').join(', ')}');
    for (final l in s.transcripts.recent(limit: 100).reversed) {
      note('line: [${l.speakerLabel}${l.emotion == null ? '' : ', ${l.emotion}'}] ${l.text}');
    }
    for (final word in ['electric bill', 'landlord', 'dentist', 'vet']) {
      expect(said, contains(word), reason: 'the recording says "$word"');
    }
  }

  /// Names each voice from the People tab's "Who's this?", picking the
  /// name by what the voice said (as a person would recognise them).
  Future<void> _nameVoices() async {
    await tab('People');
    await shot('people-voices-to-name');
    await tapText('need a name', contains: true);
    await shot('who-is-this');
    for (var i = 0; i < 8 && find.text("Who's this?").evaluate().isNotEmpty; i++) {
      final voice = s.transcripts.voicesToName(detailed: 1, samples: 20).firstOrNull;
      if (voice == null || find.text('Who is ${voice.label}?').evaluate().isEmpty) break;
      final name = names[_voiceOf(voice.samples)]!;
      note('${voice.label} (${voice.lines} lines) sounds like $name');
      final chip = find.widgetWithText(ActionChip, name);
      if (chip.evaluate().isNotEmpty) {
        await t.tap(chip);
      } else {
        await tapText('Someone new');
        await t.enterText(find.byType(TextField).first, name);
        await settle();
        await shot('who-is-this-typing');
        await t.tap(find.byTooltip('Save name'));
      }
      await settle();
      await shot('named-${name.toLowerCase()}-$i');
    }
    await back();
    await shot('people-named');
    final named = s.speakers.profiles().map((p) => p.name).toSet();
    note('people: ${named.join(', ')}; still to name: ${s.transcripts.voicesToNameCount()}');
    if (!named.containsAll(names.values)) note('WARNING: expected both ${names.values.join(' and ')} after naming');
  }

  /// The voice ('ryan' or 'amy') whose script lines best match [lines].
  String _voiceOf(List<SegmentView> lines) {
    Set<String> words(String text) => RegExp(r'[a-z]+').allMatches(text.toLowerCase()).map((m) => m.group(0)!).toSet();
    final votes = <String, int>{};
    for (final l in lines) {
      final said = words(l.text);
      var best = 0.0;
      String? who;
      for (final line in _script) {
        final w = words(line.text);
        final overlap = said.intersection(w).length / max(1, said.union(w).length);
        if (overlap > best) {
          best = overlap;
          who = line.voice;
        }
      }
      if (who != null) votes[who] = (votes[who] ?? 0) + 1;
    }
    return votes.entries.fold<MapEntry<String, int>?>(null, (a, e) => a == null || e.value > a.value ? e : a)?.key ?? 'ryan';
  }

  Future<void> _timeline() async {
    await tab('Timeline');
    await shot('timeline');
    await t.tap(find.byType(Card).first);
    await settle();
    await shot('conversation');
    await scrollDown(find.byKey(const ValueKey('conversation-lines')));
    await shot('conversation-more');
    await scrollDown(find.byKey(const ValueKey('conversation-lines')));
    await shot('conversation-end');
    await t.tap(find.textContaining('electric bill').first);
    await settle();
    await shot('conversation-line-sheet');
    await back();
    await back();
  }

  Future<void> _search() async {
    await tab('Timeline');
    await t.enterText(find.byType(TextField).first, 'money worries');
    await until('meaning search results', () => find.textContaining('meaning').evaluate().length > 1, timeout: const Duration(minutes: 5));
    await wait(const Duration(seconds: 3));
    await shot('search-meaning');
    await t.enterText(find.byType(TextField).first, 'when is the vet');
    await wait(const Duration(seconds: 5));
    await shot('search-vet');
    await t.tap(find.byIcon(Icons.close_rounded).first);
    await settle();
  }

  /// Search by meaning with the small model: chosen in Models, downloaded,
  /// every line prepared again, then searched. Then back to standard,
  /// whose lines are still prepared.
  Future<void> _switchSearchModel() async {
    await openModels();
    await scrollTo(find.text('Small · 4-bit'), find.byKey(const ValueKey('models-list')));
    await tapText('Small · 4-bit');
    await wait(const Duration(seconds: 5));
    await shot('models-switching-search');
    await until('the small search model', () => s.models.isInstalled(ModelCatalog.textEmbedderSmall), timeout: const Duration(minutes: 15));
    await until('lines prepared for the small model', () {
      final p = s.vectors.progress(s.searchModelId);
      return s.searchModelId == ModelCatalog.textEmbedderSmall.id && p.total > 0 && p.done >= p.total;
    }, timeout: const Duration(minutes: 10));
    await shot('models-switched-search');
    await back();
    await back();
    await tab('Timeline');
    await t.enterText(find.byType(TextField).first, 'money worries');
    await until('meaning search results', () => find.textContaining('meaning').evaluate().length > 1, timeout: const Duration(minutes: 5));
    await wait(const Duration(seconds: 3));
    await shot('search-meaning-small-model');
    await t.tap(find.byIcon(Icons.close_rounded).first);
    await settle();
    await openModels();
    await scrollTo(find.text('Standard · 8-bit'), find.byKey(const ValueKey('models-list')));
    await tapText('Standard · 8-bit');
    await until('back on the standard search model', () => s.searchModelId == ModelCatalog.textEmbedder.id);
    await back();
    await back();
  }

  Future<void> _insights() async {
    await tab('Insights');
    await shot('insights');
    await scrollDown(find.byKey(const ValueKey('insights-list')));
    await shot('insights-more');
    await scrollDown(find.byKey(const ValueKey('insights-list')));
    await shot('insights-end');
  }

  Future<void> _ask() async {
    await tab('Ask');
    await shot('ask');
  }

  Future<void> _people() async {
    await tab('People');
    await shot('people');
    await t.tap(find.text(s.speakers.profiles().first.name).first);
    await settle();
    await shot('person-insights');
    await back();
  }

  Future<void> _settings() async {
    await tab('Now');
    await t.tap(find.byTooltip('Settings and more').first);
    await settle();
    await shot('more');
    for (final page in ['Places', 'Automations', 'Appearance', 'Voice clips', 'Settings']) {
      await tapText(page);
      await shot('more-${page.toLowerCase().replaceAll(' ', '-')}');
      if (page == 'Settings') {
        await scrollDown(find.byType(Scrollable).first);
        await shot('more-settings-more');
        await scrollDown(find.byType(Scrollable).first);
        await shot('more-settings-end');
      }
      await back();
    }
    await back();
  }

  Future<void> _dark() async {
    await s.updateSettings(s.settings.value.copyWith(themeMode: 'dark'));
    await settle();
    for (final name in ['Now', 'Timeline', 'Insights', 'People']) {
      await tab(name);
      await shot('dark-${name.toLowerCase()}');
    }
    await tab('Timeline');
    await t.tap(find.byType(Card).first);
    await settle();
    await shot('dark-conversation');
    await back();
    await s.updateSettings(s.settings.value.copyWith(themeMode: 'system'));
    await settle();
  }

  /// A year of conversations, then the busiest screens scrolled hard while
  /// frame times are recorded. Meaning search prepares the new lines in the
  /// background meanwhile, as it would on a phone.
  Future<void> _performance() async {
    final watch = Stopwatch()..start();
    final lines = _seedArchive();
    note('archive: $lines lines added in ${watch.elapsed.inSeconds} s');
    s.dataChanged();
    await settle();

    await tab('Timeline');
    await shot('timeline-archive');
    await measure('timeline-day-strip', () async {
      for (var i = 0; i < 4; i++) {
        await t.fling(find.byKey(const ValueKey('timeline-days')), const Offset(-600, 0), 3000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await t.tap(find.widgetWithText(FilterChip, 'Pruitt'));
    await settle();
    await shot('timeline-pruitt');
    await measure('timeline-conversations', () async {
      for (var i = 0; i < 6; i++) {
        await t.fling(find.byKey(const ValueKey('timeline-filtered')), const Offset(0, -900), 4000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await t.tap(find.widgetWithText(ChoiceChip, 'All'));
    await settle();

    // The 400-line conversation three days ago.
    final day = DateTime.now().subtract(const Duration(days: 3));
    final long = s.transcripts.conversationsBetween(DateTime(day.year, day.month, day.day), DateTime(day.year, day.month, day.day + 1));
    final biggest = long.fold<ConversationSummary?>(null, (a, c) => a == null || c.segmentCount > a.segmentCount ? c : a);
    await t.tap(find.byTooltip('Pick a date'));
    await settle();
    await shot('timeline-date-picker');
    await back();
    if (biggest != null) {
      // Opened as a tap on its card would; timed to its first frame.
      final nav = t.state<NavigatorState>(find.byType(Navigator).first);
      final open = Stopwatch()..start();
      unawaited(nav.push(MaterialPageRoute<void>(builder: (_) => ConversationScreen(services: s, conversationId: biggest.id))));
      await t.pump();
      note('opened a ${biggest.segmentCount}-line conversation: first frame after ${open.elapsedMilliseconds} ms');
      await settle();
      await shot('conversation-long');
      await measure('conversation-long', () async {
        for (var i = 0; i < 6; i++) {
          await t.fling(find.byKey(const ValueKey('conversation-lines')), const Offset(0, -1200), 5000);
          await wait(const Duration(milliseconds: 900));
        }
      });
      await back();
    }

    await t.enterText(find.byType(TextField).first, 'dinner');
    await wait(const Duration(seconds: 3));
    await shot('search-words-archive');
    await measure('search-results', () async {
      for (var i = 0; i < 4; i++) {
        await t.fling(find.byKey(const ValueKey('timeline-results')), const Offset(0, -900), 4000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await t.tap(find.byIcon(Icons.close_rounded).first);
    await settle();

    await tab('Insights');
    await measure('insights', () async {
      for (var i = 0; i < 3; i++) {
        await t.fling(find.byKey(const ValueKey('insights-list')), const Offset(0, -900), 3000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await shot('insights-archive');
    await tab('People');
    await measure('people', () async {
      await t.fling(find.byKey(const ValueKey('people-list')), const Offset(0, -600), 3000);
      await wait(const Duration(milliseconds: 900));
    });
    await shot('people-archive');
    final p = s.vectors.progress(s.searchModelId);
    note('meaning search prepared ${p.done} of ${p.total} lines meanwhile');
    await tab('Timeline');
  }

  /// Fills the archive: three people over a year, three conversations a day
  /// of up to 40 lines (one of 400), with tones. Returns the line count.
  int _seedArchive() {
    final rnd = Random(7);
    final people = {for (final p in s.speakers.profiles()) p.name: p.id};
    String person(String name) =>
        people[name] ??= s.speakers.create(name: name, embeddingModel: 'tour', samples: [_vector(rnd)]).id;
    final ids = [person('Pruitt'), person('Ericah'), person('Grandma')];
    const said = [
      'Did you remember to pay the water bill this month?',
      'I think we should take the dog to the park after dinner.',
      'Work was exhausting today, the meeting went on forever.',
      'What do you want for dinner tonight?',
      'Can you pick up milk and eggs on the way home?',
      'I love you, thank you for helping with the kids.',
      'The car is making that noise again, we need to get it checked.',
      'Let us plan a trip to the coast for the long weekend.',
      'I am so tired of arguing about the dishes.',
      'My mom called, she wants to visit next Sunday.',
      'The rent went up again, I do not know how we will manage.',
      'That movie last night was hilarious.',
    ];
    const tones = ['neutral', 'neutral', 'neutral', 'happy', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted'];
    var n = 0;
    final today = DateTime.now();
    for (var day = 1; day <= 365; day++) {
      final date = DateTime(today.year, today.month, today.day - day);
      for (var c = 0; c < 3; c++) {
        var at = date.add(Duration(hours: 8 + c * 4, minutes: rnd.nextInt(50)));
        final count = day == 3 && c == 0 ? 400 : 4 + rnd.nextInt(36);
        for (var i = 0; i < count; i++) {
          final seg = s.transcripts.addSegment(
            text: said[rnd.nextInt(said.length)],
            startedAt: at,
            duration: const Duration(seconds: 3),
            speakerId: ids[rnd.nextInt(i % 7 == 0 ? 3 : 2)],
          );
          s.transcripts.setTone(seg.id, emotion: tones[rnd.nextInt(tones.length)]);
          at = at.add(const Duration(seconds: 6));
          n++;
        }
      }
    }
    return n;
  }

  static Float32List _vector(Random rnd) {
    final v = Float32List(192);
    for (var i = 0; i < v.length; i++) {
      v[i] = rnd.nextDouble() - 0.5;
    }
    return v;
  }

  // ---- helpers ---------------------------------------------------------------

  Future<void> step(String name, Future<void> Function() body) async {
    note('step: $name');
    final watch = Stopwatch()..start();
    try {
      await body();
    } finally {
      note('step done in ${watch.elapsed.inSeconds} s: $name');
    }
  }

  /// Logged for the CI log (and the screenshot watcher).
  void note(String text) => debugPrint('TOUR $text');

  /// Asks the machine running the emulator for a screenshot, and gives it
  /// time to take it.
  Future<void> shot(String name) async {
    await settle();
    debugPrint('TOUR_SHOT ${(++_shots).toString().padLeft(2, '0')}-$name');
    await wait(const Duration(seconds: 3));
  }

  int _shots = 0;

  /// Lets animations finish (spinners never do; then it just waits a while).
  Future<void> settle({Duration max = const Duration(seconds: 4)}) async {
    final end = DateTime.now().add(max);
    await t.pump(const Duration(milliseconds: 100));
    while (binding.hasScheduledFrame && DateTime.now().isBefore(end)) {
      await t.pump(const Duration(milliseconds: 100));
    }
  }

  Future<void> wait(Duration d) async {
    final end = DateTime.now().add(d);
    while (DateTime.now().isBefore(end)) {
      await t.pump(const Duration(milliseconds: 200));
    }
  }

  Future<void> until(String what, bool Function() done, {Duration timeout = const Duration(minutes: 2)}) async {
    final watch = Stopwatch()..start();
    var lastNote = 0;
    while (!done()) {
      if (watch.elapsed > timeout) {
        await shot('timed-out');
        fail('timed out after ${timeout.inMinutes} min waiting for $what');
      }
      if (watch.elapsed.inSeconds ~/ 30 > lastNote) {
        lastNote = watch.elapsed.inSeconds ~/ 30;
        note('still waiting for $what (${watch.elapsed.inSeconds} s)${_downloadNote()}');
      }
      await t.pump(const Duration(milliseconds: 500));
    }
  }

  String _downloadNote() {
    final cur = s.downloads.current;
    return cur == null ? '' : '; downloading ${cur.asset.title}: ${cur.state.status.name} ${((cur.state.progress ?? 0) * 100).round()}%';
  }

  Future<void> tab(String label) async {
    await t.tap(find.descendant(of: find.byType(NavigationBar), matching: find.text(label)));
    await settle();
  }

  Future<void> tapText(String text, {bool contains = false}) async {
    final f = contains ? find.textContaining(text) : find.text(text);
    await until('"$text"', () => f.evaluate().isNotEmpty);
    await t.ensureVisible(f.first);
    await settle();
    await t.tap(f.first);
    await settle();
  }

  Future<void> openModels() async {
    await tab('Now');
    await t.tap(find.byTooltip('Settings and more').first);
    await settle();
    await tapText('Models');
  }

  Future<void> back() async {
    final nav = t.state<NavigatorState>(find.byType(Navigator).last);
    await nav.maybePop();
    await settle();
  }

  Future<void> scrollDown(Finder scrollable) async {
    await t.fling(scrollable, const Offset(0, -700), 2500);
    await settle();
  }

  Future<void> scrollTo(Finder target, Finder scrollable) async {
    for (var i = 0; i < 20 && target.evaluate().isEmpty; i++) {
      await t.drag(scrollable, const Offset(0, -300));
      await settle();
    }
    await t.ensureVisible(target.first);
    await settle();
  }

  /// Runs [action] while recording how long each frame took.
  Future<void> measure(String key, Future<void> Function() action) async {
    await binding.watchPerformance(action, reportKey: key);
    note('frames $key: ${jsonEncode(binding.reportData?[key])}');
  }

  Future<Uint8List> fetch(String path) async {
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse('$host/$path'));
      final response = await request.close();
      if (response.statusCode != 200) fail('$path: HTTP ${response.statusCode}');
      final b = BytesBuilder(copy: false);
      await response.forEach(b.add);
      return b.takeBytes();
    } finally {
      client.close();
    }
  }
}
