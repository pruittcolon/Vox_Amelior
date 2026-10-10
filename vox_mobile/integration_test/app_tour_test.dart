import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/main.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';

/// The whole app on an Android emulator, as a person would use it (CI:
/// .github/workflows/app-tour.yml, which also takes the screenshots):
///
/// first-run setup with the real model downloads; listening to a recorded
/// two-person conversation with the real speech, voice and tone models;
/// the Timeline, a conversation, search by meaning, Insights, Ask, People
/// and every settings page; then a large archive, scrolled quickly while
/// frame times are measured.
void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('tour of the app', (tester) async {
    final tour = _Tour(tester, binding);
    await tour.run();
  }, timeout: const Timeout(Duration(minutes: 100)));
}

class _Tour {
  _Tour(this.t, this.binding);

  final WidgetTester t;
  final IntegrationTestWidgetsFlutterBinding binding;
  late final AppServices s;

  /// This machine, as the emulator sees it (tool/run_tour.sh serves the speech there).
  static const String host = 'http://10.0.2.2:8765';

  Future<void> run() async {
    binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;
    if (!t.testTextInput.isRegistered) t.testTextInput.register();
    await prepareApp();
    s = await AppServices.create();
    await t.pumpWidget(VoxApp(services: Future.value(s)));
    await step('first-run setup', _setup);
    await step('listening to a conversation', _listen);
    await step('timeline and conversation', _timeline);
    await step('search', _search);
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
    await t.tap(download);
    await wait(const Duration(seconds: 10));
    await shot('setup-downloading');
    // Tone of voice and search by meaning, queued behind the speech models.
    await s.updateSettings(s.settings.value.copyWith(hearTone: true));
    await s.downloads.download(ModelCatalog.toneModel);
    await s.downloads.download(ModelCatalog.textEmbedder);
    await until('the speech models', () => s.speechReady, timeout: const Duration(minutes: 30));
    await until('Continue', () => find.text('Continue').evaluate().isNotEmpty);
    await shot('setup-ready');
    await t.tap(find.text('Continue'));
    await wait(const Duration(seconds: 2));
    await shot('now-first');
    await until(
      'tone and search models',
      () => s.models.isInstalled(ModelCatalog.toneModel) && s.models.isInstalled(ModelCatalog.textEmbedder),
      timeout: const Duration(minutes: 20),
    );
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
    }
    final lines = chunks.fold<int>(0, (a, c) => a + (c['lines']! as List).length);
    await tapText('Start listening');
    await wait(const Duration(seconds: 6));
    await shot('now-starting');
    await until('the conversation transcribed', () => s.transcripts.count() >= lines - 2, timeout: const Duration(minutes: 20));
    // Second pass: speaker changes, final names, tone.
    await until('the speech queue to empty', () => queue.length == 0, timeout: const Duration(minutes: 5));
    await wait(const Duration(seconds: 25));
    await shot('now-heard');
    await scrollDown(find.byType(CustomScrollView).first);
    await shot('now-heard-more');
    await tapText('Turn off');
    await wait(const Duration(seconds: 3));
    final said = s.transcripts.recent(limit: 100).map((l) => l.text.toLowerCase()).join(' ');
    note('transcribed ${s.transcripts.count()} lines; '
        'voices: ${s.speakers.clusters().map((c) => '${c.label} (${c.count})').join(', ')}');
    for (final word in ['electric bill', 'landlord', 'dentist', 'vet']) {
      expect(said, contains(word), reason: 'the recording says "$word"');
    }
  }

  Future<void> _timeline() async {
    await tab('Timeline');
    await shot('timeline');
    await t.tap(find.byType(Card).first);
    await settle();
    await shot('conversation');
    await scrollDown(find.byType(ListView).last);
    await shot('conversation-end');
    // Naming a voice from one of its lines.
    await t.tap(find.textContaining('electric bill').first);
    await settle();
    await shot('conversation-line-sheet');
    await back();
    await back();
  }

  Future<void> _search() async {
    await t.enterText(find.byType(TextField).first, 'money worries');
    await until('meaning search results', () => find.textContaining('meaning').evaluate().isNotEmpty, timeout: const Duration(minutes: 5));
    await wait(const Duration(seconds: 2));
    await shot('search-meaning');
    await t.tap(find.byIcon(Icons.close_rounded).first);
    await settle();
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
  }

  Future<void> _settings() async {
    await t.tap(find.byTooltip('Settings and more').first);
    await settle();
    await shot('more');
    for (final page in ['Models', 'Places', 'Automations', 'Appearance', 'Voice clips', 'Settings']) {
      await tapText(page);
      await shot('more-${page.toLowerCase().replaceAll(' ', '-')}');
      if (page == 'Models' || page == 'Settings') {
        await scrollDown(find.byType(Scrollable).last);
        await shot('more-${page.toLowerCase()}-more');
      }
      await back();
    }
    await back();
  }

  Future<void> _dark() async {
    await s.updateSettings(s.settings.value.copyWith(themeMode: 'dark'));
    await settle();
    for (final name in ['Timeline', 'Insights', 'People']) {
      await tab(name);
      await shot('dark-${name.toLowerCase()}');
    }
    await s.updateSettings(s.settings.value.copyWith(themeMode: 'system'));
    await settle();
  }

  /// A year of conversations, then the busiest screens scrolled hard while
  /// frame times are recorded. Meaning search indexes the new lines in the
  /// background meanwhile, as it would on a phone.
  Future<void> _performance() async {
    final lines = _seedArchive();
    note('archive: $lines lines');
    s.dataChanged();
    await settle();

    await tab('Timeline');
    await measure('timeline-day-strip', () async {
      for (var i = 0; i < 4; i++) {
        await t.fling(find.byKey(const ValueKey('timeline-days')), const Offset(-600, 0), 3000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await t.tap(find.text('Pruitt').first);
    await settle();
    await shot('timeline-pruitt');
    await measure('timeline-conversations', () async {
      for (var i = 0; i < 6; i++) {
        await t.fling(find.byKey(const ValueKey('timeline-filtered')), const Offset(0, -900), 4000);
        await wait(const Duration(milliseconds: 900));
      }
    });
    await t.tap(find.text('All').first);
    await settle();

    await t.tap(find.byTooltip('Pick a date'));
    await settle();
    await shot('timeline-date-picker');
    await back();

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
    await shot('people-archive');
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
      if (watch.elapsed > timeout) fail('timed out after ${timeout.inMinutes} min waiting for $what');
      if (watch.elapsed.inSeconds ~/ 30 > lastNote) {
        lastNote = watch.elapsed.inSeconds ~/ 30;
        note('still waiting for $what (${watch.elapsed.inSeconds} s)${_downloadNote()}');
      }
      await t.pump(const Duration(milliseconds: 500));
    }
  }

  String _downloadNote() {
    final cur = s.downloads.current;
    return cur == null ? '' : '; downloading ${cur.asset.title}: ${cur.state.status.name} ${(cur.state.progress ?? 0) * 100 ~/ 1}%';
  }

  Future<void> tab(String label) async {
    await t.tap(find.descendant(of: find.byType(NavigationBar), matching: find.text(label)));
    await settle();
  }

  Future<void> tapText(String text) async {
    final f = find.text(text);
    await until('"$text"', () => f.evaluate().isNotEmpty);
    await t.ensureVisible(f.first);
    await settle();
    await t.tap(f.first);
    await settle();
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
