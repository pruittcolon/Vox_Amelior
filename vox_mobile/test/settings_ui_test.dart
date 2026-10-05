import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';
import 'package:vox_amelior_mobile/ui/listening_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/mic_tune.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/speakers_settings_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_settings_ui_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> show(WidgetTester tester, Widget screen, {double textScale = 1.3}) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3; // 360 × 760, a small phone
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: VoxTheme.light(),
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(textScale)),
        child: child!,
      ),
      home: screen,
    ));
    await tester.pumpAndSettle();
  }

  Future<void> scrollTo(WidgetTester tester, Finder f) async {
    await tester.scrollUntilVisible(f, 200, scrollable: find.byType(Scrollable).first);
    await tester.pumpAndSettle();
  }

  /// Brings the slider with [title] into view and taps its − ('Less') or + ('More') button.
  Future<void> nudge(WidgetTester tester, String title, String direction, {int times = 1}) async {
    await scrollTo(tester, find.text(title));
    final button = find.descendant(
      of: find.ancestor(of: find.text(title), matching: find.byType(SettingSlider)),
      matching: find.byTooltip(direction),
    );
    for (var i = 0; i < times; i++) {
      await tester.tap(button);
      await tester.pump();
    }
    await tester.pumpAndSettle();
  }

  group('mic boost text', () {
    test('shows the change as a percentage and reads it back', () {
      expect(formatBoost(1.15), '+15%');
      expect(formatBoost(1.0), 'None');
      expect(formatBoost(0.9), '−10%');
      expect(formatBoost(2.5), '+150%');
      expect(parseBoost('15'), closeTo(1.15, 1e-9));
      expect(parseBoost('+15%'), closeTo(1.15, 1e-9));
      expect(parseBoost('-10'), closeTo(0.9, 1e-9));
      expect(parseBoost('−10%'), closeTo(0.9, 1e-9));
      expect(parseBoost('loud'), isNull);
    });

    test('slider values land exactly on their steps', () {
      expect(snapToStep(0.6000000000000001, min: 0.3, max: 1.5, step: 0.05), 0.6);
      expect(snapToStep(0.3 + 8 * 0.05, min: 0.3, max: 1.5, step: 0.05), 0.7);
      expect(snapToStep(9, min: 0.5, max: 4, step: 0.05), 4.0);
      expect(snapToStep(-1, min: 0.5, max: 4, step: 0.05), 0.5);
      expect(snapToStep(double.nan, min: 0.5, max: 4, step: 0.05), 0.5);
      expect(snapToStep(1.149999, min: 0.5, max: 4, step: 0.05), 1.15);
    });

    test('the exact-value box starts with the shown number, sign included', () {
      expect(numberIn('+15%'), '15');
      expect(numberIn('\u221210%'), '-10');
      expect(numberIn('0.65 s'), '0.65');
      expect(numberIn('3 words'), '3');
      expect(numberIn('None'), '0');
      expect(parseBoost(numberIn(formatBoost(0.9))), closeTo(0.9, 1e-9), reason: 'opening and saving keeps -10%');
    });

    test('seconds are shown the same way everywhere', () {
      expect(formatSeconds(0.6000000000000001), '0.6 s');
      expect(formatSeconds(0.65), '0.65 s');
      expect(formatSeconds(1.5), '1.5 s');
      expect(formatSeconds(0.3), '0.3 s');
    });

    test('presets match only their exact values; the default is Balanced', () {
      expect(MicPreset.of(const AppSettings())!.name, 'Balanced');
      expect(MicPreset.of(const AppSettings(micGain: 1.5, vadThreshold: 0.4))!.name, 'Sensitive');
      expect(MicPreset.of(const AppSettings(micGain: 1.0, vadThreshold: 0.65))!.name, 'Noise-proof');
      expect(MicPreset.of(const AppSettings(micGain: 1.3)), isNull);
    });
  });

  group('level meter', () {
    testWidgets('renders every state without layout errors', (tester) async {
      for (final r in const [
        MicReading(),
        MicReading(live: true, level: 0.1, peak: 0.2),
        MicReading(live: true, level: 0.6, peak: 0.7),
        MicReading(live: true, level: 0.95, peak: 1, clipping: true),
      ]) {
        await show(tester, Scaffold(body: Padding(padding: const EdgeInsets.all(16), child: LevelMeterBar(reading: r))));
        expect(find.bySemanticsLabel('Microphone level'), findsOneWidget);
      }
    });

    testWidgets('tells the person what to do', (tester) async {
      late BuildContext ctx;
      await show(tester, Builder(builder: (c) {
        ctx = c;
        return const SizedBox();
      }));
      String say(MicReading r, {bool listening = true, double recent = 0.6}) => micVerdict(ctx, r, listening: listening, recentLevel: recent).text;
      expect(say(const MicReading(), listening: false), contains('Turn Vox on'));
      expect(say(const MicReading()), contains('Waiting'));
      expect(say(const MicReading(live: true, level: 0.1), recent: 0.1), contains('Raise the boost'));
      expect(say(const MicReading(live: true, level: 0.6)), 'Good level.');
      expect(say(const MicReading(live: true, level: 0.99, clipping: true)), contains('Lower the boost'));
    });
  });

  group('microphone & hearing page', () {
    testWidgets('boost starts at +15% and − / + nudge it by 5%', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      expect(find.text('+15%'), findsOneWidget);
      await nudge(tester, 'Mic boost', 'More');
      expect(s.settings.value.micGain, closeTo(1.20, 1e-9));
      expect(find.text('+20%'), findsOneWidget);
      await nudge(tester, 'Mic boost', 'Less', times: 2);
      expect(s.settings.value.micGain, closeTo(1.10, 1e-9), reason: 'quick taps add up');
    });

    testWidgets('the exact value can be typed, and reset returns to +15%', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await tester.tap(find.text('+15%'));
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), '40');
      await tester.tap(find.text('Set'));
      await tester.pumpAndSettle();
      expect(s.settings.value.micGain, closeTo(1.40, 1e-9));
      await tester.tap(find.byTooltip('Back to +15%'));
      await tester.pumpAndSettle();
      expect(s.settings.value.micGain, closeTo(1.15, 1e-9));
      expect(find.byTooltip('Back to +15%'), findsNothing, reason: 'nothing to reset at the default');
    });

    testWidgets('typed values outside the range are held to the range', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await tester.tap(find.text('+15%'));
      await tester.pumpAndSettle();
      expect(find.widgetWithText(TextField, '15'), findsOneWidget, reason: 'starts with the current value');
      await tester.enterText(find.byType(TextField), '900');
      await tester.tap(find.text('Set'));
      await tester.pumpAndSettle();
      expect(s.settings.value.micGain, 4.0);
    });

    testWidgets('a preset sets boost and sensitivity together', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await tester.tap(find.text('Sensitive'));
      await tester.pumpAndSettle();
      expect(s.settings.value.micGain, 1.5);
      expect(s.settings.value.vadThreshold, 0.4);
      await tester.tap(find.text('Noise-proof'));
      await tester.pumpAndSettle();
      expect(s.settings.value.micGain, 1.0);
      expect(s.settings.value.vadThreshold, 0.65);
    });

    testWidgets('sensitivity is shown the natural way round (higher hears more)', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await scrollTo(tester, find.text('Sensitivity'));
      expect(find.text('50%'), findsOneWidget);
      await nudge(tester, 'Sensitivity', 'More');
      expect(s.settings.value.vadThreshold, closeTo(0.45, 1e-9), reason: 'more sensitive = lower threshold');
    });

    testWidgets('sentence pause and short-sound sliders are adjustable', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await nudge(tester, 'Pause that ends a sentence', 'More');
      expect(s.settings.value.pauseSeconds, closeTo(0.65, 1e-9));
      await nudge(tester, 'Ignore short sounds', 'Less');
      expect(s.settings.value.minSpeechSeconds, closeTo(0.25, 1e-9));
    });

    testWidgets('both speech models are offered and fp16 is selected by default', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s));
      await scrollTo(tester, find.text('High precision (fp16)'));
      expect(find.text('Standard (int8)'), findsOneWidget);
      expect(find.textContaining('1.1 GB download'), findsOneWidget);
      expect(s.settings.value.speechModel, 'fp16');
    });

    testWidgets('fits a small phone at large text', (tester) async {
      await show(tester, ListeningSettingsScreen(services: s), textScale: 1.6);
      expect(tester.takeException(), isNull);
    });
  });

  group('restore recommended', () {
    testWidgets('puts listening settings back and leaves the rest alone', (tester) async {
      await s.updateSettings(const AppSettings(
        micGain: 3.0,
        vadThreshold: 0.3,
        pauseSeconds: 1.2,
        minSpeechSeconds: 0.8,
        matchThreshold: 0.8,
        matchMargin: 0.2,
        guestThreshold: 0.8,
        multiPatterns: false,
        splitSpeakers: false,
        splitMinSeconds: 3.0,
        splitMinWords: 5,
        retentionDays: 7,
        accent: 0xFFC2255C,
      ));
      await show(tester, SettingsScreen(services: s));
      await scrollTo(tester, find.text('Restore recommended settings'));
      await tester.tap(find.text('Restore recommended settings'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Restore').last);
      await tester.pumpAndSettle();
      final st = s.settings.value;
      expect(st.micGain, 1.15);
      expect(st.vadThreshold, 0.5);
      expect(st.pauseSeconds, 0.6);
      expect(st.minSpeechSeconds, 0.3);
      expect(st.matchThreshold, 0.55);
      expect(st.matchMargin, 0.04);
      expect(st.guestThreshold, 0.6);
      expect(st.multiPatterns, isTrue);
      expect(st.splitSpeakers, isTrue);
      expect(st.splitMinSeconds, 1.5);
      expect(st.splitMinWords, 2);
      expect(st.retentionDays, 7, reason: 'not a listening setting');
      expect(st.accent, 0xFFC2255C);
    });
  });

  group('voices & speakers page', () {
    testWidgets('naming and splitting can be tuned; split options hide when splitting is off', (tester) async {
      await show(tester, SpeakersSettingsScreen(services: s));
      expect(find.text('Shortest part of a line'), findsNothing, reason: 'below the fold');
      await nudge(tester, 'Fewest words in a part', 'More');
      expect(s.settings.value.splitMinWords, 3);
      await nudge(tester, 'Shortest part of a line', 'More');
      expect(s.settings.value.splitMinSeconds, closeTo(1.6, 1e-9));
      await scrollTo(tester, find.text('Split lines when the speaker changes'));
      await tester.tap(find.byType(Switch).last);
      await tester.pumpAndSettle();
      expect(s.settings.value.splitSpeakers, isFalse);
      expect(find.text('Shortest part of a line'), findsNothing);
    });

    testWidgets('the naming margin is now adjustable', (tester) async {
      await show(tester, SpeakersSettingsScreen(services: s));
      await scrollTo(tester, find.text('Lead over the runner-up'));
      expect(find.text('4%'), findsOneWidget);
      await nudge(tester, 'Lead over the runner-up', 'More');
      expect(s.settings.value.matchMargin, closeTo(0.05, 1e-9));
    });
  });
}
