import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/home_shell.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

import 'support/fakes.dart';

/// The tabs together: hidden tabs wait, then catch up when shown.
void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_shell_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
    s.speakers.create(name: 'Pruitt', embeddingModel: 'f', samples: [voiceprint(1)]);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  Future<void> tab(WidgetTester tester, String label) async {
    await tester.tap(find.descendant(of: find.byType(NavigationBar), matching: find.text(label)));
    await tester.pumpAndSettle();
  }

  testWidgets('lines heard while a tab is hidden are there when it is shown', (tester) async {
    tester.view
      ..physicalSize = const Size(1080, 2280)
      ..devicePixelRatio = 3;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(theme: VoxTheme.light(), home: HomeShell(services: s)));
    await tester.pumpAndSettle();
    await tab(tester, 'Timeline');
    expect(find.textContaining('A brand new line about the garden'), findsNothing);

    await tab(tester, 'Now');
    s.transcripts.addSegment(
      text: 'A brand new line about the garden',
      startedAt: DateTime.now().subtract(const Duration(minutes: 1)),
      duration: const Duration(seconds: 2),
    );
    s.dataChanged();
    await tester.pumpAndSettle();
    expect(find.textContaining('A brand new line about the garden'), findsOneWidget, reason: 'Now shows it at once');

    await tab(tester, 'Timeline');
    expect(find.textContaining('A brand new line about the garden'), findsOneWidget, reason: 'caught up when shown');
    await tab(tester, 'Insights');
    expect(find.text('Nothing to show yet'), findsNothing, reason: 'Insights loaded when first shown');
  });

  testWidgets('everyone in the household has a colour of their own', (tester) async {
    s.speakers.create(name: 'Ericah', embeddingModel: 'f', samples: [voiceprint(2)]);
    await tester.pumpWidget(MaterialApp(theme: VoxTheme.light(), home: HomeShell(services: s)));
    await tester.pumpAndSettle();
    expect(speakerColor('Pruitt'), isNot(speakerColor('Ericah')), reason: 'their letters alone pick the same colour');
    final eight = distinctSpeakerColors(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
    expect(eight.values.toSet(), hasLength(8));
  });
}
