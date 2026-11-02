import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import 'app/shell.dart';
import 'app/theme.dart';
import 'core/log.dart';
import 'core/suite.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  _logProblems();
  final suite = Suite.platform();
  runApp(EvenSuiteApp(suite: suite));
  suite.start();
}

/// Errors and UI stalls go into the diagnostics log, so a freeze on the
/// phone can be read back from Termux.
void _logProblems() {
  final previous = FlutterError.onError;
  FlutterError.onError = (details) {
    EventLog.add('app', 'error', {
      'error': details.exceptionAsString().split('\n').first,
      'where': details.library,
      'stack': details.stack.toString().split('\n').take(6).join(' | '),
    });
    previous?.call(details);
  };
  PlatformDispatcher.instance.onError = (error, stack) {
    EventLog.add('app', 'crash', {
      'error': '$error'.split('\n').first,
      'stack': stack.toString().split('\n').take(8).join(' | '),
    });
    return false;
  };
  // Slow UI frames (a stall shows up as a run of these).
  var slow = 0, worst = 0;
  var lastReport = DateTime.now();
  SchedulerBinding.instance.addTimingsCallback((timings) {
    for (final t in timings) {
      final ms = t.totalSpan.inMilliseconds;
      if (ms > 50) {
        slow++;
        if (ms > worst) worst = ms;
      }
    }
    if (slow > 0 && DateTime.now().difference(lastReport) > const Duration(seconds: 5)) {
      EventLog.add('app', 'jank', {'slowFrames': slow, 'worstMs': worst});
      slow = 0;
      worst = 0;
      lastReport = DateTime.now();
    }
  });
}

class EvenSuiteApp extends StatelessWidget {
  const EvenSuiteApp({super.key, required this.suite});

  final Suite suite;

  @override
  Widget build(BuildContext context) {
    return SuiteScope(
      suite: suite,
      child: MaterialApp(
        title: 'Even Suite',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: const Shell(),
      ),
    );
  }
}
