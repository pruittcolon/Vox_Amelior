import 'package:flutter/material.dart';

import 'app/shell.dart';
import 'app/theme.dart';
import 'core/suite.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final suite = Suite.platform();
  runApp(EvenSuiteApp(suite: suite));
  suite.start();
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
