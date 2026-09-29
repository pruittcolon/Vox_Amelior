import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/app/assistant_inbox.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/ui/home_shell.dart';
import 'package:vox_amelior_mobile/ui/setup_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FlutterGemma.initialize(inferenceEngines: const [LiteRtLmEngine()]);
  runApp(const VoxApp());
}

class VoxApp extends StatefulWidget {
  const VoxApp({super.key});

  @override
  State<VoxApp> createState() => _VoxAppState();
}

class _VoxAppState extends State<VoxApp> with WidgetsBindingObserver {
  late final Future<AppServices> _services = AppServices.create();
  AssistantInbox? _inbox;
  bool _setupDone = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _services.then((s) {
      _inbox = AssistantInbox(s)..start();
      if (mounted) setState(() => _setupDone = s.speechReady);
    }).catchError((Object e, StackTrace st) => Log.e('app', 'startup failed', e, st));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_inbox?.dispose());
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_inbox?.processPending());
      _services.then((s) => s.listening.refresh());
    }
  }

  @override
  Widget build(BuildContext context) {
    final seed = Colors.teal;
    return MaterialApp(
      title: 'Vox',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: seed, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark, useMaterial3: true),
      // Keeps the app responsive to the foreground service's notification.
      home: WithForegroundTask(
        child: FutureBuilder<AppServices>(
          future: _services,
          builder: (context, snap) {
            if (snap.hasError) {
              return Scaffold(body: Center(child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Vox could not start: ${snap.error}'),
              )));
            }
            final services = snap.data;
            if (services == null) return const Scaffold(body: Center(child: CircularProgressIndicator()));
            if (!_setupDone) {
              return SetupScreen(services: services, onDone: () => setState(() => _setupDone = true));
            }
            return HomeShell(services: services);
          },
        ),
      ),
    );
  }
}
