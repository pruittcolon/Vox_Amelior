import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/native/gemma_llm_engine.dart';
import 'package:vox_amelior_mobile/ui/home_shell.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterForegroundTask.initCommunicationPort();
  try {
    await GemmaLlmEngine.initializePlugin().timeout(const Duration(seconds: 15));
  } on Object catch (e, st) {
    // The assistant is optional; the rest of the app must still start.
    Log.e('app', 'assistant engine unavailable', e, st);
  }
  runApp(const VoxApp());
}

class VoxApp extends StatefulWidget {
  const VoxApp({super.key});

  @override
  State<VoxApp> createState() => _VoxAppState();
}

class _VoxAppState extends State<VoxApp> with WidgetsBindingObserver {
  late final Future<AppServices> _services = AppServices.create();
  bool _setupDone = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _services.then((s) {
      if (mounted) setState(() => _setupDone = s.speechReady);
    }).catchError((Object e, StackTrace st) => Log.e('app', 'startup failed', e, st));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_services.then((s) async {
        await s.listening.refresh();
        s.dataVersion.value++;
      }));
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vox',
      debugShowCheckedModeBanner: false,
      theme: VoxTheme.light(),
      darkTheme: VoxTheme.dark(),
      home: WithForegroundTask(
        child: FutureBuilder<AppServices>(
          future: _services,
          builder: (context, snap) {
            if (snap.hasError) {
              return Scaffold(
                body: Center(
                  child: Padding(padding: const EdgeInsets.all(24), child: Text('Vox could not start: ${snap.error}')),
                ),
              );
            }
            final services = snap.data;
            if (services == null) return const Scaffold(body: Center(child: CircularProgressIndicator()));
            if (!_setupDone) {
              return ModelsScreen(services: services, onDone: () => setState(() => _setupDone = true));
            }
            return HomeShell(services: services);
          },
        ),
      ),
    );
  }
}
