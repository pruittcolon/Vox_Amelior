import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/native/gemma_llm_engine.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/home_shell.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';

Future<void> main() async {
  await prepareApp();
  runApp(const VoxApp());
}

/// What the app needs before its first frame (also run by the on-device
/// tour test before it starts the app).
Future<void> prepareApp() async {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterForegroundTask.initCommunicationPort();
  try {
    await GemmaLlmEngine.initializePlugin().timeout(const Duration(seconds: 15));
  } on Object catch (e, st) {
    // The assistant is optional; the rest of the app must still start.
    Log.e('app', 'assistant engine unavailable', e, st);
  }
}

class VoxApp extends StatefulWidget {
  const VoxApp({super.key, this.services});

  /// The app's services; created here when not given (the tour test
  /// creates them itself, to reach into the app while it drives it).
  final Future<AppServices>? services;

  @override
  State<VoxApp> createState() => _VoxAppState();
}

class _VoxAppState extends State<VoxApp> with WidgetsBindingObserver {
  late final Future<AppServices> _services = widget.services ?? AppServices.create();
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
    return FutureBuilder<AppServices>(
      future: _services,
      builder: (context, snap) {
        final services = snap.data;
        if (services == null) return _app(const Appearance(), _loading(snap.error));
        return ValueListenableBuilder<AppSettings>(
          valueListenable: services.settings,
          builder: (context, settings, _) => _app(
            appearanceOf(settings),
            _setupDone ? HomeShell(services: services) : ModelsScreen(services: services, onDone: () => setState(() => _setupDone = true)),
          ),
        );
      },
    );
  }

  Widget _app(Appearance look, Widget home) => MaterialApp(
        title: 'Vox',
        debugShowCheckedModeBanner: false,
        theme: VoxTheme.light(look),
        darkTheme: VoxTheme.dark(look),
        themeMode: look.themeMode,
        builder: (context, child) {
          final mq = MediaQuery.of(context);
          return MediaQuery(
            data: mq.copyWith(textScaler: TextScaler.linear(mq.textScaler.scale(1) * look.textScale)),
            child: child!,
          );
        },
        home: WithForegroundTask(child: home),
      );

  Widget _loading(Object? error) => Scaffold(
        body: Center(
          child: error == null
              ? const CircularProgressIndicator()
              : Padding(padding: const EdgeInsets.all(24), child: Text('Vox could not start: $error')),
        ),
      );
}
