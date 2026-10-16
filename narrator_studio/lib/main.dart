import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'core/app_state.dart';
import 'core/audio/exporter.dart';
import 'core/audio/native.dart';
import 'core/audio/player.dart';
import 'core/library/importer.dart';
import 'core/tts/sherpa_synth.dart';
import 'ui/home_shell.dart';
import 'ui/preview.dart';
import 'ui/reader_screen.dart';
import 'ui/scope.dart';
import 'ui/theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final support = await getApplicationSupportDirectory();
  final documents = await getApplicationDocumentsDirectory();
  final native = AndroidNative();
  late final AppState app;
  app = AppState(
    supportDir: support,
    documentsDir: documents,
    native: native,
    synth: (a) => SherpaSynth(
      () => EngineSetup(
        kokoroDir: a.models.packDir(a.kokoroPack).path,
        supertonicDir: a.models.packDir('supertonic').path,
        pocketDir: a.models.packDir('pocket').path,
        kokoroVoicesFile: a.kokoroVoicesFile,
        threads: a.settings.threads,
      ),
    ),
  );
  await app.init();
  final player = Player(app, native);
  final exporter = Exporter(app, native);
  final preview = VoicePreview(app, native, player);
  runApp(NarratorApp(app: app, player: player, exporter: exporter, native: native, preview: preview));
}

class NarratorApp extends StatefulWidget {
  const NarratorApp({
    super.key,
    required this.app,
    required this.player,
    required this.exporter,
    required this.native,
    required this.preview,
  });

  final AppState app;
  final Player player;
  final Exporter exporter;
  final NativeApi native;
  final VoicePreview preview;

  @override
  State<NarratorApp> createState() => _NarratorAppState();
}

class _NarratorAppState extends State<NarratorApp> {
  final _nav = GlobalKey<NavigatorState>();
  final _messenger = GlobalKey<ScaffoldMessengerState>();
  StreamSubscription<Map<String, Object?>>? _shares;

  @override
  void initState() {
    super.initState();
    _shares = widget.native.shares.listen(_receive);
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        final pending = await widget.native.pendingShare();
        if (pending != null) await _receive(pending);
      } catch (_) {}
    });
  }

  @override
  void dispose() {
    _shares?.cancel();
    super.dispose();
  }

  /// Text or a file shared from another app (e.g. a script from Claude).
  Future<void> _receive(Map<String, Object?> share) async {
    final app = widget.app;
    try {
      if (share['error'] != null) throw ImportError(share['error'].toString());
      final text = share['text'] as String?;
      final path = share['path'] as String?;
      final doc = text != null
          ? await app.library.add(
              title: (share['title'] as String?) ?? _firstLine(text),
              kind: 'shared',
              text: text,
            )
          : await () async {
              final imported = await importFile(path!, displayName: share['name'] as String?);
              return app.library.add(title: imported.title, kind: imported.kind, text: imported.text);
            }();
      await _nav.currentState?.push(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: doc)));
    } catch (e) {
      _messenger.currentState?.showSnackBar(SnackBar(content: Text('Could not open what was shared: $e')));
    }
  }

  String _firstLine(String text) {
    final m = RegExp(r'^\s*(?:title:\s*)?#*\s*(.+)$', multiLine: true).firstMatch(text);
    final line = (m?[1] ?? 'Shared text').trim();
    return line == '---' ? 'Shared script' : (line.length > 60 ? '${line.substring(0, 60)}…' : line);
  }

  @override
  Widget build(BuildContext context) {
    return AppScope(
      app: widget.app,
      player: widget.player,
      exporter: widget.exporter,
      native: widget.native,
      preview: widget.preview,
      child: MaterialApp(
        title: 'Narrator Studio',
        debugShowCheckedModeBanner: false,
        navigatorKey: _nav,
        scaffoldMessengerKey: _messenger,
        theme: buildTheme(Brightness.light),
        darkTheme: buildTheme(Brightness.dark),
        home: const HomeShell(),
      ),
    );
  }
}

/// For tests and tools: true when running on a device (not `flutter test`).
bool get onDevice => Platform.isAndroid;
