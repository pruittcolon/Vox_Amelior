import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/audio/exporter.dart';
import '../core/audio/native.dart';
import '../core/audio/player.dart';
import 'preview.dart';

/// Gives every screen the app's services.
class AppScope extends InheritedWidget {
  const AppScope({
    super.key,
    required this.app,
    required this.player,
    required this.exporter,
    required this.native,
    required this.preview,
    required super.child,
  });

  final AppState app;
  final Player player;
  final Exporter exporter;
  final NativeApi native;
  final VoicePreview preview;

  // The services never change, so no dependency is registered (safe in initState).
  static AppScope of(BuildContext context) => context.getInheritedWidgetOfExactType<AppScope>()!;

  @override
  bool updateShouldNotify(AppScope oldWidget) => false;
}

extension ScopeX on BuildContext {
  AppScope get scope => AppScope.of(this);
  AppState get app => AppScope.of(this).app;

  void toast(String message, {SnackBarAction? action}) {
    ScaffoldMessenger.of(this)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message), action: action, behavior: SnackBarBehavior.floating));
  }
}

String formatDuration(Duration d) {
  final h = d.inHours;
  final m = d.inMinutes % 60;
  if (h > 0) return '${h}h ${m}m';
  if (d.inMinutes > 0) return '${d.inMinutes} min';
  return '${d.inSeconds} s';
}
