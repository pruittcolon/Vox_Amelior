import 'package:flutter/material.dart';

/// Ink-blue and lamplight amber: a reading room at night.
const _seed = Color(0xFF2B3A67);
const amber = Color(0xFFF2B84B);

ThemeData buildTheme(Brightness brightness) {
  final scheme = ColorScheme.fromSeed(
    seedColor: _seed,
    brightness: brightness,
    tertiary: brightness == Brightness.dark ? amber : const Color(0xFF8A5A00),
  );
  return ThemeData(
    colorScheme: scheme,
    useMaterial3: true,
    scaffoldBackgroundColor: scheme.surface,
    appBarTheme: AppBarTheme(
      backgroundColor: scheme.surface,
      surfaceTintColor: Colors.transparent,
      centerTitle: false,
      titleTextStyle: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: scheme.onSurface),
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: scheme.surfaceContainerLow,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: EdgeInsets.zero,
    ),
    chipTheme: ChipThemeData(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
    listTileTheme: const ListTileThemeData(contentPadding: EdgeInsets.symmetric(horizontal: 16)),
    snackBarTheme: const SnackBarThemeData(behavior: SnackBarBehavior.floating),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: scheme.surfaceContainerHighest.withValues(alpha: 0.5),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
    ),
  );
}

/// A distinct, readable colour per speaker ([index] = order of appearance;
/// -1 for the narrator).
Color speakerColor(BuildContext context, int index) {
  final scheme = Theme.of(context).colorScheme;
  if (index < 0) return scheme.onSurfaceVariant;
  const light = [Color(0xFF2558C5), Color(0xFFB0451F), Color(0xFF1D7A55), Color(0xFF7A3FB0), Color(0xFF9A6A00), Color(0xFFB02C66)];
  const night = [Color(0xFF8AB0FF), Color(0xFFFF9E7A), Color(0xFF6FD3A6), Color(0xFFC69CFF), Color(0xFFF2C25B), Color(0xFFFF8DBA)];
  final dark = Theme.of(context).brightness == Brightness.dark;
  return (dark ? night : light)[index % light.length];
}
