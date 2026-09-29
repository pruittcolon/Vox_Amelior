import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';

/// Look-and-feel choices from Settings → Appearance.
class Appearance {
  const Appearance({this.accent = VoxTheme.seed, this.corners = 'rounded', this.textScale = 1.0, this.mode = 'system'});

  final Color accent;

  /// 'square' | 'soft' | 'rounded'
  final String corners;
  final double textScale;

  /// 'system' | 'light' | 'dark'
  final String mode;

  ThemeMode get themeMode => switch (mode) {
        'light' => ThemeMode.light,
        'dark' => ThemeMode.dark,
        _ => ThemeMode.system,
      };

  /// Base corner radius for cards; other shapes scale from it.
  double get radius => switch (corners) {
        'square' => 6,
        'soft' => 14,
        _ => 22,
      };
}

/// The appearance chosen in [s].
Appearance appearanceOf(AppSettings s) =>
    Appearance(accent: Color(s.accent), corners: s.corners, textScale: s.textScale, mode: s.themeMode);

/// Accent colours offered in Settings → Appearance.
const List<(String, Color)> kAccentChoices = [
  ('Indigo', Color(0xFF4F46E5)),
  ('Ocean', Color(0xFF0B7285)),
  ('Forest', Color(0xFF2B8A3E)),
  ('Sunset', Color(0xFFE8590C)),
  ('Rose', Color(0xFFC2255C)),
  ('Grape', Color(0xFF862E9C)),
  ('Sky', Color(0xFF1971C2)),
  ('Graphite', Color(0xFF495057)),
];

/// Vox's visual style: calm colour, soft surfaces, rounded cards.
abstract final class VoxTheme {
  static const Color seed = Color(0xFF4F46E5);

  static ThemeData light([Appearance a = const Appearance()]) => build(Brightness.light, a);
  static ThemeData dark([Appearance a = const Appearance()]) => build(Brightness.dark, a);

  static ThemeData build(Brightness brightness, Appearance a) {
    final scheme = ColorScheme.fromSeed(seedColor: a.accent, brightness: brightness);
    final base = ThemeData(useMaterial3: true, colorScheme: scheme, brightness: brightness);
    final r = a.radius;
    final small = (r * 0.72).roundToDouble();
    return base.copyWith(
      scaffoldBackgroundColor: scheme.surface,
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.surface,
        surfaceTintColor: Colors.transparent,
        centerTitle: false,
        titleTextStyle: base.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700, color: scheme.onSurface),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: scheme.surfaceContainerLow,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(r)),
        clipBehavior: Clip.antiAlias,
      ),
      listTileTheme: const ListTileThemeData(contentPadding: EdgeInsets.symmetric(horizontal: 16)),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: scheme.surfaceContainerHigh,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(small), borderSide: BorderSide.none),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      chipTheme: base.chipTheme.copyWith(
        shape: a.corners == 'square' ? RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)) : const StadiumBorder(),
        side: BorderSide.none,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size(0, 52),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(small)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size(0, 48),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(small)),
        ),
      ),
      segmentedButtonTheme: SegmentedButtonThemeData(
        style: SegmentedButton.styleFrom(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(small)),
          selectedBackgroundColor: scheme.primaryContainer,
          selectedForegroundColor: scheme.onPrimaryContainer,
        ),
      ),
      bottomSheetTheme: BottomSheetThemeData(
        showDragHandle: true,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(r + 6))),
      ),
      dialogTheme: DialogThemeData(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(r + 4))),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: scheme.surfaceContainer,
        indicatorColor: scheme.primaryContainer,
        labelTextStyle: WidgetStatePropertyAll(base.textTheme.labelMedium?.copyWith(fontWeight: FontWeight.w600)),
      ),
      snackBarTheme: const SnackBarThemeData(behavior: SnackBarBehavior.floating),
      dividerTheme: DividerThemeData(color: scheme.outlineVariant.withValues(alpha: 0.5), space: 1),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        linearTrackColor: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
    );
  }
}
