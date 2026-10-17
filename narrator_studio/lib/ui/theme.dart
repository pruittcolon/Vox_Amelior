import 'package:flutter/material.dart';

/// A recording studio at night: warm paper and ink by day, deep navy with
/// lamplight amber after dark. Inter for the interface, Fraunces for titles,
/// Literata for the words being read.
const uiFont = 'Inter';
const displayFont = 'Fraunces';
const readingFont = 'Literata';

const amber = Color(0xFFF5B94F);

ColorScheme _scheme(Brightness b) {
  final base = ColorScheme.fromSeed(seedColor: const Color(0xFF4338CA), brightness: b);
  if (b == Brightness.light) {
    return base.copyWith(
      primary: const Color(0xFF3F36C2),
      onPrimary: Colors.white,
      primaryContainer: const Color(0xFFE4E2FC),
      onPrimaryContainer: const Color(0xFF1D1A5C),
      secondary: const Color(0xFFA35F00),
      onSecondary: Colors.white,
      secondaryContainer: const Color(0xFFFBE6C2),
      onSecondaryContainer: const Color(0xFF3D2400),
      tertiary: const Color(0xFFB8430E),
      tertiaryContainer: const Color(0xFFFFE1D2),
      onTertiaryContainer: const Color(0xFF3E1300),
      surface: const Color(0xFFF8F4ED),
      onSurface: const Color(0xFF1C1A22),
      onSurfaceVariant: const Color(0xFF5E5A66),
      surfaceContainerLowest: Colors.white,
      surfaceContainerLow: const Color(0xFFFFFCF7),
      surfaceContainer: const Color(0xFFF2ECE2),
      surfaceContainerHigh: const Color(0xFFEBE4D8),
      surfaceContainerHighest: const Color(0xFFE3DBCD),
      outline: const Color(0xFF8E8798),
      outlineVariant: const Color(0xFFE2DACD),
    );
  }
  return base.copyWith(
    primary: const Color(0xFFB3B7FF),
    onPrimary: const Color(0xFF16195C),
    primaryContainer: const Color(0xFF30348A),
    onPrimaryContainer: const Color(0xFFE1E1FF),
    secondary: amber,
    onSecondary: const Color(0xFF3B2600),
    secondaryContainer: const Color(0xFF573C06),
    onSecondaryContainer: const Color(0xFFFFE2AF),
    tertiary: const Color(0xFFFFA184),
    tertiaryContainer: const Color(0xFF6B2A12),
    onTertiaryContainer: const Color(0xFFFFDBCF),
    surface: const Color(0xFF0D1020),
    onSurface: const Color(0xFFECEAF4),
    onSurfaceVariant: const Color(0xFFB4B6CA),
    surfaceContainerLowest: const Color(0xFF090B17),
    surfaceContainerLow: const Color(0xFF141830),
    surfaceContainer: const Color(0xFF191E38),
    surfaceContainerHigh: const Color(0xFF212742),
    surfaceContainerHighest: const Color(0xFF29304F),
    outline: const Color(0xFF6E7391),
    outlineVariant: const Color(0xFF2A3050),
  );
}

TextTheme _text(ColorScheme s) {
  TextStyle ui(double size, FontWeight w, {double h = 1.35, double ls = 0}) =>
      TextStyle(fontFamily: uiFont, fontSize: size, fontWeight: w, height: h, letterSpacing: ls, color: s.onSurface);
  TextStyle display(double size, {double h = 1.15, double ls = -0.3}) =>
      TextStyle(fontFamily: displayFont, fontSize: size, fontWeight: FontWeight.w600, height: h, letterSpacing: ls, color: s.onSurface);
  return TextTheme(
    displayLarge: display(52, ls: -1),
    displayMedium: display(42, ls: -0.8),
    displaySmall: display(34, ls: -0.6),
    headlineLarge: display(30, ls: -0.5),
    headlineMedium: display(26, ls: -0.4),
    headlineSmall: display(23),
    titleLarge: display(20, h: 1.25, ls: -0.1),
    titleMedium: ui(16, FontWeight.w600, ls: -0.1),
    titleSmall: ui(14, FontWeight.w600),
    bodyLarge: ui(16, FontWeight.w400, h: 1.5),
    bodyMedium: ui(14, FontWeight.w400, h: 1.45),
    bodySmall: ui(12.5, FontWeight.w400, h: 1.4).copyWith(color: s.onSurfaceVariant),
    labelLarge: ui(14, FontWeight.w600, ls: 0.1),
    labelMedium: ui(12, FontWeight.w600, ls: 0.2),
    labelSmall: ui(11, FontWeight.w600, ls: 0.4),
  );
}

ThemeData buildTheme(Brightness brightness) {
  final s = _scheme(brightness);
  final text = _text(s);
  final shape16 = RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
  return ThemeData(
    colorScheme: s,
    useMaterial3: true,
    fontFamily: uiFont,
    textTheme: text,
    scaffoldBackgroundColor: s.surface,
    appBarTheme: AppBarTheme(
      backgroundColor: s.surface,
      surfaceTintColor: Colors.transparent,
      scrolledUnderElevation: 0,
      centerTitle: false,
      titleTextStyle: text.headlineSmall,
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: s.surfaceContainerLow,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: BorderSide(color: s.outlineVariant.withValues(alpha: brightness == Brightness.light ? 1 : 0.7)),
      ),
      margin: EdgeInsets.zero,
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        shape: shape16,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        textStyle: text.labelLarge,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        shape: shape16,
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
        side: BorderSide(color: s.outlineVariant),
        textStyle: text.labelLarge,
      ),
    ),
    textButtonTheme: TextButtonThemeData(style: TextButton.styleFrom(textStyle: text.labelLarge, shape: shape16)),
    floatingActionButtonTheme: FloatingActionButtonThemeData(
      backgroundColor: s.primary,
      foregroundColor: s.onPrimary,
      elevation: 2,
      highlightElevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      extendedTextStyle: text.labelLarge,
    ),
    navigationBarTheme: NavigationBarThemeData(
      height: 68,
      backgroundColor: s.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      indicatorColor: s.secondaryContainer,
      labelTextStyle: WidgetStateProperty.resolveWith(
        (st) => text.labelMedium!.copyWith(color: st.contains(WidgetState.selected) ? s.onSurface : s.onSurfaceVariant),
      ),
      iconTheme: WidgetStateProperty.resolveWith(
        (st) => IconThemeData(color: st.contains(WidgetState.selected) ? s.onSecondaryContainer : s.onSurfaceVariant),
      ),
    ),
    chipTheme: ChipThemeData(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      side: BorderSide(color: s.outlineVariant),
      labelStyle: text.labelMedium,
    ),
    sliderTheme: SliderThemeData(
      trackHeight: 4,
      activeTrackColor: s.primary,
      inactiveTrackColor: s.primary.withValues(alpha: 0.16),
      thumbColor: s.primary,
      overlayColor: s.primary.withValues(alpha: 0.12),
      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: s.primary,
      linearTrackColor: s.primary.withValues(alpha: 0.14),
      linearMinHeight: 4,
    ),
    listTileTheme: ListTileThemeData(
      contentPadding: const EdgeInsets.symmetric(horizontal: 16),
      titleTextStyle: text.titleMedium,
      subtitleTextStyle: text.bodySmall,
      iconColor: s.onSurfaceVariant,
    ),
    switchTheme: SwitchThemeData(
      thumbIcon: WidgetStateProperty.resolveWith((st) => st.contains(WidgetState.selected) ? const Icon(Icons.check, size: 14) : null),
    ),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: s.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: s.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      titleTextStyle: text.titleLarge,
    ),
    popupMenuTheme: PopupMenuThemeData(
      color: s.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      textStyle: text.bodyMedium,
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
    ),
    dividerTheme: DividerThemeData(color: s.outlineVariant, space: 24),
    expansionTileTheme: const ExpansionTileThemeData(shape: Border(), collapsedShape: Border()),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: s.surfaceContainer,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
      focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide(color: s.primary, width: 1.5)),
    ),
  );
}

/// A distinct, readable colour per speaker ([index] = order of appearance;
/// -1 for the narrator).
Color speakerColor(BuildContext context, int index) {
  final scheme = Theme.of(context).colorScheme;
  if (index < 0) return scheme.onSurfaceVariant;
  const light = [Color(0xFF3F36C2), Color(0xFFB8430E), Color(0xFF16775A), Color(0xFF8A3FB8), Color(0xFF9A6200), Color(0xFFB0285F)];
  const night = [Color(0xFFB3B7FF), Color(0xFFFFA184), Color(0xFF6FDDB0), Color(0xFFD2A6FF), Color(0xFFF5C46B), Color(0xFFFF8DBA)];
  final dark = Theme.of(context).brightness == Brightness.dark;
  return (dark ? night : light)[index % light.length];
}

/// The sentence being read: warm lamplight, readable in both themes.
(Color, Color) readingHighlight(BuildContext context) => Theme.of(context).brightness == Brightness.dark
    ? (const Color(0xFF5A4413), const Color(0xFFFFF3D6))
    : (const Color(0xFFFFE3A0), const Color(0xFF2A2000));
