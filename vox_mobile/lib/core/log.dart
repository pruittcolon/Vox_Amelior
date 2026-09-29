import 'package:flutter/foundation.dart';

/// Minimal logger. Never pass transcript text or other personal data here:
/// logs can end up in bug reports.
class Log {
  const Log._();

  static void i(String tag, String message) => debugPrint('[$tag] $message');

  static void w(String tag, String message, [Object? error]) =>
      debugPrint('[$tag] WARN $message${error == null ? '' : ': $error'}');

  static void e(String tag, String message, Object error, [StackTrace? stack]) {
    debugPrint('[$tag] ERROR $message: $error');
    if (stack != null && kDebugMode) debugPrint('$stack');
  }
}
