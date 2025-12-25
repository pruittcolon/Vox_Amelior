import 'dart:async';

import 'package:flutter/foundation.dart';

class LogEntry {
  final DateTime timestamp;
  final String tag;
  final String message;
  final bool isError;

  LogEntry({
    required this.timestamp,
    required this.tag,
    required this.message,
    required this.isError,
  });

  /// Format as ISO timestamp with milliseconds
  String get formattedTimestamp => timestamp.toIso8601String();

  @override
  String toString() => '[$formattedTimestamp] [$tag] $message';
}

class AppLogger {
  AppLogger._();

  static final AppLogger instance = AppLogger._();

  final StreamController<LogEntry> _controller =
      StreamController<LogEntry>.broadcast();

  Stream<LogEntry> get stream => _controller.stream;

  void log(String tag, String message, {bool isError = false}) {
    final entry = LogEntry(
      timestamp: DateTime.now(),
      tag: tag,
      message: message,
      isError: isError,
    );
    
    // Actually output the log with proper formatting
    final prefix = isError ? '❌' : '📋';
    debugPrint('$prefix [${entry.formattedTimestamp}] [$tag] $message');
    
    if (!_controller.isClosed) {
      _controller.add(entry);
    }
  }

  /// Log with timing category
  void logTiming(String tag, String message, int durationMs) {
    final entry = LogEntry(
      timestamp: DateTime.now(),
      tag: 'TIMING:$tag',
      message: '$message (${durationMs}ms)',
      isError: false,
    );
    
    debugPrint('⏱️ [${entry.formattedTimestamp}] [$tag] $message (${durationMs}ms)');
    
    if (!_controller.isClosed) {
      _controller.add(entry);
    }
  }

  void dispose() {
    _controller.close();
  }
}
