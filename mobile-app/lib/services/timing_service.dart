import 'dart:async';
import 'package:flutter/foundation.dart';

/// Performance timing service for precise latency measurements.
///
/// Provides infrastructure for tracking timing of various operations
/// in the voice-to-glasses pipeline. All timings are logged to console
/// and broadcast via stream for UI integration.
class TimingService {
  static final TimingService _instance = TimingService._();
  static TimingService get instance => _instance;

  TimingService._();

  /// Active timers with start timestamps
  final Map<String, int> _activeTimers = {};

  /// Baseline timestamp for relative timing
  int? _pipelineStartMs;

  /// Stream controller for timing events
  final StreamController<TimingEvent> _eventController =
      StreamController<TimingEvent>.broadcast();

  /// Stream of timing events for UI integration
  Stream<TimingEvent> get events => _eventController.stream;

  /// Start a new pipeline measurement (resets baseline)
  void startPipeline() {
    _pipelineStartMs = DateTime.now().millisecondsSinceEpoch;
    _activeTimers.clear();
    _logEvent('pipeline_started', 0, 'Pipeline baseline set');
  }

  /// Start a named timer
  void startTimer(String label) {
    final now = DateTime.now().millisecondsSinceEpoch;
    _activeTimers[label] = now;
    
    final relativeMs = _pipelineStartMs != null ? now - _pipelineStartMs! : 0;
    _logEvent('${label}_started', relativeMs, 'Timer started');
  }

  /// Stop a named timer and return elapsed milliseconds
  int stopTimer(String label) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final startTime = _activeTimers.remove(label);
    
    if (startTime == null) {
      if (kDebugMode) {
        debugPrint('⏱️ [TIMING] WARNING: No timer found for "$label"');
      }
      return 0;
    }

    final elapsed = now - startTime;
    final relativeMs = _pipelineStartMs != null ? now - _pipelineStartMs! : elapsed;
    
    _logEvent('${label}_complete', relativeMs, '+${elapsed}ms');
    return elapsed;
  }

  /// Log a milestone event (point-in-time)
  void logMilestone(String label, [String? detail]) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final relativeMs = _pipelineStartMs != null ? now - _pipelineStartMs! : 0;
    _logEvent(label, relativeMs, detail);
  }

  /// End pipeline and log total time
  int endPipeline() {
    if (_pipelineStartMs == null) {
      if (kDebugMode) {
        debugPrint('⏱️ [TIMING] WARNING: No pipeline started');
      }
      return 0;
    }

    final now = DateTime.now().millisecondsSinceEpoch;
    final totalMs = now - _pipelineStartMs!;
    
    if (kDebugMode) {
      debugPrint('');
      debugPrint('⏱️ ════════════════════════════════════════════');
      debugPrint('⏱️ [TIMING] === TOTAL PIPELINE: ${_formatMs(totalMs)} ===');
      debugPrint('⏱️ ════════════════════════════════════════════');
      debugPrint('');
    }

    _eventController.add(TimingEvent(
      label: 'pipeline_complete',
      relativeMs: totalMs,
      detail: 'Total: ${_formatMs(totalMs)}',
      isPipelineEnd: true,
    ));

    _pipelineStartMs = null;
    _activeTimers.clear();
    return totalMs;
  }

  /// Internal logging helper
  void _logEvent(String label, int relativeMs, [String? detail]) {
    final timestamp = DateTime.now().toIso8601String();
    final detailStr = detail != null ? ' ($detail)' : '';
    
    if (kDebugMode) {
      debugPrint('⏱️ [TIMING] [$timestamp] $label: ${_formatMs(relativeMs)}$detailStr');
    }

    if (!_eventController.isClosed) {
      _eventController.add(TimingEvent(
        label: label,
        relativeMs: relativeMs,
        detail: detail,
      ));
    }
  }

  /// Format milliseconds for display
  String _formatMs(int ms) {
    if (ms >= 1000) {
      return '${(ms / 1000).toStringAsFixed(2)}s';
    }
    return '${ms}ms';
  }

  void dispose() {
    _eventController.close();
  }
}

/// Timing event for stream consumers
class TimingEvent {
  final String label;
  final int relativeMs;
  final String? detail;
  final bool isPipelineEnd;
  final DateTime timestamp;

  TimingEvent({
    required this.label,
    required this.relativeMs,
    this.detail,
    this.isPipelineEnd = false,
  }) : timestamp = DateTime.now();

  @override
  String toString() => '[$label] ${relativeMs}ms${detail != null ? " ($detail)" : ""}';
}
