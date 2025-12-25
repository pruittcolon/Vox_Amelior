/// Service for maintaining a buffer of recent transcripts.
import 'package:flutter/foundation.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}
///
/// Used by RobotModeHandler to provide conversation context to ChatGPT
/// when the user says "ask the robot".
class TranscriptBufferService {
  static TranscriptBufferService? _instance;
  static TranscriptBufferService get instance =>
      _instance ??= TranscriptBufferService._();

  /// Maximum number of transcripts to keep in buffer
  static const int maxBufferSize = 20;

  /// Circular buffer of recent transcripts
  final List<TranscriptEntry> _buffer = [];

  TranscriptBufferService._();

  /// Add a new transcript to the buffer.
  ///
  /// If buffer is full, oldest entry is removed.
  void addTranscript(String text, {String speaker = 'user'}) {
    if (text.trim().isEmpty) return;

    final entry = TranscriptEntry(
      text: text.trim(),
      speaker: speaker,
      timestamp: DateTime.now(),
    );

    _buffer.add(entry);

    // Keep only the last maxBufferSize entries
    while (_buffer.length > maxBufferSize) {
      _buffer.removeAt(0);
    }

    _log("📝 TranscriptBuffer: Added entry (${_buffer.length}/$maxBufferSize)");
  }

  /// Get the last N transcripts (or all if less than N available).
  List<TranscriptEntry> getRecentTranscripts([int count = maxBufferSize]) {
    if (_buffer.isEmpty) return [];
    
    final startIdx = _buffer.length > count ? _buffer.length - count : 0;
    return _buffer.sublist(startIdx);
  }

  /// Get transcripts formatted as a conversation string for GPT.
  /// Marks undiarized entries as "[Speaker TBD]" to indicate pending diarization.
  String getFormattedConversation([int count = maxBufferSize]) {
    final transcripts = getRecentTranscripts(count);
    if (transcripts.isEmpty) {
      return "[No recent conversation]";
    }

    final buffer = StringBuffer();
    for (final entry in transcripts) {
      final timeStr = _formatTime(entry.timestamp);
      // Mark entries that haven't been diarized yet
      final speaker = (entry.speaker == 'UNKNOWN' || entry.speaker == 'user')
          ? '[Speaker TBD]'
          : entry.speaker;
      buffer.writeln("[$timeStr] $speaker: ${entry.text}");
    }
    return buffer.toString().trim();
  }

  /// Format timestamp for display
  String _formatTime(DateTime dt) {
    return "${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
  }

  /// Get the number of entries currently in buffer
  int get length => _buffer.length;

  /// Check if buffer has any entries
  bool get isEmpty => _buffer.isEmpty;

  /// Clear the buffer
  void clear() {
    _buffer.clear();
    _log("📝 TranscriptBuffer: Cleared");
  }
}

/// A single transcript entry in the buffer
class TranscriptEntry {
  final String text;
  final String speaker;
  final DateTime timestamp;

  TranscriptEntry({
    required this.text,
    required this.speaker,
    required this.timestamp,
  });

  @override
  String toString() => "[$speaker] $text";
}
