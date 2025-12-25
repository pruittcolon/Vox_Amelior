import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// File-based logger for interview sessions.
///
/// Writes all transcripts, questions, and coaching responses to a log file
/// that can be reviewed after the interview.
class InterviewLogger {
  static InterviewLogger? _instance;
  static InterviewLogger get instance => _instance ??= InterviewLogger._();
  
  File? _logFile;
  bool _isInitialized = false;
  String? _sessionId;
  
  InterviewLogger._();

  /// Initialize a new logging session
  Future<void> startSession() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      _sessionId = DateTime.now().toIso8601String().replaceAll(':', '-').split('.')[0];
      final fileName = 'interview_log_$_sessionId.txt';
      _logFile = File('${appDir.path}/$fileName');
      
      await _logFile!.writeAsString(
        '=== INTERVIEW SESSION LOG ===\n'
        'Started: ${DateTime.now()}\n'
        '================================\n\n',
      );
      
      _isInitialized = true;
      _log('📝 InterviewLogger: Session started - ${_logFile!.path}');
    } catch (e) {
      _log('📝 InterviewLogger: Failed to initialize - $e');
      _isInitialized = false;
    }
  }

  /// Log a transcription from the diarizer
  Future<void> logTranscript({
    required int speakerId,
    required String text,
    bool isQuestion = false,
  }) async {
    if (!_isInitialized || _logFile == null) return;
    
    final timestamp = DateTime.now().toIso8601String();
    final speakerLabel = speakerId == 0 ? 'INTERVIEWER' : 'CANDIDATE';
    final questionTag = isQuestion ? ' [QUESTION]' : '';
    
    final entry = '[$timestamp] [$speakerLabel]$questionTag $text\n';
    
    try {
      await _logFile!.writeAsString(entry, mode: FileMode.append);
    } catch (e) {
      _log('📝 InterviewLogger: Write error - $e');
    }
  }

  /// Log a coaching response
  Future<void> logCoachingResponse({
    required String question,
    required String response,
  }) async {
    if (!_isInitialized || _logFile == null) return;
    
    final timestamp = DateTime.now().toIso8601String();
    
    final entry = '''
[$timestamp] [COACH REQUEST]
  Question: $question
  
[$timestamp] [COACH RESPONSE]
  $response

''';
    
    try {
      await _logFile!.writeAsString(entry, mode: FileMode.append);
    } catch (e) {
      _log('📝 InterviewLogger: Write error - $e');
    }
  }

  /// Log a general event
  Future<void> logEvent(String event) async {
    if (!_isInitialized || _logFile == null) return;
    
    final timestamp = DateTime.now().toIso8601String();
    final entry = '[$timestamp] [EVENT] $event\n';
    
    try {
      await _logFile!.writeAsString(entry, mode: FileMode.append);
    } catch (e) {
      _log('📝 InterviewLogger: Write error - $e');
    }
  }

  /// End the logging session
  Future<void> endSession() async {
    if (!_isInitialized || _logFile == null) return;
    
    final timestamp = DateTime.now().toIso8601String();
    final entry = '''

================================
Session ended: $timestamp
================================
''';
    
    try {
      await _logFile!.writeAsString(entry, mode: FileMode.append);
      _log('📝 InterviewLogger: Session ended - ${_logFile!.path}');
    } catch (e) {
      _log('📝 InterviewLogger: End session error - $e');
    }
    
    _isInitialized = false;
    _logFile = null;
    _sessionId = null;
  }

  /// Get the current log file path
  String? get logFilePath => _logFile?.path;
  
  /// Check if logging is active
  bool get isActive => _isInitialized;
}
