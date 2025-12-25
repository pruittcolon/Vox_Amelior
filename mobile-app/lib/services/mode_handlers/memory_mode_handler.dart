import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/memory_service.dart';
import 'package:dio/dio.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for memory server Q&A sessions.
///
/// This mode uses WhisperServer instead of Deepgram for transcription.
class MemoryModeHandler extends ModeHandler with TerminateDetection, WordMatching {
  final MemoryService _memoryService;

  bool _isActive = false;
  String _sessionId = '';

  /// Wake word to trigger memory mode
  static const String wakeWord = 'memory';

  MemoryModeHandler({MemoryService? memoryService})
      : _memoryService = memoryService ?? MemoryService.instance;

  @override
  String get modeName => 'memory';

  @override
  bool get isActive => _isActive;

  /// Current session ID for context preservation
  String get sessionId => _sessionId;

  @override
  bool canEnterMode(String transcript) {
    return transcript.toLowerCase().contains(wakeWord);
  }

  @override
  bool canHandleInMode(String transcript) {
    return _isActive;
  }

  @override
  bool isTerminateCommand(String transcript) {
    final lower = transcript.toLowerCase();
    return matchesTerminate(lower) || lower.contains('end memory');
  }

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _log("🧠 MemoryModeHandler: Entering Memory Mode.");
    _isActive = true;
    _sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _memoryService.setActiveSessionId(_sessionId);

    // Check if there's a question following the wake word
    final remainder = _stripWakeWordAndRemainder(transcript);

    if (remainder.isNotEmpty) {
      // Immediate question following the wake word
      _log("🧠 MemoryModeHandler: Processing immediate question: '$remainder'");
      return await _handleQuery(remainder);
    }

    // Just the wake word - prompt for question
    return const ModeResult(
      continueListening: true,
      useWhisperServer: true,
      displayText: "Memory Mode Started:\nAsk your question.\nSay 'End Memory' to exit.\n\nContext: Will remember conversation history.",
    );
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // Check for termination
    if (isTerminateCommand(lower)) {
      _log("🧠 MemoryModeHandler: User terminated memory mode.");
      reset();
      return ModeResult.endSession;
    }

    // Handle the query
    if (lower.isNotEmpty) {
      return await _handleQuery(transcript);
    }

    // Empty transcript - keep listening
    _log("🧠 MemoryModeHandler: No speech detected, continuing to listen.");
    return const ModeResult(
      continueListening: true,
      useWhisperServer: true,
    );
  }

  Future<ModeResult> _handleQuery(String question) async {
    try {
      _log("🧠 MemoryModeHandler: Submitting question: '$question'");

      // Submit the question to the memory server
      final response = await _memoryService.askQuestion(
        question,
        sessionId: _sessionId,
      );

      final answer = response.answer.trim().isEmpty
          ? "No answer yet. Try again later."
          : response.answer.trim();

      _log("🧠 MemoryModeHandler: Received answer.");

      // Continue in memory mode for follow-up questions
      return ModeResult(
        continueListening: true,
        useWhisperServer: true,
        displayText: "$answer\n\nAsk another question or say 'End Memory' to exit.",
      );
    } catch (e) {
      String reason = e.toString();
      if (e is DioException) {
        final se = e.error;
        if (se is SocketException) {
          reason = "Socket error: ${se.osError?.errorCode ?? ''} ${se.osError?.message ?? se.message}";
        } else if (e.type == DioExceptionType.connectionError) {
          reason = "Connection error (server unreachable)";
        }
      }

      _log("🧠 MemoryModeHandler: Error - $reason");

      return ModeResult(
        continueListening: true,
        useWhisperServer: true,
        displayText: "Memory server error.\n($reason)\n\nTry again or say 'End Memory' to exit.",
      );
    }
  }

  /// Strip the wake word from the given text and return the remainder.
  String _stripWakeWordAndRemainder(String text) {
    final t = text.trim().toLowerCase();
    final idx = t.indexOf(wakeWord);
    if (idx == -1) return "";
    final after = t.substring(idx + wakeWord.length).trimLeft();
    // Remove leading punctuation and whitespace
    return after.replaceFirst(RegExp(r'^[,.:;\-\s]+'), '').trim();
  }

  @override
  void reset() {
    _log("🧠 MemoryModeHandler: Resetting mode.");
    _isActive = false;
    _sessionId = '';
  }
}
