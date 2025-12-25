import 'dart:async';
import 'package:flutter/foundation.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for managing conversation memory in interview mode.
///
/// Maintains a rolling window of messages for GPT context, handling
/// token limits and providing formatted history for API calls.
class ConversationMemoryService {
  /// Maximum approximate tokens to keep in context
  static const int maxContextTokens = 6000;
  
  /// Approximate chars per token for estimation
  static const double charsPerToken = 4.0;
  
  /// Message history for GPT
  final List<ChatMessage> _messages = [];
  
  /// System prompt (set once)
  String? _systemPrompt;
  
  /// Get all messages including system prompt
  List<Map<String, String>> getMessagesForAPI() {
    final result = <Map<String, String>>[];
    
    if (_systemPrompt != null) {
      result.add({'role': 'system', 'content': _systemPrompt!});
    }
    
    for (final msg in _messages) {
      result.add({'role': msg.role, 'content': msg.content});
    }
    
    return result;
  }

  /// Set the system prompt
  void setSystemPrompt(String prompt) {
    _systemPrompt = prompt;
  }

  /// Add a user message
  void addUserMessage(String content) {
    _messages.add(ChatMessage(role: 'user', content: content, timestamp: DateTime.now()));
    _enforceTokenLimit();
  }

  /// Add an assistant response
  void addAssistantMessage(String content) {
    _messages.add(ChatMessage(role: 'assistant', content: content, timestamp: DateTime.now()));
    _enforceTokenLimit();
  }

  /// Add interview context (transcript update)
  void addInterviewContext(String transcript) {
    // Find and update existing interview context or add new
    final existingIdx = _messages.indexWhere((m) => m.isInterviewContext);
    
    if (existingIdx >= 0) {
      _messages[existingIdx] = ChatMessage(
        role: 'user',
        content: '[INTERVIEW TRANSCRIPT]\n$transcript',
        timestamp: DateTime.now(),
        isInterviewContext: true,
      );
    } else {
      _messages.add(ChatMessage(
        role: 'user',
        content: '[INTERVIEW TRANSCRIPT]\n$transcript',
        timestamp: DateTime.now(),
        isInterviewContext: true,
      ));
    }
    _enforceTokenLimit();
  }

  /// Enforce token limit by removing oldest non-system messages
  void _enforceTokenLimit() {
    int totalChars = (_systemPrompt?.length ?? 0);
    for (final msg in _messages) {
      totalChars += msg.content.length;
    }
    
    final estimatedTokens = totalChars / charsPerToken;
    
    while (estimatedTokens > maxContextTokens && _messages.length > 1) {
      // Remove oldest non-interview-context message
      final idx = _messages.indexWhere((m) => !m.isInterviewContext);
      if (idx >= 0) {
        _messages.removeAt(idx);
      } else {
        break;
      }
    }
  }

  /// Get message count
  int get messageCount => _messages.length;
  
  /// Estimate current token usage
  int get estimatedTokens {
    int totalChars = (_systemPrompt?.length ?? 0);
    for (final msg in _messages) {
      totalChars += msg.content.length;
    }
    return (totalChars / charsPerToken).ceil();
  }

  /// Clear all messages (but keep system prompt)
  void clear() {
    _messages.clear();
  }

  /// Full reset including system prompt
  void reset() {
    _messages.clear();
    _systemPrompt = null;
  }

  /// Get the last N messages (for context or recall)
  List<ChatMessage> getLastNMessages(int n) {
    if (_messages.length <= n) {
      return List.from(_messages);
    }
    return _messages.sublist(_messages.length - n);
  }

  /// Get all questions that have been asked (user messages only)
  List<String> getQuestionHistory() {
    return _messages
        .where((m) => m.role == 'user' && !m.isInterviewContext)
        .map((m) => m.content)
        .toList();
  }

  /// Get answer for a specific question by index (0 = first question)
  String? getAnswerForQuestion(int questionIndex) {
    final questions = getQuestionHistory();
    if (questionIndex < 0 || questionIndex >= questions.length) return null;
    
    // Find the assistant message that follows this question
    int questionCount = 0;
    for (int i = 0; i < _messages.length; i++) {
      if (_messages[i].role == 'user' && !_messages[i].isInterviewContext) {
        if (questionCount == questionIndex) {
          // Look for next assistant message
          for (int j = i + 1; j < _messages.length; j++) {
            if (_messages[j].role == 'assistant') {
              return _messages[j].content;
            }
          }
        }
        questionCount++;
      }
    }
    return null;
  }
}
class ChatMessage {
  final String role;
  final String content;
  final DateTime timestamp;
  final bool isInterviewContext;

  ChatMessage({
    required this.role,
    required this.content,
    required this.timestamp,
    this.isInterviewContext = false,
  });
}
