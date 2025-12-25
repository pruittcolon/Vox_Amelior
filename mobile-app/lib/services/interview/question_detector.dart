/// Question detection utilities for interview mode.
import 'package:flutter/foundation.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}
///
/// Detects when the interviewer asks a question that requires a coached response.
class QuestionDetector {
  /// Patterns that indicate a question is being asked
  static final List<RegExp> _questionPatterns = [
    // Direct question words
    RegExp(r'\b(who|what|when|where|why|how)\b.*\??', caseSensitive: false),
    
    // Would/could/can/do/have you questions
    RegExp(r'\b(would you|could you|can you|do you|have you|are you)\b', caseSensitive: false),
    
    // Tell me about / describe / explain
    RegExp(r'\btell me (about|more)\b', caseSensitive: false),
    RegExp(r'\b(describe|explain|walk me through)\b', caseSensitive: false),
    
    // Give me an example
    RegExp(r'\bgive (me )?(an |a )?(example|scenario)\b', caseSensitive: false),
    
    // What would you / How would you
    RegExp(r'\b(what|how) would you\b', caseSensitive: false),
  ];

  /// Patterns that indicate this is likely NOT a coaching-worthy question
  /// (e.g., small talk, procedural questions)
  static final List<RegExp> _excludePatterns = [
    RegExp(r'\b(how are you|nice to meet|thank you)\b', caseSensitive: false),
    RegExp(r'\bdo you have (any )?questions\b', caseSensitive: false),
    RegExp(r'\bcan you hear me\b', caseSensitive: false),
  ];

  /// Check if the transcript contains a question that needs coaching.
  /// 
  /// Returns a [QuestionDetectionResult] with detection status and confidence.
  static QuestionDetectionResult detect(String transcript) {
    final normalized = transcript.trim().toLowerCase();
    
    if (normalized.isEmpty) {
      return QuestionDetectionResult(
        isQuestion: false,
        confidence: 0.0,
        matchedPattern: null,
      );
    }

    // Check exclusions first
    for (final pattern in _excludePatterns) {
      if (pattern.hasMatch(normalized)) {
        return QuestionDetectionResult(
          isQuestion: false,
          confidence: 0.0,
          matchedPattern: null,
          reason: 'Excluded pattern match',
        );
      }
    }

    // Check for question patterns
    for (final pattern in _questionPatterns) {
      if (pattern.hasMatch(normalized)) {
        // Higher confidence if ends with question mark
        final hasQuestionMark = transcript.trim().endsWith('?');
        final confidence = hasQuestionMark ? 0.95 : 0.75;
        
        return QuestionDetectionResult(
          isQuestion: true,
          confidence: confidence,
          matchedPattern: pattern.pattern,
        );
      }
    }

    // Check if ends with question mark (fallback)
    if (transcript.trim().endsWith('?')) {
      return QuestionDetectionResult(
        isQuestion: true,
        confidence: 0.6,
        matchedPattern: 'question_mark_only',
      );
    }

    return QuestionDetectionResult(
      isQuestion: false,
      confidence: 0.0,
      matchedPattern: null,
    );
  }

  /// Check if the question contains a "research" trigger for web search
  static bool hasResearchTrigger(String question) {
    return RegExp(r'\bresearch\b', caseSensitive: false).hasMatch(question);
  }

  /// Strip the research trigger word from the question
  static String stripResearchTrigger(String question) {
    return question.replaceAll(
      RegExp(r'\bresearch\b[,\s]*', caseSensitive: false), 
      ''
    ).trim();
  }
}

/// Result of question detection
class QuestionDetectionResult {
  /// Whether a question was detected
  final bool isQuestion;
  
  /// Confidence score 0.0 to 1.0
  final double confidence;
  
  /// The pattern that matched (for debugging)
  final String? matchedPattern;
  
  /// Reason for the result (for debugging)
  final String? reason;

  const QuestionDetectionResult({
    required this.isQuestion,
    required this.confidence,
    required this.matchedPattern,
    this.reason,
  });

  @override
  String toString() => 
    'QuestionDetectionResult(isQuestion: $isQuestion, confidence: $confidence, pattern: $matchedPattern)';
}
