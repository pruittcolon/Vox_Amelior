import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/chat_service.dart';
import 'package:demo_ai_even/services/web_search_service.dart';
import 'package:demo_ai_even/services/interview/question_detector.dart';
import 'package:demo_ai_even/services/interview/conversation_memory_service.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for generating interview coaching responses.
///
/// Combines resume, application, and transcript context to provide
/// real-time suggested answers during interviews.
class InterviewCoachService {
  static InterviewCoachService? _instance;
  static InterviewCoachService get instance => _instance ??= InterviewCoachService._();
  
  final ChatService _chatService = ChatService.instance;
  final WebSearchService _webSearchService = WebSearchService.instance;
  final ConversationMemoryService _memory = ConversationMemoryService();
  
  String _resumeContent = '';
  String _applicationContent = '';

  InterviewCoachService._();

  /// Initialize with resume and application content
  void initialize({String? resumeContent, String? applicationContent}) {
    _resumeContent = resumeContent ?? '[No resume loaded]';
    _applicationContent = applicationContent ?? '[No job application loaded]';
    
    _memory.reset();
    _memory.setSystemPrompt(_buildSystemPrompt());
    
    _log("InterviewCoach: Initialized with resume (${_resumeContent.length} chars) and application (${_applicationContent.length} chars)");
  }

  /// Build the coaching system prompt
  String _buildSystemPrompt() {
    return '''You are an elite interview coach providing real-time suggested answers during a live job interview.

YOUR MISSION:
Connect the candidate's experience DIRECTLY to the job requirements. Every answer should show why their background makes them the ideal fit.

CRITICAL RULES:
1. Be CONCISE - max 80 words. Candidate reads on tiny smart glasses.
2. ALWAYS reference specific job requirements when relevant
3. Use STAR method (Situation, Task, Action, Result) for behavioral questions
4. Draw specific examples from the resume that match job needs
5. Sound authentic and conversational, not robotic
6. Never say "I would say..." - give the direct answer

CANDIDATE'S RESUME:
$_resumeContent

JOB DESCRIPTION/REQUIREMENTS:
$_applicationContent

When answering, explicitly connect resume experience to job requirements. Be specific, not generic.''';
  }

  /// Generate a coaching response for a question
  Future<String> generateResponse({
    required String question,
    required String transcriptContext,
  }) async {
    _log("InterviewCoach: Generating response for: '$question'");
    
    // Check for research trigger
    final useWebSearch = QuestionDetector.hasResearchTrigger(question);
    final cleanQuestion = useWebSearch 
        ? QuestionDetector.stripResearchTrigger(question)
        : question;

    // Build the prompt
    final userPrompt = '''INTERVIEW CONTEXT:
$transcriptContext

CURRENT QUESTION: $cleanQuestion

Provide a suggested answer (max 80 words):''';

    String response;
    
    if (useWebSearch) {
      _log("InterviewCoach: Using web search for factual query...");
      response = await _webSearchService.sendWebSearchRequest(cleanQuestion);
    } else {
      _log("InterviewCoach: Using GPT for coaching response...");
      // Use chat with custom system prompt
      response = await _chatService.sendChatRequestWithPrompt(
        userPrompt,
        _buildSystemPrompt(),
      );
    }

    // Store in memory for context
    _memory.addUserMessage(question);
    _memory.addAssistantMessage(response);

    return _formatResponseForDisplay(response);
  }

  /// Format response for glasses display
  String _formatResponseForDisplay(String response) {
    // Remove any markdown or extra formatting
    String clean = response
        .replaceAll(RegExp(r'\*\*'), '')
        .replaceAll(RegExp(r'\*'), '')
        .replaceAll(RegExp(r'#+\s*'), '')
        .trim();

    // No truncation - let TextService pagination handle long responses
    return clean;
  }

  /// Update transcript context
  void updateTranscript(String transcript) {
    _memory.addInterviewContext(transcript);
  }

  /// Get estimated token usage
  int get estimatedTokens => _memory.estimatedTokens;

  /// Get question history for recall
  List<String> getQuestionHistory() => _memory.getQuestionHistory();

  /// Get answer for a specific question index
  String? getAnswerForQuestion(int index) => _memory.getAnswerForQuestion(index);

  /// Reset the service
  void reset() {
    _memory.reset();
    _resumeContent = '';
    _applicationContent = '';
  }
}
