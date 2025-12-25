import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/chat_service.dart';
import 'package:demo_ai_even/services/web_search_service.dart';
import 'package:demo_ai_even/services/transcript_buffer_service.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/evenai.dart';
import 'package:demo_ai_even/services/timing_service.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for "ask the robot" feature with web search integration.
///
/// Variants:
/// - "ask the robot [question]" - 20 context lines + web search
/// - "double ask the robot [question]" - 40 context lines + web search
/// - "triple ask the robot [question]" - 60 context lines + web search
/// - "100 ask the robot [question]" - 100 context lines + web search
///
/// All variants use web search to provide current information.
class RobotModeHandler extends ModeHandler with TerminateDetection, WordMatching {
  final ChatService _chatService;
  final WebSearchService _webSearchService;
  final TranscriptBufferService _transcriptBuffer;
  final TimingService _timing = TimingService.instance;

  bool _isActive = false;

  /// Standard trigger phrases (20 context lines)
  static const List<String> triggerPhrases = [
    'ask the robot',
    'ask robot',
    'hey robot',
    'robot',
  ];

  /// Double trigger phrases (40 context lines)
  static const List<String> doubleTriggerPhrases = [
    'double ask the robot',
    'double ask robot',
    'double robot',
  ];

  /// Triple trigger phrases (60 context lines)
  static const List<String> tripleTriggerPhrases = [
    'triple ask the robot',
    'triple ask robot',
    'triple robot',
  ];

  /// 100 context trigger phrases (100 context lines)
  /// Supports: "100", "one hundred", "hundred", "1 hundred"
  static const List<String> hundredTriggerPhrases = [
    '100 ask the robot',
    '100 ask robot',
    '100 robot',
    'one hundred ask the robot',
    'one hundred ask robot',
    'one hundred robot',
    'hundred ask the robot',
    'hundred ask robot',
    'hundred robot',
    '1 hundred ask the robot',
    '1 hundred ask robot',
    '1 hundred robot',
  ];

  RobotModeHandler({
    ChatService? chatService,
    WebSearchService? webSearchService,
    TranscriptBufferService? transcriptBuffer,
  })  : _chatService = chatService ?? ChatService.instance,
        _webSearchService = webSearchService ?? WebSearchService.instance,
        _transcriptBuffer = transcriptBuffer ?? TranscriptBufferService.instance;

  @override
  String get modeName => 'robot';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    final lower = transcript.toLowerCase();
    // Check all trigger variants
    return hundredTriggerPhrases.any((p) => lower.contains(p)) ||
           tripleTriggerPhrases.any((p) => lower.contains(p)) ||
           doubleTriggerPhrases.any((p) => lower.contains(p)) ||
           triggerPhrases.any((p) => lower.contains(p));
  }

  @override
  bool canHandleInMode(String transcript) {
    return _isActive;
  }

  @override
  bool isTerminateCommand(String transcript) {
    return matchesTerminate(transcript);
  }

  /// Determine context line count based on trigger phrase
  int _getContextLines(String transcript) {
    final lower = transcript.toLowerCase();
    
    // Check in order of specificity (most specific first)
    if (hundredTriggerPhrases.any((p) => lower.contains(p))) {
      return 100;
    }
    if (tripleTriggerPhrases.any((p) => lower.contains(p))) {
      return 60;
    }
    if (doubleTriggerPhrases.any((p) => lower.contains(p))) {
      return 40;
    }
    return 20; // Default
  }

  /// Get the trigger type for display
  String _getTriggerType(String transcript) {
    final lower = transcript.toLowerCase();
    
    if (hundredTriggerPhrases.any((p) => lower.contains(p))) {
      return '100';
    }
    if (tripleTriggerPhrases.any((p) => lower.contains(p))) {
      return 'triple';
    }
    if (doubleTriggerPhrases.any((p) => lower.contains(p))) {
      return 'double';
    }
    return 'standard';
  }

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _timing.logMilestone('robot_mode_entered');
    final contextLines = _getContextLines(transcript);
    final triggerType = _getTriggerType(transcript);
    
    _log("🤖 RobotModeHandler: Activated ($triggerType mode, $contextLines context lines)");
    _isActive = true;

    // Extract the question from the transcript
    String question = _extractQuestion(transcript);

    if (question.isEmpty) {
      // No specific question - ask for a summary
      question = "What was the conversation about?";
    }

    return await _handleRobotQuery(question, contextLines);
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    if (isTerminateCommand(lower)) {
      _log("🤖 RobotModeHandler: User terminated.");
      reset();
      return ModeResult.endSession;
    }

    // Check if this is a new robot query with different context level
    final contextLines = _getContextLines(transcript);
    
    // Treat any follow-up as a new question about the conversation
    return await _handleRobotQuery(transcript, contextLines);
  }

  /// Extract the actual question from trigger phrase
  String _extractQuestion(String transcript) {
    String question = transcript.toLowerCase();

    // Remove trigger phrases (check longer ones first)
    final allTriggers = [
      ...hundredTriggerPhrases,
      ...tripleTriggerPhrases,
      ...doubleTriggerPhrases,
      ...triggerPhrases,
    ];
    
    for (final phrase in allTriggers) {
      if (question.contains(phrase)) {
        question = question.replaceFirst(phrase, '').trim();
        break;
      }
    }

    // Clean up common artifacts
    question = question
        .replaceAll(RegExp(r'^[,.\s]+'), '') // Leading punctuation
        .replaceAll(RegExp(r'[,.\s]+$'), '') // Trailing punctuation
        .trim();

    return question;
  }

  Future<ModeResult> _handleRobotQuery(String question, int contextLines) async {
    try {
      // Get conversation context
      final conversationContext = _transcriptBuffer.getFormattedConversation(contextLines);
      final transcriptCount = _transcriptBuffer.length;

      _log("🤖 RobotModeHandler: Processing with $contextLines context lines ($transcriptCount available)");
      _log("🌐 RobotModeHandler: Fetching web search context...");

      // Display question on glasses
      final formattedQ = "🤖 $question";
      EvenAI.updateDynamicText(formattedQ);
      await TextService.get.startSendText(formattedQ);

      _timing.startTimer('robot_query');

      // Fetch web search context for current information
      String webContext = "";
      if (_webSearchService.isConfigured) {
        try {
          webContext = await _webSearchService.sendWebSearchRequest(question);
          _log("🌐 RobotModeHandler: Got web search context (${webContext.length} chars)");
        } catch (e) {
          _log("🌐 RobotModeHandler: Web search failed, continuing without: $e");
        }
      } else {
        _log("🌐 RobotModeHandler: Web search not configured, skipping");
      }

      // Build the improved prompt that handles noisy ASR transcriptions
      final systemPrompt = '''You are a knowledgeable assistant analyzing a real-time conversation.

IMPORTANT - ABOUT THIS TRANSCRIPTION:
• This is AI-generated speech-to-text which may contain errors
• Common issues: homophones ("their/there"), mishearing ("kitchen/chicken"), garbled words
• Recent entries may not have speaker labels (diarization runs every 30 seconds)
• Entries marked "[Speaker TBD]" or "UNKNOWN" are not yet diarized
• Use context and best judgment to infer the intended meaning

YOUR TASK:
Focus on answering the question accurately. The TRUTH matters more than who said what.
If speakers disagree, determine the factually correct answer using:
1. Your knowledge base
2. The web search context provided (if available)
3. Logical reasoning

CONVERSATION CONTEXT ($contextLines lines):
---
$conversationContext
---

${webContext.isNotEmpty ? '''
CURRENT WEB INFORMATION:
---
$webContext
---
''' : ''}

RESPONSE GUIDELINES:
• Keep answers under 100 words (displayed on smart glasses)
• Be direct and factual
• If the web provides more current information, prefer it
• Don't mention transcription quality issues in your response''';

      // Send to ChatGPT with context
      final answer = await _chatService.sendChatRequestWithPrompt(
        question,
        systemPrompt,
      );

      final latency = _timing.stopTimer('robot_query');
      _log("🤖 RobotModeHandler: Got answer in ${latency}ms");

      // Ensure minimum display time for question
      if (latency < 1500) {
        await Future.delayed(Duration(milliseconds: 1500 - latency));
      }

      // Done with this query, but stay available for follow-ups
      _isActive = false;

      return ModeResult(
        continueListening: true,
        displayText: answer,
      );
    } catch (e) {
      _log("🤖 RobotModeHandler: Error - $e");
      reset();
      return ModeResult(
        continueListening: false,
        displayText: "Sorry, I couldn't process that question.",
      );
    }
  }

  @override
  void reset() {
    _log("🤖 RobotModeHandler: Resetting mode.");
    _isActive = false;
  }
}
