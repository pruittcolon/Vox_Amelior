import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/chat_service.dart';
import 'package:demo_ai_even/services/web_search_service.dart';
import 'package:demo_ai_even/services/timing_service.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/evenai.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for default AI chat interactions.
///
/// This is the fallback mode when no other mode is active.
/// Supports web search via "Uplink" trigger word.
class ChatModeHandler extends ModeHandler with TerminateDetection {
  final ChatService _chatService;
  final WebSearchService _webSearchService;
  final TimingService _timing = TimingService.instance;

  bool _isActive = false;
  bool _isAwaitingFollowUp = false;

  ChatModeHandler({ChatService? chatService, WebSearchService? webSearchService})
      : _chatService = chatService ?? ChatService.instance,
        _webSearchService = webSearchService ?? WebSearchService.instance;

  @override
  String get modeName => 'chat';

  @override
  bool get isActive => _isActive;

  /// Whether we're waiting for a follow-up command (continue/terminate)
  bool get isAwaitingFollowUp => _isAwaitingFollowUp;

  @override
  bool canEnterMode(String transcript) {
    // Chat mode is the default - always accepts if no other mode handles it
    return true;
  }

  @override
  bool canHandleInMode(String transcript) {
    return _isActive || _isAwaitingFollowUp;
  }

  @override
  bool isTerminateCommand(String transcript) {
    return matchesTerminate(transcript);
  }

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _timing.logMilestone('chat_mode_entered');
    
    // CHECK TERMINATE FIRST
    if (isTerminateCommand(transcript)) {
      _log("💬 ChatModeHandler: Terminate detected on entry - ending immediately.");
      reset();
      return ModeResult.endSession;
    }
    
    _log("💬 ChatModeHandler: Processing chat query.");
    _isActive = true;
    _isAwaitingFollowUp = false;

    return await _handleChatQuery(transcript);
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    final lower = transcript.toLowerCase().trim();

    // CHECK TERMINATE FIRST
    if (isTerminateCommand(lower)) {
      _log("💬 ChatModeHandler: User terminated chat mode.");
      reset();
      return ModeResult.endSession;
    }

    // Handle follow-up commands
    if (_isAwaitingFollowUp) {
      if (lower.contains('continue')) {
        _log("💬 ChatModeHandler: User wants to continue.");
        _isAwaitingFollowUp = false;
        return const ModeResult(
          continueListening: true,
          displayText: null,
          handled: true,
        );
      }

      // Treat as a new question
      _isAwaitingFollowUp = false;
    }

    return await _handleChatQuery(transcript);
  }

  Future<ModeResult> _handleChatQuery(String question) async {
    try {
      // TIER 1: Check for web search trigger ("research")
      final bool isWebSearch = WebSearchService.hasUplinkTrigger(question);
      
      // TIER 2: Check for advanced trigger ("advanced") - only if not web search
      final bool isAdvanced = !isWebSearch && ChatService.hasAdvancedTrigger(question);
      
      // Clean the query based on which trigger was found
      String queryToUse = question;
      if (isWebSearch) {
        queryToUse = WebSearchService.stripUplinkTrigger(question);
      } else if (isAdvanced) {
        queryToUse = ChatService.stripAdvancedTrigger(question);
      }

      // DEBUG: Log trigger detection
      _log("🔍 DEBUG: Raw input='$question'");
      _log("🔍 DEBUG: research=$isWebSearch, advanced=$isAdvanced, clean='$queryToUse'");

      _timing.logMilestone('question_received', 'Q: "$queryToUse"');

      // Determine prefix based on mode
      // research → "GPT Web"
      // advanced → "5.1"  
      // default  → "Q"
      String prefix;
      if (isWebSearch) {
        prefix = 'GPT Web';
      } else if (isAdvanced) {
        prefix = '5.1';
      } else {
        prefix = 'Q';
      }

      _log("💬 ChatModeHandler: Mode prefix='$prefix', query='$queryToUse'");

      // Build question display
      final formattedQ = _formatQuestion(queryToUse);
      final questionText = "$prefix: $formattedQ";
      
      // Send question to glasses
      EvenAI.updateDynamicText(questionText);
      await TextService.get.startSendText(questionText);
      
      _log("💬 ChatModeHandler: Question displayed, waiting 1.5s...");

      // Start timing
      _timing.startTimer('llm_request');

      // Get answer from appropriate service
      String answer;
      if (isWebSearch) {
        _log("🌐 ChatModeHandler: Using web search service...");
        answer = await _webSearchService.sendWebSearchRequest(queryToUse);
      } else if (isAdvanced) {
        _log("🧠 ChatModeHandler: Using advanced (5.1) AI...");
        answer = await _chatService.sendAdvancedChatRequest(queryToUse);
      } else {
        _log("🤖 ChatModeHandler: Using standard (5.1mini) AI...");
        answer = await _chatService.sendChatRequest(queryToUse);
      }
      
      final latency = _timing.stopTimer('llm_request');
      _log("💬 ChatModeHandler: Got answer (${latency}ms).");

      // Ensure minimum 1.5s display time for question
      if (latency < 1500) {
        await Future.delayed(Duration(milliseconds: 1500 - latency));
      }

      // Ready for follow-up
      _isAwaitingFollowUp = true;
      _isActive = false;

      _timing.logMilestone('response_formatted', 'Ready to display');

      return ModeResult(
        continueListening: true,
        displayText: answer,
      );
    } catch (e) {
      _log("💬 ChatModeHandler: Error - $e");
      _timing.logMilestone('chat_error', e.toString());
      reset();
      return ModeResult(
        continueListening: false,
        displayText: "Sorry, an error occurred.",
      );
    }
  }

  /// Format question for display
  String _formatQuestion(String question) {
    if (question.isEmpty) return question;
    
    String formatted = question[0].toUpperCase() + question.substring(1);
    if (!formatted.endsWith('?')) {
      formatted = '$formatted?';
    }
    return formatted;
  }

  @override
  void reset() {
    _log("💬 ChatModeHandler: Resetting mode.");
    _isActive = false;
    _isAwaitingFollowUp = false;
  }
}
