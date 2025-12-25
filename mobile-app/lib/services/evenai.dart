// EvenAI Coordinator - Refactored from monolithic to microservices architecture
//
// This file now serves as a lightweight coordinator that delegates to specialized services
// and mode handlers. The original 1200+ line file has been reduced to ~350 lines.

import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/ble_manager.dart';
import 'package:demo_ai_even/controllers/evenai_model_controller.dart';
import 'package:demo_ai_even/services/features_services.dart';
import 'package:demo_ai_even/services/proto.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/deepgram_service.dart';
import 'package:demo_ai_even/services/whisperserver_service.dart';

// Mode handlers
import 'package:demo_ai_even/services/mode_handlers/mode_handlers.dart';

import 'package:get/get.dart';

// Debug logging helper - tree-shaken in release builds
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// EvenAI - Main coordinator for voice-controlled smart glasses interactions.
///
/// Responsibilities:
/// - Audio recording start/stop coordination
/// - Speech-to-text service management (Deepgram/WhisperServer)
/// - Mode switching and delegation to appropriate handlers
/// - State management (isRunning, isReceivingAudio)
///
/// All business logic for specific modes (Roku, Alexa, Memory, Chat, etc.)
/// has been extracted to dedicated ModeHandler classes.
class EvenAI {
  static EvenAI? _instance;
  static EvenAI get get => _instance ??= EvenAI._();

  // --- Speech-to-Text Services ---
  final DeepgramService _deepgramService = DeepgramService();
  final WhisperServerService _whisperService = WhisperServerService();

  // --- Mode Handlers ---
  late final List<ModeHandler> _modeHandlers;
  late final RokuModeHandler _rokuHandler;
  late final AlexaModeHandler _alexaHandler;
  late final MemoryModeHandler _memoryHandler;
  late final MenuModeHandler _menuHandler;
  late final ChatModeHandler _chatHandler;
  late final InterviewModeHandler _interviewHandler;
  late final GoogleAIModeHandler _googleAIHandler;

  // --- State ---
  Future<String>? _transcriptFuture;
  bool isReceivingAudio = false;
  ModeHandler? _activeHandler;

  static bool _isRunning = false;
  static bool get isRunning => _isRunning;

  Timer? _recordingTimer;
  final int maxRecordingDuration = 60;

  int _lastStartTime = 0;
  int _lastStopTime = 0;
  final int startTimeGap = 500;
  final int stopTimeGap = 500;

  static set isRunning(bool value) {
    _isRunning = value;
    isEvenAIOpen.value = value;
    isEvenAISyncing.value = value;
  }

  static RxBool isEvenAIOpen = false.obs;
  static RxBool isEvenAISyncing = false.obs;

  // --- Dynamic Text Stream ---
  static final StreamController<String> _textStreamController =
      StreamController<String>.broadcast();
  static Stream<String> get textStream => _textStreamController.stream;

  static void updateDynamicText(String newText) {
    _textStreamController.add(newText);
  }

  EvenAI._() {
    // Initialize mode handlers
    _rokuHandler = RokuModeHandler();
    _alexaHandler = AlexaModeHandler();
    _memoryHandler = MemoryModeHandler();
    _menuHandler = MenuModeHandler();
    _chatHandler = ChatModeHandler();
    _interviewHandler = InterviewModeHandler();
    _googleAIHandler = GoogleAIModeHandler();

    // Order matters: first handler that can handle the transcript wins
    // ChatHandler should be last as it's the fallback
    // Order matters: first handler that can handle the transcript wins
    // GoogleAI should be early (trigger phrase based)
    // ChatHandler should be last as it's the fallback
    _modeHandlers = [
      _googleAIHandler,   // "Hey Google" trigger phrase
      _memoryHandler,
      _alexaHandler,
      _rokuHandler,
      _menuHandler,
      _interviewHandler,  // Before chat handler
      _chatHandler,
    ];
  }

  // --- Audio Handling ---

  void onVoiceChunkReceived(Uint8List pcmData) {
    if (isReceivingAudio) {
      // Route audio to appropriate service based on active mode
      if (_memoryHandler.isActive) {
        _whisperService.sendAudio(pcmData);
      } else if (_interviewHandler.isActive) {
        // Interview mode uses its own transcription service with diarization
        _interviewHandler.sendAudioToDiarizer(pcmData);
        // Also send to regular Deepgram for command detection
        _deepgramService.sendAudio(pcmData);
      } else {
        _deepgramService.sendAudio(pcmData);
      }
    }
  }

  // --- Recording Control ---

  void toStartEvenAIByOS() async {
    _log("EvenAI: Received start command from OS.");
    BleManager.get().startSendBeatHeart();

    int currentTime = DateTime.now().millisecondsSinceEpoch;
    if (currentTime - _lastStartTime < startTimeGap) {
      _log("EvenAI: Start command received too quickly. Ignoring.");
      return;
    }
    _lastStartTime = currentTime;

    // CRITICAL FIX: Don't clear() if we're in an active follow-up state!
    // This was causing _isAwaitingFollowUp to be reset to false
    // before the user could even speak "terminate"
    final preserveState = _chatHandler.isAwaitingFollowUp || 
                          (_activeHandler != null && _activeHandler!.isActive);
    
    if (preserveState) {
      _log("EvenAI: Preserving follow-up state (awaiting: ${_chatHandler.isAwaitingFollowUp})");
      // Only reset audio state, not handlers
      isReceivingAudio = false;
      _recordingTimer?.cancel();
      _recordingTimer = null;
    } else {
      clear();
    }
    
    isReceivingAudio = true;
    isRunning = true;

    _transcriptFuture = _deepgramService.startStreaming();

    _transcriptFuture?.then((transcript) {
      if (transcript.trim().isNotEmpty && isReceivingAudio) {
        _log("EvenAI: Proactively processing received transcript.");
        recordOverByOS();
      }
    });

    await BleManager.invokeMethod("startEvenAI");
    await openEvenAIMic();
    _startRecordingTimer();
  }

  void _startRecordingTimer() {
    _recordingTimer?.cancel();
    _recordingTimer = Timer(Duration(seconds: maxRecordingDuration), () {
      if (isReceivingAudio) {
        _log("EvenAI: Recording timer expired. Forcing end of recording.");
        recordOverByOS();
      }
    });
  }

  /// Start listening for a follow-up command using Deepgram
  void _listenForFollowUpCommand() {
    _log("EvenAI: Starting to listen for a follow-up command.");
    isReceivingAudio = true;
    isRunning = true;
    _transcriptFuture = _deepgramService.startStreaming();

    _transcriptFuture?.then((transcript) {
      if (transcript.trim().isNotEmpty && isReceivingAudio) {
        _log("EvenAI: Proactively processing follow-up transcript.");
        recordOverByOS();
      }
    });

    BleManager.invokeMethod("startEvenAI");
    openEvenAIMic();
    _startRecordingTimer();
  }

  /// Start listening using WhisperServer (for memory mode)
  void _listenWithWhisperServer() {
    _log("EvenAI: Starting to listen using WhisperServer.");
    isReceivingAudio = true;
    isRunning = true;
    _transcriptFuture = _whisperService.startStreaming();

    _transcriptFuture?.then((transcript) {
      if (transcript.trim().isNotEmpty && isReceivingAudio) {
        _log("EvenAI: Processing WhisperServer transcript.");
        recordOverByOS();
      }
    });

    BleManager.invokeMethod("startEvenAI");
    openEvenAIMic();
    _startRecordingTimer();
  }

  // --- Main Recording Over Handler ---

  Future<void> recordOverByOS() async {
    _log('🔍 recordOverByOS: _activeHandler=${_activeHandler?.modeName}, isActive=${_activeHandler?.isActive}, isReceivingAudio=$isReceivingAudio, _isRunning=$_isRunning');
    
    // Handle active mode if one exists
    if (_activeHandler != null && _activeHandler!.isActive) {
      _log('🔍 recordOverByOS: Going to _handleActiveMode (active handler exists)');
      await _handleActiveMode();
      return;
    }

    // Check for special handler states (like chat follow-up)
    if (_chatHandler.isAwaitingFollowUp) {
      _log('🔍 recordOverByOS: Going to _handleActiveMode (chat follow-up)');
      await _handleActiveMode();
      return;
    }

    // No active mode - process as new command
    if (!isReceivingAudio && !_isRunning) {
      _log("EvenAI: recordOverByOS called but not in a receiving state. Ignoring.");
      return;
    }

    _log('EvenAI: Received stop command from OS.');
    int currentTime = DateTime.now().millisecondsSinceEpoch;
    if (currentTime - _lastStopTime < stopTimeGap) {
      _log("EvenAI: Stop command received too quickly. Ignoring.");
      return;
    }
    _lastStopTime = currentTime;

    isReceivingAudio = false;
    _recordingTimer?.cancel();
    _recordingTimer = null;
    updateDynamicText("Processing...");

    await _deepgramService.stopStreaming();
    await BleManager.invokeMethod("stopEvenAI");

    try {
      final transcript = await _transcriptFuture;
      _log("EvenAI: Final transcript: '$transcript'");

      if (transcript == null || transcript.trim().isEmpty) {
        updateDynamicText("No Speech Recognized");
        isEvenAISyncing.value = false;
        await TextService.get.startSendText("No Speech Recognized");
        isRunning = false;
        return;
      }

      final cleanTranscript = transcript.toLowerCase().trim();

      // Find a handler for this command
      for (final handler in _modeHandlers) {
        if (handler.canEnterMode(cleanTranscript)) {
          _activeHandler = handler;
          final result = await handler.enterMode(cleanTranscript);
          await _processResult(result, transcript);
          return;
        }
      }

      // No handler found - shouldn't happen as ChatHandler accepts everything
      _log("EvenAI: No handler found for transcript.");
      await _endSession();

    } catch (e) {
      _log("EvenAI: Error during transcription or processing: $e");
      updateDynamicText("Sorry, an error occurred.");
      isEvenAISyncing.value = false;
      await TextService.get.startSendText("Sorry, an error occurred.");
      isRunning = false;
    }
  }

  Future<void> _handleActiveMode() async {
    isReceivingAudio = false;

    // Stop the appropriate STT service
    if (_memoryHandler.isActive) {
      await _whisperService.stopStreaming();
    } else {
      await _deepgramService.stopStreaming();
    }

    final transcript = await _transcriptFuture;
    final cleanTranscript = transcript?.toLowerCase().trim() ?? "";

    if (cleanTranscript.isEmpty) {
      _log("EvenAI: No speech detected in active mode.");
      // Keep listening in the active mode
      if (_memoryHandler.isActive) {
        _listenWithWhisperServer();
      } else if (_activeHandler?.isActive ?? false) {
        _listenForFollowUpCommand();
      }
      return;
    }

    // Get the active handler
    ModeHandler? handler = _activeHandler;
    if (handler == null || !handler.isActive) {
      // Check if chat handler is awaiting follow-up
      if (_chatHandler.isAwaitingFollowUp) {
        handler = _chatHandler;
      } else {
        await _endSession();
        return;
      }
    }

    // Process the command
    final result = await handler.handleCommand(cleanTranscript);
    await _processResult(result, transcript ?? "");
  }

  Future<void> _processResult(ModeResult result, String originalTranscript) async {
    // Handle special case: save chat Q&A
    if (_activeHandler == _chatHandler && result.displayText != null) {
      final answer = result.displayText!.split("\n\nSay 'Continue")[0];
      if (!answer.contains("error") && originalTranscript.isNotEmpty) {
        saveQuestionItem(originalTranscript, answer);
      }
    }

    // Display text if provided
    if (result.displayText != null) {
      updateDynamicText(result.displayText!);
      isEvenAISyncing.value = false;
      await TextService.get.startSendText(result.displayText!);
    }

    // Handle continue/end logic
    if (!result.handled) {
      // Handler didn't handle it - try other handlers
      await _endSession();
      return;
    }

    if (result.continueListening) {
      // Brief pause for display text to be read
      if (result.displayText != null) {
        await Future.delayed(const Duration(seconds: 2));
      }

      // Start listening again
      if (result.useWhisperServer) {
        _listenWithWhisperServer();
      } else {
        _listenForFollowUpCommand();
      }
    } else {
      // End session - FAST PATH for terminate commands
      // Only delay if there was text displayed that user needs to read
      if (result.displayText != null) {
        await Future.delayed(const Duration(milliseconds: 500));
      }
      // Immediate exit - no delay for terminate commands
      await _endSession();
    }
  }

  Future<void> _endSession() async {
    // CRITICAL: Stop TextService pagination timer BEFORE clearing
    // This prevents pages from continuing to display after terminate
    TextService.get.clear();
    
    await _robustExitBmp();
    clear();
    isRunning = false;
  }

  Future<void> _robustExitBmp() async {
    if (BleManager.get().isConnected) {
      _log("EvenAI: Clearing screen on glasses (double exit).");
      // Call twice rapidly - glasses don't always clear on first try
      await FeaturesServices().exitBmp();
      await FeaturesServices().exitBmp();
    }
  }

  void saveQuestionItem(String title, String content) {
    try {
      final controller = Get.find<EvenaiModelController>();
      controller.addItem(title, content);
    } catch (e) {
      _log("EvenAI: Could not save Q&A: $e");
    }
  }

  void clear() {
    _log("EvenAI: Clearing state.");
    isReceivingAudio = false;
    _recordingTimer?.cancel();
    _recordingTimer = null;
    _transcriptFuture = null;

    // Reset all mode handlers
    for (final handler in _modeHandlers) {
      handler.reset();
    }
    _activeHandler = null;
  }

  Future<void> openEvenAIMic() async {
    final (_, isStartSucc) = await Proto.micOn(lr: "R");
    _log('EvenAI: Mic activation success: $isStartSucc');
    if (!isStartSucc && isReceivingAudio) {
      await Future.delayed(const Duration(milliseconds: 500));
      await openEvenAIMic();
    }
  }

  Future<void> stopEvenAIByOS() async {
    isRunning = false;
    clear();
    await BleManager.invokeMethod("stopEvenAI");
  }

  // --- Touchpad Navigation (delegate to TextDisplayService) ---

  void nextPageByTouchpad() {
    // This is handled by the text display service via TextService
    // Keep for backward compatibility
    _log("EvenAI: nextPageByTouchpad called");
  }

  void lastPageByTouchpad() {
    // This is handled by the text display service via TextService
    // Keep for backward compatibility
    _log("EvenAI: lastPageByTouchpad called");
  }

  static void dispose() {
    _textStreamController.close();
  }
}
