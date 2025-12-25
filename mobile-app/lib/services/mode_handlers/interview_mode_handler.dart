import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/mode_handlers/mode_handler.dart';
import 'package:demo_ai_even/services/text_service.dart';
import 'package:demo_ai_even/services/text_display_service.dart';
import 'package:demo_ai_even/services/evenai.dart';
import 'package:demo_ai_even/services/interview/interview_coach_service.dart';
import 'package:demo_ai_even/services/interview/interview_transcription_service.dart';
import 'package:demo_ai_even/services/interview/interview_logger.dart';
import 'package:demo_ai_even/services/interview/question_detector.dart';
import 'package:path_provider/path_provider.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Mode handler for real-time interview coaching.
///
/// Triggered by "interview" keyword. Provides continuous transcription with
/// diarization and suggests answers when interviewer asks questions.
class InterviewModeHandler extends ModeHandler with TerminateDetection {
  final InterviewCoachService _coachService = InterviewCoachService.instance;
  final InterviewTranscriptionService _transcriptionService = InterviewTranscriptionService();
  final InterviewLogger _logger = InterviewLogger.instance;

  bool _isActive = false;
  
  /// Interview-specific timer interval (7 seconds for more reading time)
  static const int _interviewTimerMs = 7000;
  
  /// Debounce timer for question processing
  Timer? _questionDebounce;
  
  /// Last processed question to avoid duplicates
  String _lastProcessedQuestion = '';

  InterviewModeHandler();

  @override
  String get modeName => 'interview';

  @override
  bool get isActive => _isActive;

  @override
  bool canEnterMode(String transcript) {
    return RegExp(r'\binterview\b', caseSensitive: false).hasMatch(transcript);
  }

  @override
  bool canHandleInMode(String transcript) => _isActive;

  @override
  bool isTerminateCommand(String transcript) => matchesTerminate(transcript);

  @override
  Future<ModeResult> enterMode(String transcript) async {
    _log("🎤 InterviewModeHandler: Entering interview coaching mode.");
    
    // Full reset before entering (ensures clean state on re-entry)
    _fullReset();
    
    _isActive = true;
    
    // Set interview-specific timer (7 seconds instead of 5) for BOTH text services
    TextService.customIntervalMs = _interviewTimerMs;
    TextDisplayService.customIntervalSeconds = 7;
    _log('🎤 InterviewModeHandler: Set timer interval to 7 seconds');
    
    // Start logging session
    await _logger.startSession();
    await _logger.logEvent('Interview mode started');
    
    // Load resume and application content
    final resumeContent = await _loadResumeContent();
    final applicationContent = await _loadApplicationContent();
    
    // Initialize coach service with loaded content
    _coachService.initialize(
      resumeContent: resumeContent,
      applicationContent: applicationContent,
    );
    
    await _logger.logEvent('Resume loaded (${resumeContent.length} chars)');
    
    // Set up transcription callbacks for diarization
    _transcriptionService.onQuestionDetected = _onQuestionDetected;
    _transcriptionService.onUtterance = _onUtterance;
    
    // Start continuous transcription with diarization
    await _transcriptionService.startContinuousTranscription();
    
    // Display entry message
    final welcomeText = 'Interview Coach Active. Listening with speaker detection. Say "terminate" to exit.';
    EvenAI.updateDynamicText(welcomeText);
    await TextService.get.startSendText(welcomeText);
    
    return const ModeResult(
      continueListening: true,
      displayText: null,
      handled: true,
    );
  }

  /// Callback when any utterance is received (for logging)
  void _onUtterance(DiarizedUtterance utterance) {
    _logger.logTranscript(
      speakerId: utterance.speakerId,
      text: utterance.text,
      isQuestion: false,
    );
  }

  /// Send audio to the diarization transcription service
  void sendAudioToDiarizer(Uint8List pcmData) {
    if (_isActive) {
      _transcriptionService.sendAudio(pcmData);
    }
  }

  /// Load resume content from file
  Future<String> _loadResumeContent() async {
    try {
      // Try app documents directory first
      final appDir = await getApplicationDocumentsDirectory();
      var file = File('${appDir.path}/Resume.txt');
      
      if (await file.exists()) {
        _log("🎤 InterviewModeHandler: Loaded resume from ${file.path}");
        return await file.readAsString();
      }
      
      // Fallback to bundled resume (hardcoded for now)
      _log("🎤 InterviewModeHandler: Using bundled resume content");
      return _getBundledResume();
    } catch (e) {
      _log("🎤 InterviewModeHandler: Error loading resume: $e");
      return _getBundledResume();
    }
  }

  /// Load application/job description content from file
  Future<String> _loadApplicationContent() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      
      // Try Job_Description.txt first (preferred)
      var file = File('${appDir.path}/Job_Description.txt');
      if (await file.exists()) {
        _log("🎤 InterviewModeHandler: Loaded job description from ${file.path}");
        return await file.readAsString();
      }
      
      // Fallback to application.txt
      file = File('${appDir.path}/application.txt');
      if (await file.exists()) {
        _log("🎤 InterviewModeHandler: Loaded application from ${file.path}");
        return await file.readAsString();
      }
      
      // Try bundled Job_Description.txt in app directory
      _log("🎤 InterviewModeHandler: Using bundled job description");
      return _getBundledJobDescription();
    } catch (e) {
      _log("🎤 InterviewModeHandler: Error loading job description: $e");
      return _getBundledJobDescription();
    }
  }

  @override
  Future<ModeResult> handleCommand(String transcript) async {
    // Check for terminate
    if (isTerminateCommand(transcript)) {
      _log("🎤 InterviewModeHandler: User terminated interview mode.");
      await _logger.logEvent('User said terminate');
      reset();
      return ModeResult.endSession;
    }

    // Check for "repeat" command to recall previous questions
    if (_isRepeatCommand(transcript)) {
      return await _handleRepeatCommand(transcript);
    }

    _log("🎤 InterviewModeHandler: Received: '$transcript'");

    // Detect if this is a question we should answer
    final detection = QuestionDetector.detect(transcript);
    _log("🎤 InterviewModeHandler: Detection: $detection");

    if (detection.isQuestion && detection.confidence >= 0.6) {
      // Avoid duplicate processing
      if (transcript == _lastProcessedQuestion) {
        _log("🎤 InterviewModeHandler: Skipping duplicate question.");
        return const ModeResult(continueListening: true, handled: true);
      }
      _lastProcessedQuestion = transcript;
      
      // Log the question
      await _logger.logTranscript(speakerId: 0, text: transcript, isQuestion: true);
      
      // Generate coaching response
      _log("🎤 InterviewModeHandler: Question detected, generating coaching response...");
      
      final response = await _coachService.generateResponse(
        question: transcript,
        transcriptContext: _transcriptionService.getFormattedTranscript(),
      );
      
      // Log the coaching response
      await _logger.logCoachingResponse(question: transcript, response: response);
      
      // Display with Coach: prefix - ensure interview timer is set
      final displayText = 'Coach: $response';
      EvenAI.updateDynamicText(displayText);
      TextService.customIntervalMs = _interviewTimerMs; // Force 7-second timer
      await TextService.get.startSendText(displayText);
      
      return const ModeResult(
        continueListening: true,
        displayText: null,
        handled: true,
      );
    }

    // Not a question - continue listening
    return const ModeResult(
      continueListening: true,
      displayText: null,
      handled: true,
    );
  }

  /// Check if this is a repeat/recall command
  bool _isRepeatCommand(String transcript) {
    final lower = transcript.toLowerCase();
    return lower.contains('repeat') || 
           lower.contains('again') || 
           lower.contains('first question') ||
           lower.contains('last question');
  }

  /// Handle repeat command to recall previous Q&A
  Future<ModeResult> _handleRepeatCommand(String transcript) async {
    final lower = transcript.toLowerCase();
    final questions = _coachService.getQuestionHistory();
    
    if (questions.isEmpty) {
      final msg = 'No previous questions to repeat.';
      EvenAI.updateDynamicText(msg);
      await TextService.get.startSendText(msg);
      return const ModeResult(continueListening: true, handled: true);
    }

    String question;
    String? answer;
    
    if (lower.contains('first')) {
      question = questions.first;
      answer = _coachService.getAnswerForQuestion(0);
    } else {
      // Default to last question
      question = questions.last;
      answer = _coachService.getAnswerForQuestion(questions.length - 1);
    }

    final displayText = 'Q: $question\n\nCoach: ${answer ?? "No answer recorded"}';
    EvenAI.updateDynamicText(displayText);
    await TextService.get.startSendText(displayText);
    
    await _logger.logEvent('Repeated question: $question');
    
    return const ModeResult(continueListening: true, handled: true);
  }

  /// Callback when a question is detected from diarization service
  void _onQuestionDetected(String question, int speakerId) {
    // Only process if from interviewer (typically speaker 0)
    if (speakerId == 0) {
      _questionDebounce?.cancel();
      _questionDebounce = Timer(const Duration(milliseconds: 800), () async {
        if (!_isActive) return;
        
        _log("🎤 InterviewModeHandler: Diarizer detected question from interviewer");
        
        // Avoid duplicate processing
        if (question == _lastProcessedQuestion) return;
        _lastProcessedQuestion = question;
        
        await _logger.logTranscript(speakerId: 0, text: question, isQuestion: true);
        
        final response = await _coachService.generateResponse(
          question: question,
          transcriptContext: _transcriptionService.getFormattedTranscript(),
        );
        
        await _logger.logCoachingResponse(question: question, response: response);
        
        final displayText = 'Coach: $response';
        EvenAI.updateDynamicText(displayText);
        TextService.customIntervalMs = _interviewTimerMs; // Force 7-second timer
        await TextService.get.startSendText(displayText);
      });
    }
  }

  /// Bundled resume content (Pruitt's actual resume)
  String _getBundledResume() {
    return '''PRUITT COLON | AI/ML Software Engineer
Casa Grande, AZ | PruittColon@gmail.com | (207) 337-0340 | whyhirepruitt.dev | github.com/pruittcolon

PROFESSIONAL SUMMARY
AI/ML Engineer with Master's degree in Software Engineering (AI specialization) and proven track record building production-grade AI systems. Architected a cognitive AI orchestration platform solving single-GPU contention through novel semaphore design while implementing Two system reasoning for mathematical validation of LLM outputs.

CORE COMPETENCIES
AI/ML: Deep Learning • Neural Networks • NLP • Real-Time ASR • Speaker Diarization • RAG • LLM Integration • AutoML • Time-Series Forecasting • Graph Neural Networks
MLOps & Infrastructure: GPU Optimization • Docker • Kubernetes • CUDA • Distributed Systems • PostgreSQL
Software Engineering: Python • FastAPI • REST APIs • WebSocket • Microservices • RBAC

SIGNATURE PROJECT
Cognitive AI Orchestration Platform (NeMo_Server) | 2023 – Present
• Pioneered two system cognitive architecture integrating Titan (AutoML), Oracle (causality), Newton (Genetic Programming), Chronos (forecasting), and Galileo (GNN)
• Solved critical GPU contention through novel Redis-backed GPU semaphore design
• Architected zero-trust RAG memory system with FAISS and encrypted vector stores
• Delivered real-time speech transcription, speaker diarization, emotion analysis, and Gemma LLM reasoning

PROFESSIONAL EXPERIENCE
BTI Solutions | Network Engineer | San Diego, CA | June 2024 – June 2025
• Executed Samsung 5G/LTE site validation for Tier-1 carriers
• Engineered automated PowerShell toolkit that became mandatory team workflow

EDUCATION
MS Software Engineering (AI Specialization) | Western Governors University
BS Computer Science | University of the People''';
  }

  /// Bundled job description (from Job_Description.txt)
  String _getBundledJobDescription() {
    return '''Product Owner, AI Solutions

About the Role:
Tech-forward company serving residential, commercial, and short-term rental industries.
Industry leader in vacation rental software with cutting-edge technology.

Key Responsibilities:
• Define, refine, and manage product backlogs with AI-driven enhancements
• Collaborate with stakeholders to gather and prioritize requirements for AI features
• Write user stories and acceptance criteria for AI-powered features
• Work with UI/UX, developers, QA, and AI/ML teams
• Lead sprint planning, backlog grooming, and product demos
• Coordinate with integration partners/vendors for APIs and data pipelines
• Perform user acceptance testing for AI/ML features
• Research industry trends with focus on AI advancements

Requirements:
• 2+ years as Product Owner with AI/ML project experience
• Understanding of machine learning, NLP, computer vision, data science
• AI literacy, prompt engineering, ethical/regulatory awareness
• Agile methodologies (JIRA, Asana, Figma)
• Experience with APIs, data integrations
• Understanding of XML/JSON

Note: This is a completely backend role.''';
  }

  /// Full internal reset (used before re-entry)
  void _fullReset() {
    _isActive = false;
    _questionDebounce?.cancel();
    _questionDebounce = null;
    _lastProcessedQuestion = '';
    _transcriptionService.stopTranscription();
    _transcriptionService.clearBuffer();
    _coachService.reset();
    TextService.customIntervalMs = null; // Reset timer to default
    TextDisplayService.customIntervalSeconds = null; // Reset timer to default
  }

  @override
  void reset() {
    _log("🎤 InterviewModeHandler: Resetting mode.");
    _logger.logEvent('Interview mode ended');
    _logger.endSession();
    _fullReset();
  }
}
