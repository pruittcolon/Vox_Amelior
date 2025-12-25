import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:deepgram_speech_to_text/deepgram_speech_to_text.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:demo_ai_even/services/interview/question_detector.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Callback for when a diarized utterance is received
typedef UtteranceCallback = void Function(DiarizedUtterance utterance);

/// Callback for when a question is detected
typedef QuestionCallback = void Function(String question, int speakerId);

/// Service for continuous transcription with speaker diarization for interview mode.
///
/// Unlike DeepgramService which waits for final transcripts, this service
/// provides continuous callbacks for each utterance with speaker identification.
class InterviewTranscriptionService {
  final String _apiKey = dotenv.env['DEEPGRAM_API_KEY'] ?? '';

  Deepgram? _deepgram;
  StreamSubscription<DeepgramListenResult>? _responseSubscription;
  StreamController<List<int>>? _audioStreamController;
  
  bool _isActive = false;
  bool _firstChunkReceived = false;
  
  /// Callback when utterance is received
  UtteranceCallback? onUtterance;
  
  /// Callback when question is detected
  QuestionCallback? onQuestionDetected;
  
  /// Full transcript buffer with speaker labels
  final List<DiarizedUtterance> _transcriptBuffer = [];
  
  /// Current partial utterance being built
  String _currentPartial = '';
  int _currentSpeaker = -1;

  /// Whether the service is currently active
  bool get isActive => _isActive;
  
  /// Get the full transcript buffer
  List<DiarizedUtterance> get transcriptBuffer => List.unmodifiable(_transcriptBuffer);

  /// Start continuous transcription with diarization
  Future<void> startContinuousTranscription() async {
    if (_apiKey.isEmpty) {
      _log("InterviewTranscription: ERROR - DEEPGRAM_API_KEY not set.");
      return;
    }

    _log("InterviewTranscription: Starting continuous diarized stream...");
    _isActive = true;
    _firstChunkReceived = false;
    _transcriptBuffer.clear();
    _currentPartial = '';
    _currentSpeaker = -1;
    
    _audioStreamController = StreamController<List<int>>();
    _deepgram = Deepgram(_apiKey);

    _responseSubscription = _deepgram!.listen.live(
      _audioStreamController!.stream,
      queryParams: {
        'encoding': 'linear16',
        'sampleRate': 16000,
        'interim_results': true,
        'smart_format': true,
        'diarize': true,  // Enable speaker diarization
        'punctuate': true,
      },
    ).listen(
      _handleResponse,
      onDone: _handleStreamDone,
      onError: _handleStreamError,
    );
  }

  /// Handle incoming transcription response
  void _handleResponse(DeepgramListenResult response) {
    final transcript = response.transcript ?? '';
    final isFinal = response.map['is_final'] == true;
    
    // Extract speaker info from words if available
    final words = response.map['channel']?['alternatives']?[0]?['words'] as List?;
    int speakerId = _currentSpeaker;
    
    if (words != null && words.isNotEmpty) {
      // Get speaker from first word
      speakerId = words[0]['speaker'] ?? 0;
    }

    if (transcript.isEmpty) return;

    if (isFinal) {
      // Final utterance - add to buffer and check for questions
      _log("InterviewTranscription: [Speaker $speakerId] $transcript");
      
      final utterance = DiarizedUtterance(
        speakerId: speakerId,
        text: transcript,
        timestamp: DateTime.now(),
        isFinal: true,
      );
      
      _transcriptBuffer.add(utterance);
      onUtterance?.call(utterance);
      
      // Check if this is a question (from interviewer - typically speaker 0)
      final detection = QuestionDetector.detect(transcript);
      if (detection.isQuestion && detection.confidence >= 0.6) {
        _log("InterviewTranscription: Question detected from speaker $speakerId!");
        onQuestionDetected?.call(transcript, speakerId);
      }
      
      _currentPartial = '';
      _currentSpeaker = -1;
    } else {
      // Interim result - update partial
      _currentPartial = transcript;
      _currentSpeaker = speakerId;
    }
  }

  void _handleStreamDone() {
    _log("InterviewTranscription: Stream completed.");
    _isActive = false;
  }

  void _handleStreamError(dynamic error) {
    _log("InterviewTranscription: Stream error: $error");
    _isActive = false;
  }

  /// Send audio chunk to the transcription stream
  void sendAudio(Uint8List pcmData) {
    if (_audioStreamController != null && !_audioStreamController!.isClosed) {
      if (!_firstChunkReceived) {
        _firstChunkReceived = true;
        _log("InterviewTranscription: First audio chunk received.");
      }
      _audioStreamController!.add(pcmData);
    }
  }

  /// Stop the continuous transcription
  Future<void> stopTranscription() async {
    _log("InterviewTranscription: Stopping stream...");
    _isActive = false;
    
    await _audioStreamController?.close();
    await _responseSubscription?.cancel();
    _audioStreamController = null;
    _responseSubscription = null;
    _deepgram = null;
    _firstChunkReceived = false;
  }

  /// Get formatted transcript for GPT context
  String getFormattedTranscript() {
    return _transcriptBuffer
        .map((u) => '[Speaker ${u.speakerId}]: ${u.text}')
        .join('\n');
  }

  /// Clear the transcript buffer
  void clearBuffer() {
    _transcriptBuffer.clear();
  }
}

/// A single diarized utterance
class DiarizedUtterance {
  final int speakerId;
  final String text;
  final DateTime timestamp;
  final bool isFinal;

  DiarizedUtterance({
    required this.speakerId,
    required this.text,
    required this.timestamp,
    required this.isFinal,
  });
  
  @override
  String toString() => '[Speaker $speakerId]: $text';
}
