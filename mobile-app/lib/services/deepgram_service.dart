import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:deepgram_speech_to_text/deepgram_speech_to_text.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:demo_ai_even/services/timing_service.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

class DeepgramService {
  final String _apiKey = dotenv.env['DEEPGRAM_API_KEY'] ?? '';
  final TimingService _timing = TimingService.instance;

  Deepgram? _deepgram;
  StreamSubscription<DeepgramListenResult>? _responseSubscription;
  StreamController<List<int>>? _audioStreamController;
  Completer<String>? _transcriptCompleter;
  
  bool _firstChunkReceived = false;

  /// Starts a new streaming transcription session with Deepgram.
  ///
  /// Returns a Future that completes with the final transcript once the stream is closed.
  Future<String> startStreaming() {
    _timing.startTimer('stt_session');
    _log("DeepgramService: Starting stream...");
    
    if (_apiKey.isEmpty) {
      _log("DeepgramService: ERROR - DEEPGRAM_API_KEY is not set in .env file.");
      return Future.value('');
    }

    _firstChunkReceived = false;
    _transcriptCompleter = Completer<String>();
    _audioStreamController = StreamController<List<int>>();
    _deepgram = Deepgram(_apiKey);

    _timing.logMilestone('deepgram_stream_opened');

    // Corrected: Use the `listen` instance method and provide the correct type.
    _responseSubscription = _deepgram!.listen.live(
      _audioStreamController!.stream,
      queryParams: {
        'encoding': 'linear16',
        'sampleRate': 16000,
        'interim_results': true,
        'smart_format': true,
      },
    ).listen(
      (response) {
        final transcript = response.transcript ?? '';
        // Corrected: Check the 'is_final' key in the response map to determine if it is a final result.
        final isFinal = response.map['is_final'] == true;

        if (transcript.isNotEmpty && isFinal) {
          final sttLatency = _timing.stopTimer('stt_session');
          _log("DeepgramService: Received final transcript: '$transcript' (${sttLatency}ms total STT)");
          _timing.logMilestone('stt_final_transcript', 'Length: ${transcript.length} chars');
          
          if (!_transcriptCompleter!.isCompleted) {
            _transcriptCompleter!.complete(transcript);
          }
        }
      },
      onDone: () {
        _log("DeepgramService: Stream 'onDone' called.");
        if (!_transcriptCompleter!.isCompleted) {
          _transcriptCompleter!.complete('');
        }
      },
      onError: (error) {
        _log("DeepgramService: Stream error: $error");
        _timing.logMilestone('stt_error', error.toString());
        if (!_transcriptCompleter!.isCompleted) {
          _transcriptCompleter!.completeError(error);
        }
      },
    );

    return _transcriptCompleter!.future;
  }

  /// Sends a chunk of PCM audio data to the Deepgram stream.
  void sendAudio(Uint8List pcmData) {
    if (_audioStreamController != null && !_audioStreamController!.isClosed) {
      // Log first chunk timing
      if (!_firstChunkReceived) {
        _firstChunkReceived = true;
        _timing.logMilestone('first_audio_chunk', 'Size: ${pcmData.length} bytes');
      }
      _audioStreamController!.add(pcmData);
    }
  }

  /// Stops the audio stream and signals to Deepgram to finalize the transcript.
  Future<void> stopStreaming() async {
    _log("DeepgramService: Stopping stream...");
    _timing.logMilestone('stt_stream_closing');
    
    await _audioStreamController?.close();
    await _responseSubscription?.cancel();
    _audioStreamController = null;
    _responseSubscription = null;
    _deepgram = null;
    _firstChunkReceived = false;
  }
}
