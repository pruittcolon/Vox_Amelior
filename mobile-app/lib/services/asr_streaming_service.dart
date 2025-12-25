import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/app_logger.dart';
import 'package:demo_ai_even/services/auth_service.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Real-time streaming ASR service using WebSocket.
/// 
/// Provides ~500ms latency transcription with deferred diarization.
/// Much faster than the HTTP-based AsrHttpService.
class AsrStreamingService {
  static final AsrStreamingService _instance = AsrStreamingService._();
  static AsrStreamingService get I => _instance;

  AsrStreamingService._();

  final AudioRecorder _recorder = AudioRecorder();
  final StreamController<AsrStreamEvent> _events = StreamController<AsrStreamEvent>.broadcast();
  
  Stream<AsrStreamEvent> get stream => _events.stream;
  
  bool _running = false;
  bool get isRunning => _running;
  
  WebSocketChannel? _channel;
  StreamSubscription? _audioSubscription;
  String? _sessionId;
  
  // Server URL (WebSocket endpoint)
  String get _wsUrl {
    final base = dotenv.env['ASR_SERVER_BASE']?.trim() ?? '';
    if (base.isEmpty) return '';
    // Convert http:// to ws:// or https:// to wss://
    final wsBase = base
        .replaceFirst('http://', 'ws://')
        .replaceFirst('https://', 'wss://');
    return '$wsBase/stream';
  }

  void _logConfig() {
    _log("🔧 [ASR STREAMING] WebSocket URL: '$_wsUrl'");
    if (_wsUrl.isEmpty) {
      _log("❌ [ASR STREAMING] Empty WebSocket URL! Set ASR_SERVER_BASE in .env");
    }
  }

  Future<void> start() async {
    if (_running) return;

    if (!await _ensurePerms()) {
      _emitSystem('Microphone permission denied.', isError: true);
      return;
    }

    _logConfig();
    
    if (_wsUrl.isEmpty) {
      _emitSystem('WebSocket URL not configured.', isError: true);
      return;
    }

    try {
      // Get session token for authentication
      String authenticatedWsUrl = _wsUrl;
      try {
        final sessionToken = await AuthService.instance.getSessionToken();
        if (sessionToken != null && sessionToken.isNotEmpty) {
          // Add token as query parameter for WebSocket authentication
          authenticatedWsUrl = '$_wsUrl?token=$sessionToken';
          _log("🔑 [ASR STREAMING] Session token attached to WebSocket URL");
        } else {
          _log("⚠️ [ASR STREAMING] No session token available, connection may fail");
        }
      } catch (e) {
        _log("⚠️ [ASR STREAMING] Could not get session token: $e");
      }
      
      // Connect to WebSocket
      // Note: SSL bypass is handled globally in main.dart via HttpOverrides.global
      final uri = Uri.parse(authenticatedWsUrl);
      _log("🔗 [ASR STREAMING] Connecting to ${uri.host}:${uri.port == 0 ? 443 : uri.port}${uri.path}");
      
      if (authenticatedWsUrl.startsWith('wss://')) {
        // Use dart:io WebSocket.connect for secure connection
        // HttpOverrides.global (set in main.dart) handles SSL bypass
        final socket = await WebSocket.connect(authenticatedWsUrl);
        _log("✅ [ASR STREAMING] WebSocket connected successfully");
        
        // Wrap the raw WebSocket in IOWebSocketChannel for stream API
        _channel = IOWebSocketChannel(socket);
      } else {
        // For non-secure connections, use regular WebSocketChannel
        _channel = WebSocketChannel.connect(uri);
      }
      
      // Listen for messages from server
      _channel!.stream.listen(
        _handleServerMessage,
        onError: (error) {
          _log("❌ [ASR STREAMING] WebSocket error: $error");
          _emitSystem('WebSocket error: $error', isError: true);
          _running = false;
        },
        onDone: () {
          _log("📋 [ASR STREAMING] WebSocket closed");
          if (_running) {
            _emitSystem('Connection closed unexpectedly', isError: true);
            _running = false;
          }
        },
      );
      
      // Start recording and streaming audio
      await _startAudioStream();
      
      _running = true;
      _emitSystem('Real-time streaming started');
      
    } catch (e, stackTrace) {
      _log("❌ [ASR STREAMING] Failed to start: $e");
      _log("📋 [ASR STREAMING] Stack trace: $stackTrace");
      _emitSystem('Failed to start streaming: $e', isError: true);
    }
  }

  Future<void> _startAudioStream() async {
    // Configure for raw PCM 16-bit mono audio
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      autoGain: true,
      echoCancel: true,
      noiseSuppress: true,
    );
    
    // Start streaming audio
    final audioStream = await _recorder.startStream(config);
    
    _audioSubscription = audioStream.listen((data) {
      // Send raw PCM bytes to server
      if (_channel != null && _running) {
        _channel!.sink.add(data);
      }
    });
    
    _log("🎤 [ASR STREAMING] Audio stream started (16kHz mono PCM)");
  }

  void _handleServerMessage(dynamic message) {
    try {
      if (message is String) {
        final data = jsonDecode(message) as Map<String, dynamic>;
        final type = data['type'] as String?;
        
        switch (type) {
          case 'session_start':
            _sessionId = data['session_id'] as String?;
            _log("🔗 [ASR STREAMING] Session started: $_sessionId");
            _emitSystem('Connected to server');
            break;
            
          case 'transcript':
            // Real-time transcript (no speaker yet)
            final segment = data['segment'] as Map<String, dynamic>?;
            if (segment != null) {
              final text = segment['text'] as String? ?? '';
              final speaker = segment['speaker'] as String? ?? 'UNKNOWN';
              final isFinal = data['is_final'] as bool? ?? false;
              
              if (text.isNotEmpty) {
                _log("📝 [ASR STREAMING] Transcript: '$text' (speaker: $speaker, final: $isFinal)");
                _events.add(AsrStreamEvent(
                  type: isFinal ? 'final' : 'partial',
                  text: text,
                  speaker: speaker,
                  startTime: segment['start'] as double? ?? 0,
                  endTime: segment['end'] as double? ?? 0,
                ));
              }
            }
            break;
            
          case 'diarization_update':
            // Speaker labels updated after diarization
            final segments = data['segments'] as List<dynamic>? ?? [];
            _log("👥 [ASR STREAMING] Diarization update: ${segments.length} segments");
            
            for (final seg in segments) {
              final segMap = seg as Map<String, dynamic>;
              _events.add(AsrStreamEvent(
                type: 'diarization_update',
                text: segMap['text'] as String? ?? '',
                speaker: segMap['speaker'] as String? ?? 'UNKNOWN',
                startTime: segMap['start'] as double? ?? 0,
                endTime: segMap['end'] as double? ?? 0,
                emotion: segMap['emotion'] as Map<String, dynamic>?,
              ));
            }
            break;
            
          case 'error':
            final errorMsg = data['message'] as String? ?? 'Unknown error';
            _log("❌ [ASR STREAMING] Server error: $errorMsg");
            _emitSystem('Server error: $errorMsg', isError: true);
            break;
            
          case 'pong':
            // Ping response - connection alive
            break;
            
          default:
            _log("📋 [ASR STREAMING] Unknown message type: $type");
        }
      }
    } catch (e) {
      _log("❌ [ASR STREAMING] Error handling message: $e");
    }
  }

  Future<void> stop() async {
    if (!_running) return;
    
    _running = false;
    
    // Stop audio recording
    await _audioSubscription?.cancel();
    _audioSubscription = null;
    
    try {
      await _recorder.stop();
    } catch (_) {}
    
    // Send end session message
    if (_channel != null) {
      try {
        _channel!.sink.add(jsonEncode({'type': 'end_session'}));
        await Future.delayed(const Duration(milliseconds: 200));
        await _channel!.sink.close();
      } catch (_) {}
      _channel = null;
    }
    
    _sessionId = null;
    _emitSystem('Streaming stopped');
  }

  /// Force diarization update (can be called manually)
  void requestDiarization() {
    if (_channel != null && _running) {
      _channel!.sink.add(jsonEncode({'type': 'force_diarize'}));
      _log("🔄 [ASR STREAMING] Requested diarization");
    }
  }

  Future<bool> _ensurePerms() async {
    final mic = await Permission.microphone.request();
    return mic.isGranted;
  }

  void _emitSystem(String message, {bool isError = false}) {
    AppLogger.instance.log('ASR_STREAM', message, isError: isError);
    if (_events.isClosed) return;
    _events.add(AsrStreamEvent(
      type: isError ? 'error' : 'system',
      text: message,
      speaker: 'SYSTEM',
    ));
  }
}

/// Event emitted by the streaming ASR service
class AsrStreamEvent {
  final String type;  // 'partial', 'final', 'diarization_update', 'system', 'error'
  final String text;
  final String speaker;
  final double startTime;
  final double endTime;
  final Map<String, dynamic>? emotion;
  final String? error;

  AsrStreamEvent({
    required this.type,
    required this.text,
    required this.speaker,
    this.startTime = 0,
    this.endTime = 0,
    this.emotion,
    this.error,
  });
  
  bool get isPartial => type == 'partial';
  bool get isFinal => type == 'final';
  bool get isDiarizationUpdate => type == 'diarization_update';
  bool get isSystem => type == 'system';
  bool get isError => type == 'error';
}
