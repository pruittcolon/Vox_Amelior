import 'package:record/record.dart';
import 'package:audio_session/audio_session.dart';

class MicService {
  final AudioRecorder _recorder = AudioRecorder();
  bool _sessionConfigured = false;

  /// Configure audio session to not pause other media
  Future<void> _configureAudioSession() async {
    if (_sessionConfigured) return;
    
    final session = await AudioSession.instance;
    await session.configure(AudioSessionConfiguration(
      avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
      avAudioSessionCategoryOptions: AVAudioSessionCategoryOptions.mixWithOthers |
          AVAudioSessionCategoryOptions.defaultToSpeaker,
      avAudioSessionMode: AVAudioSessionMode.voiceChat,
      androidAudioAttributes: const AndroidAudioAttributes(
        contentType: AndroidAudioContentType.speech,
        usage: AndroidAudioUsage.voiceCommunication,
      ),
      androidAudioFocusGainType: AndroidAudioFocusGainType.gainTransientMayDuck,
    ));
    _sessionConfigured = true;
  }

  /// Record for a fixed number of seconds
  Future<String?> recordForSeconds(int seconds) async {
    if (await _recorder.hasPermission()) {
      await _configureAudioSession();
      
      final path =
          '/sdcard/Download/recording_${DateTime.now().millisecondsSinceEpoch}.m4a';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.aacLc,
          bitRate: 128000,
          sampleRate: 16000,
        ),
        path: path,
      );

      await Future.delayed(Duration(seconds: seconds));
      return await _recorder.stop();
    }
    return null;
  }

  /// Start continuous recording
  Future<void> startRecording(String path) async {
    if (await _recorder.hasPermission()) {
      await _configureAudioSession();
      
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.aacLc,
          bitRate: 128000,
          sampleRate: 16000,
        ),
        path: path,
      );
    }
  }

  /// Stop recording and return file path
  Future<String?> stopRecording() async {
    return await _recorder.stop();
  }

  /// Check if recording is active
  Future<bool> isRecording() async {
    return await _recorder.isRecording();
  }
}
