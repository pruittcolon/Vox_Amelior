import 'dart:convert';
import 'dart:io';

import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';

/// Paths to the speech models, once installed.
class SpeechModelPaths {
  const SpeechModelPaths({
    required this.encoder,
    required this.decoder,
    required this.joiner,
    required this.tokens,
    required this.vad,
    required this.speaker,
  });

  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
  final String vad;
  final String speaker;

  /// Null unless every speech model is fully installed.
  static SpeechModelPaths? fromStore(ModelStore store) {
    const needed = [ModelCatalog.parakeet, ModelCatalog.voiceActivity, ModelCatalog.speakerVoiceprint];
    if (!needed.every(store.isInstalled)) return null;
    String path(ModelAsset a, String name) => store.file(a, name).path;
    return SpeechModelPaths(
      encoder: path(ModelCatalog.parakeet, 'encoder.int8.onnx'),
      decoder: path(ModelCatalog.parakeet, 'decoder.int8.onnx'),
      joiner: path(ModelCatalog.parakeet, 'joiner.int8.onnx'),
      tokens: path(ModelCatalog.parakeet, 'tokens.txt'),
      vad: path(ModelCatalog.voiceActivity, 'silero_vad.onnx'),
      speaker: path(ModelCatalog.speakerVoiceprint, 'nemo_en_titanet_small.onnx'),
    );
  }

  Map<String, Object?> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'joiner': joiner,
        'tokens': tokens,
        'vad': vad,
        'speaker': speaker,
      };

  factory SpeechModelPaths.fromJson(Map<String, Object?> j) => SpeechModelPaths(
        encoder: j['encoder']! as String,
        decoder: j['decoder']! as String,
        joiner: j['joiner']! as String,
        tokens: j['tokens']! as String,
        vad: j['vad']! as String,
        speaker: j['speaker']! as String,
      );
}

/// Everything the background listener needs, handed over as a JSON file
/// (the UI and the service run in different isolates and share only disk).
class ServiceConfig {
  const ServiceConfig({
    required this.dbPath,
    required this.paths,
    required this.settings,
    this.embeddingModelId = 'nemo-titanet-small',
  });

  final String dbPath;
  final SpeechModelPaths paths;
  final AppSettings settings;
  final String embeddingModelId;

  Map<String, Object?> toJson() => {
        'dbPath': dbPath,
        'paths': paths.toJson(),
        'settings': settings.toJson(),
        'embeddingModelId': embeddingModelId,
      };

  factory ServiceConfig.fromJson(Map<String, Object?> j) => ServiceConfig(
        dbPath: j['dbPath']! as String,
        paths: SpeechModelPaths.fromJson(j['paths']! as Map<String, Object?>),
        settings: AppSettings.fromJson(j['settings']! as Map<String, Object?>),
        embeddingModelId: j['embeddingModelId'] as String? ?? 'nemo-titanet-small',
      );

  /// Writes atomically so the service never reads a half-written file.
  void writeTo(File file) {
    file.parent.createSync(recursive: true);
    final tmp = File('${file.path}.tmp')..writeAsStringSync(jsonEncode(toJson()), flush: true);
    tmp.renameSync(file.path);
  }

  static ServiceConfig? readFrom(File file) {
    if (!file.existsSync()) return null;
    try {
      return ServiceConfig.fromJson(jsonDecode(file.readAsStringSync()) as Map<String, Object?>);
    } on Object {
      return null;
    }
  }
}
