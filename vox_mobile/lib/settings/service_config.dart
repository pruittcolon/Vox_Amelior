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
    this.diarizer,
    this.toneModel,
    this.toneTokens,
  });

  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
  final String vad;
  final String speaker;

  /// Speaker-change model (optional: lines are simply not split without it).
  final String? diarizer;

  /// Tone-of-voice model and its tokens (optional: lines get no tone without it).
  final String? toneModel;
  final String? toneTokens;

  /// Null unless the speech models and at least one recognizer are fully
  /// installed. [asr] is the recognizer the user chose; when it is not
  /// installed the other one (fp16 first) is used.
  static SpeechModelPaths? fromStore(ModelStore store, {ModelAsset? asr}) {
    if (!ModelCatalog.speechReady(store.isInstalled)) return null;
    String path(ModelAsset a, String name) => store.file(a, name).path;
    final model = asr != null && store.isInstalled(asr) ? asr : ModelCatalog.recognizers.firstWhere(store.isInstalled);
    String named(String prefix) => path(model, model.installedFileNames.firstWhere((n) => n.startsWith(prefix)));
    final tone = store.isInstalled(ModelCatalog.toneModel);
    return SpeechModelPaths(
      encoder: named('encoder'),
      decoder: named('decoder'),
      joiner: named('joiner'),
      tokens: named('tokens'),
      vad: path(ModelCatalog.voiceActivity, 'silero_vad.onnx'),
      speaker: path(ModelCatalog.speakerVoiceprint, 'nemo_en_titanet_small.onnx'),
      diarizer: store.isInstalled(ModelCatalog.diarizer) ? path(ModelCatalog.diarizer, 'diarizer.int8.onnx') : null,
      toneModel: tone ? path(ModelCatalog.toneModel, 'model.int8.onnx') : null,
      toneTokens: tone ? path(ModelCatalog.toneModel, 'tokens.txt') : null,
    );
  }

  /// Required model files that are not on disk (checked before native code
  /// loads them: a missing file can end the whole process there).
  List<String> missingFiles() => [
        for (final f in [encoder, decoder, joiner, tokens, vad, speaker])
          if (!File(f).existsSync()) f,
      ];

  Map<String, Object?> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'joiner': joiner,
        'tokens': tokens,
        'vad': vad,
        'speaker': speaker,
        'diarizer': diarizer,
        'toneModel': toneModel,
        'toneTokens': toneTokens,
      };

  factory SpeechModelPaths.fromJson(Map<String, Object?> j) => SpeechModelPaths(
        encoder: j['encoder']! as String,
        decoder: j['decoder']! as String,
        joiner: j['joiner']! as String,
        tokens: j['tokens']! as String,
        vad: j['vad']! as String,
        speaker: j['speaker']! as String,
        diarizer: j['diarizer'] as String?,
        toneModel: j['toneModel'] as String?,
        toneTokens: j['toneTokens'] as String?,
      );
}

/// The assistant model file and how to run it.
class LlmConfig {
  const LlmConfig({required this.path, required this.modelType, required this.supportsTools, required this.title});

  final String path;
  final String modelType;
  final bool supportsTools;
  final String title;

  static LlmConfig? fromStore(ModelStore store, ModelAsset asset) {
    if (!store.isInstalled(asset)) return null;
    return LlmConfig(
      path: store.file(asset, asset.files.first.fileName).path,
      modelType: asset.llmType,
      supportsTools: asset.supportsTools,
      title: asset.title,
    );
  }

  Map<String, Object?> toJson() => {'path': path, 'modelType': modelType, 'supportsTools': supportsTools, 'title': title};

  static LlmConfig? fromJson(Object? j) {
    if (j is! Map || j['path'] is! String) return null;
    return LlmConfig(
      path: j['path']! as String,
      modelType: j['modelType'] as String? ?? 'gemma4',
      supportsTools: j['supportsTools'] as bool? ?? false,
      title: j['title'] as String? ?? 'Assistant',
    );
  }
}

/// Everything the background service needs, handed over as a JSON file
/// (the UI and the service run in different isolates and share only disk).
class ServiceConfig {
  const ServiceConfig({
    required this.dbPath,
    required this.paths,
    required this.settings,
    required this.queueDir,
    this.llm,
    this.embeddingModelId = 'nemo-titanet-small',
  });

  final String dbPath;
  final SpeechModelPaths paths;
  final AppSettings settings;

  /// Where captured speech waits to be transcribed.
  final String queueDir;

  /// Null when no assistant model is installed.
  final LlmConfig? llm;
  final String embeddingModelId;

  Map<String, Object?> toJson() => {
        'dbPath': dbPath,
        'paths': paths.toJson(),
        'settings': settings.toJson(),
        'queueDir': queueDir,
        'llm': llm?.toJson(),
        'embeddingModelId': embeddingModelId,
      };

  factory ServiceConfig.fromJson(Map<String, Object?> j) => ServiceConfig(
        dbPath: j['dbPath']! as String,
        paths: SpeechModelPaths.fromJson(j['paths']! as Map<String, Object?>),
        settings: AppSettings.fromJson(j['settings']! as Map<String, Object?>),
        queueDir: j['queueDir'] as String? ?? '${File(j['dbPath']! as String).parent.path}/speech_queue',
        llm: LlmConfig.fromJson(j['llm']),
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
