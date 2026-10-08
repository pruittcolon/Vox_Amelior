/// Which on-device model a feature needs.
enum ModelKind { speechToText, voiceActivity, speakerVoiceprint, speakerTurns, toneOfVoice, languageModel }

/// One file to fetch for a model.
class RemoteFile {
  const RemoteFile({
    required this.url,
    required this.fileName,
    this.sha256,
    this.sizeBytes,
    this.extractFromArchive = const {},
  });

  final String url;

  /// Name the file is stored under (for archives: the archive's own name).
  final String fileName;

  /// Expected SHA-256 of the *downloaded* file, when known.
  final String? sha256;
  final int? sizeBytes;

  /// If non-empty, [fileName] is a `.tar.bz2` and only these files are kept.
  final Set<String> extractFromArchive;

  bool get isArchive => extractFromArchive.isNotEmpty;
}

class ModelAsset {
  const ModelAsset({
    required this.id,
    required this.kind,
    required this.title,
    required this.description,
    required this.files,
    required this.approxDownloadBytes,
    this.requiresToken = false,
    this.licenseUrl,
    this.essential = true,
    this.llmType = 'gemma4',
    this.supportsTools = false,
  });

  final String id;
  final ModelKind kind;
  final String title;
  final String description;
  final List<RemoteFile> files;
  final int approxDownloadBytes;

  /// Gated on Hugging Face: the user must accept a licence and give a token.
  final bool requiresToken;
  final String? licenseUrl;

  /// Needed before listening can start.
  final bool essential;

  /// For language models: flutter_gemma model family ('gemma4', 'gemmaIt', ...).
  final String llmType;

  /// For language models: understands tool calls (agent features).
  final bool supportsTools;

  /// Names of the files that must exist after installation.
  Set<String> get installedFileNames => {
        for (final f in files) ...(f.isArchive ? f.extractFromArchive : {f.fileName}),
      };
}

/// The models the app downloads.
///
/// Speech: NVIDIA Parakeet TDT 0.6B v2 (the ready-made sherpa-onnx exports in
/// fp16 and int8; it writes punctuation and capitals), Silero VAD and
/// TitaNet voiceprints. The fp16 recognizer is downloaded by default; the
/// int8 one only when the user picks it. Assistant: Google Gemma 4 E4B for LiteRT-LM.
class ModelCatalog {
  const ModelCatalog._();

  static const String _sherpa = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';

  static const ModelAsset parakeet = ModelAsset(
    id: 'parakeet-tdt-0.6b-v2-int8',
    kind: ModelKind.speechToText,
    title: 'Parakeet speech recognition',
    description: 'NVIDIA Parakeet TDT 0.6B (int8). Smaller and lighter on memory; downloads only if you choose it.',
    approxDownloadBytes: 482468385,
    essential: false,
    files: [
      RemoteFile(
        url: '$_sherpa/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2',
        fileName: 'parakeet.tar.bz2',
        sha256: '157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad',
        sizeBytes: 482468385,
        extractFromArchive: {'encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'},
      ),
    ],
  );

  /// The same model in half precision: no quantization. The default
  /// recognizer, and the first model downloaded at setup.
  static const ModelAsset parakeetFp16 = ModelAsset(
    id: 'parakeet-tdt-0.6b-v2-fp16',
    kind: ModelKind.speechToText,
    title: 'Parakeet speech recognition (fp16)',
    description: 'NVIDIA Parakeet TDT 0.6B in half precision (fp16). Turns speech into text on your phone.',
    approxDownloadBytes: 1120982957,
    files: [
      RemoteFile(
        url: '$_sherpa/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2',
        fileName: 'parakeet-fp16.tar.bz2',
        sha256: '37f67a1a6c942dae27d345ee395fbd19e25ee48996faf70fca25779026054cf0',
        sizeBytes: 1120982957,
        extractFromArchive: {'encoder.fp16.onnx', 'decoder.fp16.onnx', 'joiner.fp16.onnx', 'tokens.txt'},
      ),
    ],
  );

  static const ModelAsset voiceActivity = ModelAsset(
    id: 'silero-vad',
    kind: ModelKind.voiceActivity,
    title: 'Speech detector',
    description: 'Silero VAD. Notices when someone is talking so the phone can rest in silence.',
    approxDownloadBytes: 643854,
    files: [
      RemoteFile(
        url: '$_sherpa/asr-models/silero_vad.onnx',
        fileName: 'silero_vad.onnx',
        sha256: '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6',
        sizeBytes: 643854,
      ),
    ],
  );

  static const ModelAsset speakerVoiceprint = ModelAsset(
    id: 'titanet-small',
    kind: ModelKind.speakerVoiceprint,
    title: 'Voice recognition',
    description: 'NVIDIA TitaNet Small. Learns who is speaking so transcripts show names.',
    approxDownloadBytes: 40257283,
    files: [
      RemoteFile(
        url: '$_sherpa/speaker-recongition-models/nemo_en_titanet_small.onnx',
        fileName: 'nemo_en_titanet_small.onnx',
        sha256: 'ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e',
        sizeBytes: 40257283,
      ),
    ],
  );

  /// NVIDIA's diarizer for Parakeet, converted to ONNX by this repo's CI.
  static const String _diarizer =
      'https://github.com/pruittcolon/Vox_Amelior/releases/download/nemotron-3-diarization-onnx';

  static const ModelAsset diarizer = ModelAsset(
    id: 'nemotron-3-diarization',
    kind: ModelKind.speakerTurns,
    title: 'Speaker changes',
    description: 'NVIDIA Nemotron 3 Diarization (Sortformer family). Splits quick back-and-forth into separate lines '
        'and marks people talking at the same time.',
    approxDownloadBytes: _diarizerBytes,
    essential: false,
    files: [
      RemoteFile(
        url: '$_diarizer/diarizer.int8.onnx',
        fileName: 'diarizer.int8.onnx',
        sha256: _diarizerSha,
        sizeBytes: _diarizerBytes,
      ),
    ],
  );
  static const int _diarizerBytes = 107759677;
  static const String _diarizerSha = 'f468ec639d5cd4c9df925b4f54398d68d23fdc8ebc171e6cda1f3d5e0b281886';

  /// SenseVoice Small (FunAudioLLM), sherpa-onnx int8 export. Only its
  /// emotion and sound tags are used.
  static const ModelAsset toneModel = ModelAsset(
    id: 'sense-voice-small-int8',
    kind: ModelKind.toneOfVoice,
    title: 'Tone of voice',
    description: 'SenseVoice Small. Hears how each line was said (happy, sad, angry, ...) and sounds like laughter '
        'or music, so you can sort conversations by mood. Optional.',
    approxDownloadBytes: 163002883,
    essential: false,
    licenseUrl: 'https://github.com/FunAudioLLM/SenseVoice/blob/main/MODEL_LICENSE',
    files: [
      RemoteFile(
        url: '$_sherpa/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2',
        fileName: 'sense-voice.tar.bz2',
        sha256: '7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e',
        sizeBytes: 163002883,
        extractFromArchive: {'model.int8.onnx', 'tokens.txt'},
      ),
    ],
  );

  static const String _lc = 'https://huggingface.co/litert-community';

  static const ModelAsset gemma4E4b = ModelAsset(
    id: 'gemma-4-e4b-it',
    kind: ModelKind.languageModel,
    title: 'Gemma 4 E4B',
    description: 'Google Gemma 4 (E4B) for LiteRT-LM. Answers questions and can take actions. Best quality; needs ~6 GB RAM.',
    approxDownloadBytes: 3659530240,
    licenseUrl: '$_lc/gemma-4-E4B-it-litert-lm',
    essential: false,
    llmType: 'gemma4',
    supportsTools: true,
    files: [
      RemoteFile(
        url: '$_lc/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm',
        fileName: 'gemma-4-E4B-it.litertlm',
        sha256: '0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0',
        sizeBytes: 3659530240,
      ),
    ],
  );

  static const ModelAsset gemma4E2b = ModelAsset(
    id: 'gemma-4-e2b-it',
    kind: ModelKind.languageModel,
    title: 'Gemma 4 E2B (faster)',
    description: 'Smaller Gemma 4 for quicker answers and phones with less memory.',
    approxDownloadBytes: 2588147712,
    licenseUrl: '$_lc/gemma-4-E2B-it-litert-lm',
    essential: false,
    llmType: 'gemma4',
    supportsTools: true,
    files: [
      RemoteFile(
        url: '$_lc/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
        fileName: 'gemma-4-E2B-it.litertlm',
        sha256: '181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c',
        sizeBytes: 2588147712,
      ),
    ],
  );

  /// Downloaded at setup, in this order (the fp16 recognizer first).
  static const List<ModelAsset> speech = [parakeetFp16, voiceActivity, speakerVoiceprint];

  /// Needed before listening can start, together with any one of [recognizers].
  static const List<ModelAsset> speechSupport = [voiceActivity, speakerVoiceprint];

  /// Downloaded with the speech models, but listening works without them.
  static const List<ModelAsset> speechExtras = [diarizer];

  /// Speech-recognition models to choose from (settings `speechModel`: 'fp16' or 'int8'),
  /// in order of preference when the chosen one is not installed.
  static const List<ModelAsset> recognizers = [parakeetFp16, parakeet];
  static const List<ModelAsset> assistants = [gemma4E4b, gemma4E2b];
  static const List<ModelAsset> all = [...speech, parakeet, ...speechExtras, toneModel, ...assistants];

  /// Listening can start: the support models plus either recognizer are installed.
  static bool speechReady(bool Function(ModelAsset asset) isInstalled) =>
      speechSupport.every(isInstalled) && recognizers.any(isInstalled);

  static const String customLlmId = 'custom-llm';

  /// A user-supplied LiteRT-LM model (any `.litertlm` URL).
  static ModelAsset custom({
    required String url,
    String name = 'Custom model',
    String llmType = 'gemma4',
    bool supportsTools = true,
    bool requiresToken = false,
  }) {
    final path = Uri.tryParse(url)?.pathSegments.where((s) => s.isNotEmpty).lastOrNull;
    final fileName = (path == null || !path.contains('.')) ? 'custom.litertlm' : path;
    return ModelAsset(
      id: customLlmId,
      kind: ModelKind.languageModel,
      title: name.trim().isEmpty ? 'Custom model' : name.trim(),
      description: url,
      approxDownloadBytes: 0,
      essential: false,
      llmType: llmType,
      supportsTools: supportsTools,
      requiresToken: requiresToken,
      files: [RemoteFile(url: url, fileName: fileName)],
    );
  }
}
