/// Which on-device model a feature needs.
enum ModelKind { speechToText, voiceActivity, speakerVoiceprint, speakerTurns, languageModel }

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
/// Speech: NVIDIA Parakeet RNNT 1.1B (the Vox server's model, int8 sherpa-onnx
/// export built by this repo's CI), Silero VAD and
/// TitaNet voiceprints. Assistant: Google Gemma 4 E4B for LiteRT-LM.
class ModelCatalog {
  const ModelCatalog._();

  static const String _sherpa = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';
  static const String _parakeet =
      'https://github.com/pruittcolon/Vox_Amelior/releases/download/parakeet-rnnt-1.1b-int8';

  static const ModelAsset parakeet = ModelAsset(
    id: 'parakeet-rnnt-1.1b-int8',
    kind: ModelKind.speechToText,
    title: 'Parakeet 1.1B speech recognition',
    description: 'NVIDIA Parakeet RNNT 1.1B (int8). Turns speech into text on your phone.',
    approxDownloadBytes: 1118000000,
    files: [
      RemoteFile(
        url: '$_parakeet/encoder.int8.onnx',
        fileName: 'encoder.int8.onnx',
        sha256: 'f6a9b8ecbf0e62423d4c1e3cb111d5edaddf9aaceff0e89bf932732cd09c73e8',
        sizeBytes: 43658261,
      ),
      RemoteFile(
        url: '$_parakeet/encoder.int8.weights',
        fileName: 'encoder.int8.weights',
        sha256: '2a0e6f3868cbd75d785a9e0b04f8b036dfd8882d6bbe7fbabf4e0e89200df00e',
        sizeBytes: 1065281280,
      ),
      RemoteFile(
        url: '$_parakeet/decoder.int8.onnx',
        fileName: 'decoder.int8.onnx',
        sha256: 'c254a0f48b94e13c7982bd5454ffffafa9aea4d5a1ceec9ad45ec0b84848e6bb',
        sizeBytes: 7257753,
      ),
      RemoteFile(
        url: '$_parakeet/joiner.int8.onnx',
        fileName: 'joiner.int8.onnx',
        sha256: 'da6b8f93c0922c987d12bb9b0bf4c2716c93750dd97725ec7824198d1b0196da',
        sizeBytes: 1735860,
      ),
      RemoteFile(
        url: '$_parakeet/tokens.txt',
        fileName: 'tokens.txt',
        sha256: 'ed16e1a4e3a3aa379138c0b1888e5d49f993c9d512b2be4d46e90a87afd54921',
        sizeBytes: 10374,
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

  /// Needed before listening can start.
  static const List<ModelAsset> speech = [parakeet, voiceActivity, speakerVoiceprint];

  /// Downloaded with the speech models, but listening works without them.
  static const List<ModelAsset> speechExtras = [diarizer];
  static const List<ModelAsset> assistants = [gemma4E4b, gemma4E2b];
  static const List<ModelAsset> all = [...speech, ...speechExtras, ...assistants];

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
