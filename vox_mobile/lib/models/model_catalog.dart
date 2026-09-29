/// Which on-device model a feature needs.
enum ModelKind { speechToText, voiceActivity, speakerVoiceprint, languageModel }

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
/// Speech: NVIDIA Parakeet TDT 1.1B (int8, sherpa-onnx export), Silero VAD and
/// TitaNet voiceprints. Assistant: Google Gemma 4 E4B for LiteRT-LM.
class ModelCatalog {
  const ModelCatalog._();

  static const String _sherpa = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';
  static const String _parakeet = 'https://huggingface.co/mortenfc/sherpa-onnx-parakeet-tdt-1.1b-int8/resolve/main';

  static const ModelAsset parakeet = ModelAsset(
    id: 'parakeet-tdt-1.1b-int8',
    kind: ModelKind.speechToText,
    title: 'Parakeet 1.1B speech recognition',
    description: 'NVIDIA Parakeet TDT 1.1B (int8). Turns speech into text on your phone.',
    approxDownloadBytes: 1117000000,
    files: [
      RemoteFile(
        url: '$_parakeet/encoder.int8.onnx',
        fileName: 'encoder.int8.onnx',
        sha256: 'a8874229954fc2c21b01a4d085fdd22bb6e07fc1feb01bd608df4b3d2ca439f6',
        sizeBytes: 42680013,
      ),
      RemoteFile(
        url: '$_parakeet/encoder.int8.weights',
        fileName: 'encoder.int8.weights',
        sha256: '04f27664e6d9e2e7fa7c2d8ceb8bd99336bc694a3f2d53d3dbb69c8f65dd163e',
        sizeBytes: 1065283320,
      ),
      RemoteFile(
        url: '$_parakeet/decoder.int8.onnx',
        fileName: 'decoder.int8.onnx',
        sha256: 'c687095464c7efde24a4f1e9d1b20ac5c9b9356ee1324b2e1e43da717d7954ce',
        sizeBytes: 7258064,
      ),
      RemoteFile(
        url: '$_parakeet/joiner.int8.onnx',
        fileName: 'joiner.int8.onnx',
        sha256: '80ac06ea5b342624647bf5024c0fdf1171bf5bc51edd332cc8096ce077c75eaa',
        sizeBytes: 1739391,
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

  static const List<ModelAsset> speech = [parakeet, voiceActivity, speakerVoiceprint];
  static const List<ModelAsset> assistants = [gemma4E4b, gemma4E2b];
  static const List<ModelAsset> all = [...speech, ...assistants];

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
