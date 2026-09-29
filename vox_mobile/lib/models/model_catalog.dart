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
    this.approxDiskBytes,
    this.requiresToken = false,
    this.licenseUrl,
    this.essential = true,
  });

  final String id;
  final ModelKind kind;
  final String title;
  final String description;
  final List<RemoteFile> files;
  final int approxDownloadBytes;

  /// Space needed once unpacked, when larger than the download.
  final int? approxDiskBytes;

  /// Gated on Hugging Face: the user must accept a licence and give a token.
  final bool requiresToken;
  final String? licenseUrl;

  /// Needed before listening can start.
  final bool essential;

  /// Names of the files that must exist after installation.
  Set<String> get installedFileNames => {
        for (final f in files) ...(f.isArchive ? f.extractFromArchive : {f.fileName}),
      };

  ModelAsset withFiles(List<RemoteFile> newFiles) => ModelAsset(
        id: id,
        kind: kind,
        title: title,
        description: description,
        files: newFiles,
        approxDownloadBytes: approxDownloadBytes,
        approxDiskBytes: approxDiskBytes,
        requiresToken: requiresToken,
        licenseUrl: licenseUrl,
        essential: essential,
      );
}

/// The models the app can download. Speech and voice models come from the
/// public sherpa-onnx releases; Gemma comes from Google's Hugging Face repo.
class ModelCatalog {
  const ModelCatalog._();

  static const String _sherpa = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';

  static const ModelAsset parakeet = ModelAsset(
    id: 'parakeet-tdt-0.6b-v2-int8',
    kind: ModelKind.speechToText,
    title: 'Parakeet speech recognition',
    description: 'NVIDIA Parakeet TDT 0.6B (int8). Turns speech into text on your phone.',
    approxDownloadBytes: 482468385,
    approxDiskBytes: 661190513,
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

  static const String _hf = 'https://huggingface.co/google';

  static const ModelAsset gemma3nE4b = ModelAsset(
    id: 'gemma-3n-e4b-it-int4',
    kind: ModelKind.languageModel,
    title: 'Gemma 3n E4B (assistant)',
    description: 'Google Gemma 3n, 4-bit. Answers questions about your conversations. Needs about 6 GB of free RAM.',
    approxDownloadBytes: 4919541760,
    requiresToken: true,
    licenseUrl: '$_hf/gemma-3n-E4B-it-litert-lm',
    essential: false,
    files: [
      RemoteFile(
        url: '$_hf/gemma-3n-E4B-it-litert-lm/resolve/main/gemma-3n-E4B-it-int4.litertlm',
        fileName: 'gemma-3n-E4B-it-int4.litertlm',
        // Integrity is also checked against Hugging Face's x-linked-etag (SHA-256).
        sizeBytes: 4919541760,
      ),
    ],
  );

  static const ModelAsset gemma3nE2b = ModelAsset(
    id: 'gemma-3n-e2b-it-int4',
    kind: ModelKind.languageModel,
    title: 'Gemma 3n E2B (lighter assistant)',
    description: 'Smaller and faster Gemma 3n for phones with less memory.',
    approxDownloadBytes: 3655827456,
    requiresToken: true,
    licenseUrl: '$_hf/gemma-3n-E2B-it-litert-lm',
    essential: false,
    files: [
      RemoteFile(
        url: '$_hf/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm',
        fileName: 'gemma-3n-E2B-it-int4.litertlm',
        sizeBytes: 3655827456,
      ),
    ],
  );

  static const List<ModelAsset> all = [parakeet, voiceActivity, speakerVoiceprint, gemma3nE4b, gemma3nE2b];

  static ModelAsset byId(String id) => all.firstWhere((m) => m.id == id);
}
