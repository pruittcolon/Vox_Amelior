/// What can be downloaded: the voice packs and the Voice Gallery.
library;

import 'dart:convert';

import 'catalog_data.dart';

const releaseBaseUrl = 'https://github.com/pruittcolon/Vox_Amelior/releases/download/narrator-models';

class PackFile {
  const PackFile(this.asset, this.path, this.size, this.sha256, {this.unzip = false});
  final String asset;
  final String path;
  final int size;
  final String sha256;
  final bool unzip;
}

class VoicePack {
  const VoicePack({
    required this.id,
    required this.title,
    required this.summary,
    required this.license,
    required this.files,
    this.required = false,
  });

  final String id;
  final String title;
  final String summary;
  final String license;
  final List<PackFile> files;
  final bool required;

  int get bytes => files.fold(0, (a, f) => a + f.size);

  /// Changes whenever any file changes, so an updated pack re-downloads.
  String get fingerprint => files.map((f) => f.sha256.substring(0, 12)).join('.');
}

/// Kokoro files both variants need (Mandarin front-end data is skipped).
const _kokoroShared = {'voices.bin', 'tokens.txt', 'lexicon-us-en.txt', 'lexicon-gb-en.txt', 'espeak-ng-data'};

List<PackFile> _files(String pack, {Set<String>? only}) => [
      for (final f in packFileData[pack] ?? const <PackFile>[])
        if (only == null || only.contains(f.path)) f,
    ];

final voicePacks = <String, VoicePack>{
  'kokoro': VoicePack(
    id: 'kokoro',
    title: 'Kokoro voices',
    summary: '54 natural voices: American, British and accented English. The main voice pack.',
    license: 'Apache-2.0 (hexgrad/Kokoro-82M)',
    files: [..._files('kokoro', only: _kokoroShared), ..._files('kokoro-fp32', only: {'model.onnx'})],
    required: true,
  ),
  'kokoro-lite': VoicePack(
    id: 'kokoro-lite',
    title: 'Kokoro voices (compact)',
    summary: 'The same 54 voices in a smaller download, but about 3x slower to generate. '
        'Only for phones short on space.',
    license: 'Apache-2.0 (hexgrad/Kokoro-82M)',
    files: _files('kokoro', only: {..._kokoroShared, 'model.int8.onnx'}),
  ),
  // Full precision (HD): the int8 versions quantize the vocoder and score
  // clearly lower in the voice lab (tool/lab/voicelab.py).
  'supertonic': VoicePack(
    id: 'supertonic',
    title: 'Studio voices (Supertonic HD)',
    summary: '10 studio-quality voices at full 44 kHz: the crispest and most natural in the app, '
        'and the cast of the podcast characters. As fast as Kokoro.',
    license: 'OpenRAIL model (Supertone), MIT code',
    files: [
      ..._files('supertonic', only: {'tts.json', 'unicode_indexer.bin', 'voice.bin', 'LICENSE'}),
      ..._files('supertonic-fp32'),
    ],
  ),
  'pocket': VoicePack(
    id: 'pocket',
    title: 'Voice cloning (Pocket TTS HD)',
    summary: 'Speaks in any voice from a 10-second clip: the Voice Gallery\'s hundreds of '
        'donated voices, or your own recording. Slower than real time on most phones: export, then listen.',
    license: 'CC-BY-4.0 (kyutai/pocket-tts)',
    files: [
      ..._files('pocket', only: {'encoder.onnx', 'text_conditioner.onnx', 'vocab.json', 'token_scores.json', 'test_wavs/bria.wav'}),
      ..._files('pocket-fp32'),
    ],
  ),
  'clarity': VoicePack(
    id: 'clarity',
    title: 'Clarity (HD sound)',
    summary: 'Cloned voices stop at 8 kHz, which sounds dull. Clarity rebuilds their crisp top end '
        'for full 48 kHz sound. Small download; get it with voice cloning.',
    license: 'Apache-2.0 (LavaSR by ysharma3501; ONNX export by Topping1)',
    files: _files('clarity'),
  ),
};

class GalleryVoice {
  const GalleryVoice({
    required this.id,
    required this.set,
    required this.name,
    required this.asset,
    required this.size,
    required this.sha256,
    required this.seconds,
    required this.tone,
    required this.license,
    required this.source,
    this.gender,
    this.age,
    this.accent,
    this.region,
  });

  final String id;
  final String set;
  final String name;
  final String asset;
  final int size;
  final String sha256;
  final double seconds;
  final String tone;
  final String license;
  final String source;
  final String? gender;
  final String? age;
  final String? accent;
  final String? region;

  String get voiceId => 'gallery:$id';

  String get description {
    final parts = <String>[
      if (accent != null && accent!.isNotEmpty) accent!,
      if (region != null && region!.isNotEmpty && region != accent) region!,
      if (gender == 'F') 'female' else if (gender == 'M') 'male',
      if (tone != 'unknown') '$tone voice',
    ];
    return parts.isEmpty ? source : parts.join(' · ');
  }

  /// Sets shown first: hand-picked narrators and characters.
  int get rank => switch (set) {
        'voice-zero' => 0,
        'alba-mackenna' => 1,
        'unmute-prod-website' => 2,
        'vctk' => 3,
        _ => 4,
      };
}

List<GalleryVoice> parseGallery(String json) {
  final data = jsonDecode(json) as Map<String, Object?>;
  final voices = (data['voices'] as List<Object?>).cast<Map<String, Object?>>();
  final list = [
    for (final v in voices)
      GalleryVoice(
        id: v['id']! as String,
        set: v['set']! as String,
        name: v['name']! as String,
        asset: v['asset']! as String,
        size: (v['size']! as num).toInt(),
        sha256: v['sha256']! as String,
        seconds: (v['seconds']! as num).toDouble(),
        tone: (v['tone'] as String?) ?? 'unknown',
        license: v['license']! as String,
        source: v['source']! as String,
        gender: v['gender'] as String?,
        age: v['age']?.toString(),
        accent: v['accent'] as String?,
        region: v['region'] as String?,
      ),
  ];
  list.sort((a, b) => a.rank != b.rank ? a.rank.compareTo(b.rank) : a.name.compareTo(b.name));
  return list;
}

String formatBytes(int bytes) {
  if (bytes >= 1 << 30) return '${(bytes / (1 << 30)).toStringAsFixed(1)} GB';
  if (bytes >= 1 << 20) return '${(bytes / (1 << 20)).toStringAsFixed(0)} MB';
  if (bytes >= 1 << 10) return '${(bytes / (1 << 10)).toStringAsFixed(0)} KB';
  return '$bytes B';
}
