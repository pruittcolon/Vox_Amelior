/// App-wide preferences and your own voices (clones, mixes, gallery picks,
/// custom characters).
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../tts/voice_mix.dart';
import '../tts/voices.dart';

/// A voice cloned from a clip you recorded or picked.
class ClonedVoice {
  const ClonedVoice({required this.id, required this.name, required this.file, required this.created});
  final String id;
  final String name;

  /// File name inside `<support>/clones/`.
  final String file;
  final DateTime created;

  String get voiceId => 'clone:$id';

  Map<String, Object?> toJson() => {'id': id, 'name': name, 'file': file, 'created': created.toIso8601String()};

  static ClonedVoice fromJson(Map<String, Object?> j) => ClonedVoice(
        id: j['id']! as String,
        name: j['name']! as String,
        file: j['file']! as String,
        created: DateTime.parse(j['created']! as String),
      );
}

/// A Voice Gallery voice you added to your list.
class SavedGalleryVoice {
  const SavedGalleryVoice({
    required this.id,
    required this.name,
    required this.description,
    required this.asset,
    required this.size,
    required this.sha256,
    required this.license,
  });

  final String id;
  final String name;
  final String description;
  final String asset;
  final int size;
  final String sha256;
  final String license;

  String get voiceId => 'gallery:$id';

  Map<String, Object?> toJson() => {
        'id': id,
        'name': name,
        'description': description,
        'asset': asset,
        'size': size,
        'sha256': sha256,
        'license': license,
      };

  static SavedGalleryVoice fromJson(Map<String, Object?> j) => SavedGalleryVoice(
        id: j['id']! as String,
        name: j['name']! as String,
        description: (j['description'] as String?) ?? '',
        asset: j['asset']! as String,
        size: (j['size']! as num).toInt(),
        sha256: j['sha256']! as String,
        license: (j['license'] as String?) ?? '',
      );
}

class Settings extends ChangeNotifier {
  Settings(this.file);

  final File file;

  /// Your overall speed multiplier.
  double speed = 1.0;

  /// Threads for speech generation.
  int threads = 4;

  /// Music beds and sound cues.
  bool music = true;

  /// Clarity: rebuild the top octave of 24 kHz voices (when downloaded).
  bool clarity = true;

  /// Best quality: more refinement steps for Supertonic and Pocket TTS (slower).
  bool bestQuality = true;

  /// Which voices the characters use: 'natural' (Kokoro, most human) or
  /// 'studio' (Supertonic HD). See voiceSets.
  String voiceSet = 'natural';
  bool cloneConsent = false;
  String exportFormat = 'm4a';
  List<VoiceMix> mixes = [];
  List<ClonedVoice> clones = [];
  List<SavedGalleryVoice> gallery = [];
  List<Character> characters = [];

  Future<void> load() async {
    if (!file.existsSync()) return;
    try {
      final j = jsonDecode(await file.readAsString()) as Map<String, Object?>;
      speed = ((j['speed'] as num?) ?? 1.0).toDouble();
      threads = ((j['threads'] as num?) ?? 4).toInt();
      music = (j['music'] as bool?) ?? true;
      clarity = (j['clarity'] as bool?) ?? true;
      bestQuality = (j['bestQuality'] as bool?) ?? true;
      voiceSet = (j['voiceSet'] as String?) ?? 'natural';
      cloneConsent = (j['cloneConsent'] as bool?) ?? false;
      exportFormat = (j['exportFormat'] as String?) ?? 'm4a';
      List<Map<String, Object?>> list(String k) => ((j[k] as List?) ?? const []).cast<Map<String, Object?>>();
      mixes = list('mixes').map(VoiceMix.fromJson).toList();
      clones = list('clones').map(ClonedVoice.fromJson).toList();
      gallery = list('gallery').map(SavedGalleryVoice.fromJson).toList();
      characters = list('characters').map(Character.fromJson).toList();
    } catch (_) {
      await file.copy('${file.path}.broken');
    }
    notifyListeners();
  }

  Future<void> _saving = Future.value();

  /// Saves are queued: two overlapping writes would race on the temp file.
  Future<void> save() => _saving = _saving.catchError((Object _) {}).then((_) => _write());

  Future<void> _write() async {
    await file.parent.create(recursive: true);
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(
      jsonEncode({
        'speed': speed,
        'threads': threads,
        'music': music,
        'clarity': clarity,
        'bestQuality': bestQuality,
        'voiceSet': voiceSet,
        'cloneConsent': cloneConsent,
        'exportFormat': exportFormat,
        'mixes': mixes.map((m) => m.toJson()).toList(),
        'clones': clones.map((c) => c.toJson()).toList(),
        'gallery': gallery.map((g) => g.toJson()).toList(),
        'characters': characters.map((c) => c.toJson()).toList(),
      }),
      flush: true,
    );
    await tmp.rename(file.path);
  }

  Future<void> update(void Function(Settings s) change) async {
    change(this);
    notifyListeners();
    await save();
  }
}
