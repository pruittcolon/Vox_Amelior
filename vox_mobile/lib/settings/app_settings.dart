import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

/// User preferences. Immutable; changed via [copyWith] and saved as one JSON blob.
class AppSettings {
  const AppSettings({
    this.wakePhrases = const ['hey vox', 'ok vox', 'vox'],
    this.matchThreshold = 0.55,
    this.matchMargin = 0.04,
    this.guestThreshold = 0.6,
    this.vadThreshold = 0.5,
    this.retentionDays = 90,
    this.speakReplies = true,
    this.gemmaModelId = 'gemma-3n-e4b-it-int4',
    this.gemmaUrlOverride,
  });

  /// Phrases that address the assistant ("Hey Vox, ...").
  final List<String> wakePhrases;

  /// How similar a voice must be to an enrolled person to be named (0–1).
  final double matchThreshold;

  /// How much better the best person must beat the runner-up (0–0.3).
  final double matchMargin;

  /// How similar a stranger's voice must be to a known "Guest" to be grouped.
  final double guestThreshold;

  /// Speech-detector sensitivity (lower hears quieter speech, more noise).
  final double vadThreshold;

  /// Transcripts older than this are deleted automatically. 0 keeps everything.
  final int retentionDays;

  /// Read assistant answers aloud.
  final bool speakReplies;
  final String gemmaModelId;

  /// Use another address for the Gemma download (e.g. a mirror).
  final String? gemmaUrlOverride;

  IdentifierConfig get identifierConfig => IdentifierConfig(
        threshold: matchThreshold,
        margin: matchMargin,
        clusterThreshold: guestThreshold,
      );

  ModelAsset get gemmaAsset {
    final base = ModelCatalog.all.firstWhere(
      (m) => m.id == gemmaModelId && m.kind == ModelKind.languageModel,
      orElse: () => ModelCatalog.gemma3nE4b,
    );
    final override = gemmaUrlOverride?.trim();
    if (override == null || override.isEmpty) return base;
    return base.withFiles([
      RemoteFile(url: override, fileName: base.files.first.fileName),
    ]);
  }

  AppSettings copyWith({
    List<String>? wakePhrases,
    double? matchThreshold,
    double? matchMargin,
    double? guestThreshold,
    double? vadThreshold,
    int? retentionDays,
    bool? speakReplies,
    String? gemmaModelId,
    String? gemmaUrlOverride,
    bool clearGemmaUrl = false,
  }) =>
      AppSettings(
        wakePhrases: wakePhrases ?? this.wakePhrases,
        matchThreshold: matchThreshold ?? this.matchThreshold,
        matchMargin: matchMargin ?? this.matchMargin,
        guestThreshold: guestThreshold ?? this.guestThreshold,
        vadThreshold: vadThreshold ?? this.vadThreshold,
        retentionDays: retentionDays ?? this.retentionDays,
        speakReplies: speakReplies ?? this.speakReplies,
        gemmaModelId: gemmaModelId ?? this.gemmaModelId,
        gemmaUrlOverride: clearGemmaUrl ? null : (gemmaUrlOverride ?? this.gemmaUrlOverride),
      );

  Map<String, Object?> toJson() => {
        'wakePhrases': wakePhrases,
        'matchThreshold': matchThreshold,
        'matchMargin': matchMargin,
        'guestThreshold': guestThreshold,
        'vadThreshold': vadThreshold,
        'retentionDays': retentionDays,
        'speakReplies': speakReplies,
        'gemmaModelId': gemmaModelId,
        'gemmaUrlOverride': gemmaUrlOverride,
      };

  /// Tolerant of missing or invalid fields so an old or damaged settings
  /// blob never prevents the app from starting.
  factory AppSettings.fromJson(Map<String, Object?> j) {
    const d = AppSettings();
    double num01(String k, double fallback, {double min = 0, double max = 1}) {
      final v = j[k];
      return v is num ? v.toDouble().clamp(min, max) : fallback;
    }

    final phrases = (j['wakePhrases'] is List<Object?>)
        ? (j['wakePhrases']! as List<Object?>).whereType<String>().map((s) => s.trim()).where((s) => s.isNotEmpty).toList()
        : null;
    final days = j['retentionDays'];
    return AppSettings(
      wakePhrases: (phrases == null || phrases.isEmpty) ? d.wakePhrases : phrases,
      matchThreshold: num01('matchThreshold', d.matchThreshold, min: 0.2, max: 0.95),
      matchMargin: num01('matchMargin', d.matchMargin, max: 0.3),
      guestThreshold: num01('guestThreshold', d.guestThreshold, min: 0.2, max: 0.95),
      vadThreshold: num01('vadThreshold', d.vadThreshold, min: 0.2, max: 0.9),
      retentionDays: days is int && days >= 0 ? days : d.retentionDays,
      speakReplies: j['speakReplies'] is bool ? j['speakReplies']! as bool : d.speakReplies,
      gemmaModelId: j['gemmaModelId'] is String ? j['gemmaModelId']! as String : d.gemmaModelId,
      gemmaUrlOverride: j['gemmaUrlOverride'] is String ? j['gemmaUrlOverride']! as String : null,
    );
  }
}

class SettingsRepository {
  static const _key = 'app_settings_v1';

  Future<AppSettings> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null) return const AppSettings();
    try {
      return AppSettings.fromJson(jsonDecode(raw) as Map<String, Object?>);
    } on Object {
      return const AppSettings();
    }
  }

  Future<void> save(AppSettings settings) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(settings.toJson()));
  }
}
