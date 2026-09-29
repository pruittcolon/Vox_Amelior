import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
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
    this.llmId = 'gemma-4-e4b-it',
    this.customLlmUrl = '',
    this.customLlmName = '',
    this.customLlmType = 'gemma4',
    this.customLlmTools = true,
    this.customLlmNeedsToken = false,
    this.agentMode = true,
    this.instructions = '',
    this.locationMode = LocationMode.off,
    this.places = const [],
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

  /// Which assistant model to use: a catalog id or [ModelCatalog.customLlmId].
  final String llmId;
  final String customLlmUrl;
  final String customLlmName;

  /// flutter_gemma model family of the custom model ('gemma4', 'gemmaIt', 'qwen3'...).
  final String customLlmType;
  final bool customLlmTools;
  final bool customLlmNeedsToken;

  /// Let the assistant call tools (search, notes, reminders, automations).
  final bool agentMode;

  /// Custom instructions for the assistant; empty uses the default.
  final String instructions;

  final LocationMode locationMode;
  final List<Place> places;

  IdentifierConfig get identifierConfig => IdentifierConfig(
        threshold: matchThreshold,
        margin: matchMargin,
        clusterThreshold: guestThreshold,
      );

  /// The assistant model currently selected.
  ModelAsset get llmAsset {
    if (llmId == ModelCatalog.customLlmId && customLlmUrl.trim().isNotEmpty) {
      return ModelCatalog.custom(
        url: customLlmUrl.trim(),
        name: customLlmName,
        llmType: customLlmType,
        supportsTools: customLlmTools,
        requiresToken: customLlmNeedsToken,
      );
    }
    return ModelCatalog.assistants.firstWhere((m) => m.id == llmId, orElse: () => ModelCatalog.gemma4E4b);
  }

  AppSettings copyWith({
    List<String>? wakePhrases,
    double? matchThreshold,
    double? matchMargin,
    double? guestThreshold,
    double? vadThreshold,
    int? retentionDays,
    bool? speakReplies,
    String? llmId,
    String? customLlmUrl,
    String? customLlmName,
    String? customLlmType,
    bool? customLlmTools,
    bool? customLlmNeedsToken,
    bool? agentMode,
    String? instructions,
    LocationMode? locationMode,
    List<Place>? places,
  }) =>
      AppSettings(
        wakePhrases: wakePhrases ?? this.wakePhrases,
        matchThreshold: matchThreshold ?? this.matchThreshold,
        matchMargin: matchMargin ?? this.matchMargin,
        guestThreshold: guestThreshold ?? this.guestThreshold,
        vadThreshold: vadThreshold ?? this.vadThreshold,
        retentionDays: retentionDays ?? this.retentionDays,
        speakReplies: speakReplies ?? this.speakReplies,
        llmId: llmId ?? this.llmId,
        customLlmUrl: customLlmUrl ?? this.customLlmUrl,
        customLlmName: customLlmName ?? this.customLlmName,
        customLlmType: customLlmType ?? this.customLlmType,
        customLlmTools: customLlmTools ?? this.customLlmTools,
        customLlmNeedsToken: customLlmNeedsToken ?? this.customLlmNeedsToken,
        agentMode: agentMode ?? this.agentMode,
        instructions: instructions ?? this.instructions,
        locationMode: locationMode ?? this.locationMode,
        places: places ?? this.places,
      );

  Map<String, Object?> toJson() => {
        'wakePhrases': wakePhrases,
        'matchThreshold': matchThreshold,
        'matchMargin': matchMargin,
        'guestThreshold': guestThreshold,
        'vadThreshold': vadThreshold,
        'retentionDays': retentionDays,
        'speakReplies': speakReplies,
        'llmId': llmId,
        'customLlmUrl': customLlmUrl,
        'customLlmName': customLlmName,
        'customLlmType': customLlmType,
        'customLlmTools': customLlmTools,
        'customLlmNeedsToken': customLlmNeedsToken,
        'agentMode': agentMode,
        'instructions': instructions,
        'locationMode': locationMode.name,
        'places': [for (final p in places) p.toJson()],
      };

  /// Tolerant of missing or invalid fields so an old or damaged settings
  /// blob never prevents the app from starting.
  factory AppSettings.fromJson(Map<String, Object?> j) {
    const d = AppSettings();
    double num01(String k, double fallback, {double min = 0, double max = 1}) {
      final v = j[k];
      return v is num ? v.toDouble().clamp(min, max) : fallback;
    }

    T typed<T>(String k, T fallback) {
      final v = j[k];
      return v is T ? v : fallback;
    }

    final phrases = (j['wakePhrases'] is List<Object?>)
        ? (j['wakePhrases']! as List<Object?>).whereType<String>().map((s) => s.trim()).where((s) => s.isNotEmpty).toList()
        : null;
    final days = j['retentionDays'];
    final places = (j['places'] is List<Object?>)
        ? (j['places']! as List<Object?>).map(Place.fromJson).whereType<Place>().toList()
        : const <Place>[];
    return AppSettings(
      wakePhrases: (phrases == null || phrases.isEmpty) ? d.wakePhrases : phrases,
      matchThreshold: num01('matchThreshold', d.matchThreshold, min: 0.2, max: 0.95),
      matchMargin: num01('matchMargin', d.matchMargin, max: 0.3),
      guestThreshold: num01('guestThreshold', d.guestThreshold, min: 0.2, max: 0.95),
      vadThreshold: num01('vadThreshold', d.vadThreshold, min: 0.2, max: 0.9),
      retentionDays: days is int && days >= 0 ? days : d.retentionDays,
      speakReplies: typed('speakReplies', d.speakReplies),
      llmId: typed('llmId', d.llmId),
      customLlmUrl: typed('customLlmUrl', d.customLlmUrl),
      customLlmName: typed('customLlmName', d.customLlmName),
      customLlmType: typed('customLlmType', d.customLlmType),
      customLlmTools: typed('customLlmTools', d.customLlmTools),
      customLlmNeedsToken: typed('customLlmNeedsToken', d.customLlmNeedsToken),
      agentMode: typed('agentMode', d.agentMode),
      instructions: typed('instructions', d.instructions),
      locationMode: LocationMode.values.asNameMap()[j['locationMode']] ?? d.locationMode,
      places: places,
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
