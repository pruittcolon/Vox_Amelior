import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

/// User preferences. Immutable; changed via [copyWith] and saved as one JSON blob.
class AppSettings {
  const AppSettings({
    this.wakePhrases = const [],
    this.matchThreshold = 0.55,
    this.matchMargin = 0.04,
    this.guestThreshold = 0.6,
    this.vadThreshold = 0.5,
    this.micGain = 1.15,
    this.pauseSeconds = 0.6,
    this.minSpeechSeconds = 0.3,
    this.speechModel = 'fp16',
    this.splitMinSeconds = 1.5,
    this.splitMinWords = 2,
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
    this.multiPatterns = true,
    this.splitSpeakers = true,
    this.hearTone = true,
    this.clipMode = ClipMode.off,
    this.clipPeople = const [],
    this.clipLimitMb = 2048,
    this.contextTokens = ContextBudget.defaultContext,
    this.contextTested = 0,
    this.contextTestNote = '',
    this.reviewChunkTokens = 0,
    this.customTemplates = const [],
    this.themeMode = 'system',
    this.accent = 0xFF4F46E5,
    this.textScale = 1.0,
    this.corners = 'rounded',
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

  /// Microphone boost applied before anything else hears the audio
  /// (1.0 = as recorded; the default 1.15 is +15%).
  final double micGain;

  /// How long a pause must last before a sentence counts as finished.
  final double pauseSeconds;

  /// Shorter bursts of sound are ignored (coughs, clicks).
  final double minSpeechSeconds;

  /// Which speech model: 'fp16' (half precision, the default) or 'int8' (smaller, optional download).
  final String speechModel;

  /// A line is only split at a speaker change when every part lasts at least
  /// this long and has at least [splitMinWords] words.
  final double splitMinSeconds;
  final int splitMinWords;

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

  /// Match against several voice patterns per person, not just one average.
  final bool multiPatterns;

  /// Split a line where the speaker changes and mark people talking at once
  /// (needs the speaker-change model).
  final bool splitSpeakers;

  /// Note the tone of voice and sounds of each line (needs the tone model).
  final bool hearTone;

  /// Saving audio clips of what was said (for training later).
  final ClipMode clipMode;
  final List<String> clipPeople;
  final int clipLimitMb;

  /// The assistant's context window in tokens (see [ContextBudget]).
  final int contextTokens;

  /// Largest context the phone test passed (0 = not tested yet).
  final int contextTested;

  /// What the last phone test found, in words.
  final String contextTestNote;

  /// Transcript per review part; 0 uses the recommendation (half the context).
  final int reviewChunkTokens;

  /// Review templates the user saved.
  final List<ReviewTemplate> customTemplates;

  /// Appearance: 'system' | 'light' | 'dark'.
  final String themeMode;

  /// Accent colour (ARGB).
  final int accent;

  /// Text size multiplier (0.85–1.4).
  final double textScale;

  /// Corner style: 'square' | 'soft' | 'rounded'.
  final String corners;

  IdentifierConfig get identifierConfig => IdentifierConfig(
        threshold: matchThreshold,
        margin: matchMargin,
        clusterThreshold: guestThreshold,
        usePatterns: multiPatterns,
      );

  /// The speech-recognition model chosen in settings.
  ModelAsset get asrAsset => speechModel == 'int8' ? ModelCatalog.parakeet : ModelCatalog.parakeetFp16;

  ContextBudget get budget => ContextBudget(contextTokens, chunkTokens: reviewChunkTokens);

  ClipPolicy get clipPolicy => ClipPolicy(mode: clipMode, people: clipPeople, limitBytes: clipLimitMb * 1024 * 1024);

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
    double? micGain,
    double? pauseSeconds,
    double? minSpeechSeconds,
    String? speechModel,
    double? splitMinSeconds,
    int? splitMinWords,
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
    bool? multiPatterns,
    bool? splitSpeakers,
    bool? hearTone,
    ClipMode? clipMode,
    List<String>? clipPeople,
    int? clipLimitMb,
    int? contextTokens,
    int? contextTested,
    String? contextTestNote,
    int? reviewChunkTokens,
    List<ReviewTemplate>? customTemplates,
    String? themeMode,
    int? accent,
    double? textScale,
    String? corners,
  }) =>
      AppSettings(
        wakePhrases: wakePhrases ?? this.wakePhrases,
        matchThreshold: matchThreshold ?? this.matchThreshold,
        matchMargin: matchMargin ?? this.matchMargin,
        guestThreshold: guestThreshold ?? this.guestThreshold,
        vadThreshold: vadThreshold ?? this.vadThreshold,
        micGain: micGain ?? this.micGain,
        pauseSeconds: pauseSeconds ?? this.pauseSeconds,
        minSpeechSeconds: minSpeechSeconds ?? this.minSpeechSeconds,
        speechModel: speechModel ?? this.speechModel,
        splitMinSeconds: splitMinSeconds ?? this.splitMinSeconds,
        splitMinWords: splitMinWords ?? this.splitMinWords,
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
        multiPatterns: multiPatterns ?? this.multiPatterns,
        splitSpeakers: splitSpeakers ?? this.splitSpeakers,
        hearTone: hearTone ?? this.hearTone,
        clipMode: clipMode ?? this.clipMode,
        clipPeople: clipPeople ?? this.clipPeople,
        clipLimitMb: clipLimitMb ?? this.clipLimitMb,
        contextTokens: contextTokens ?? this.contextTokens,
        contextTested: contextTested ?? this.contextTested,
        contextTestNote: contextTestNote ?? this.contextTestNote,
        reviewChunkTokens: reviewChunkTokens ?? this.reviewChunkTokens,
        customTemplates: customTemplates ?? this.customTemplates,
        themeMode: themeMode ?? this.themeMode,
        accent: accent ?? this.accent,
        textScale: textScale ?? this.textScale,
        corners: corners ?? this.corners,
      );

  Map<String, Object?> toJson() => {
        'wakePhrases': wakePhrases,
        'matchThreshold': matchThreshold,
        'matchMargin': matchMargin,
        'guestThreshold': guestThreshold,
        'vadThreshold': vadThreshold,
        'micGain': micGain,
        'pauseSeconds': pauseSeconds,
        'minSpeechSeconds': minSpeechSeconds,
        'asrModel': speechModel,
        'splitMinSeconds': splitMinSeconds,
        'splitMinWords': splitMinWords,
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
        'multiPatterns': multiPatterns,
        'splitSpeakers': splitSpeakers,
        'hearTone': hearTone,
        'clipMode': clipMode.name,
        'clipPeople': clipPeople,
        'clipLimitMb': clipLimitMb,
        'contextTokens': contextTokens,
        'contextTested': contextTested,
        'contextTestNote': contextTestNote,
        'reviewChunkTokens': reviewChunkTokens,
        'customTemplates': [for (final t in customTemplates) t.toJson()],
        'themeMode': themeMode,
        'accent': accent,
        'textScale': textScale,
        'corners': corners,
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
    final clipPeople = (j['clipPeople'] is List<Object?>)
        ? (j['clipPeople']! as List<Object?>).whereType<String>().toList()
        : const <String>[];
    final templates = (j['customTemplates'] is List<Object?>)
        ? (j['customTemplates']! as List<Object?>).map(ReviewTemplate.fromJson).whereType<ReviewTemplate>().toList()
        : const <ReviewTemplate>[];
    int intIn(String k, int fallback, int min, int max) {
      final v = j[k];
      return v is int && v >= min && v <= max ? v : fallback;
    }
    return AppSettings(
      // The old built-in phrases ("hey vox", …) were a default, not a choice: now off unless set.
      wakePhrases: (phrases == null || _oldDefaultWake(phrases)) ? d.wakePhrases : phrases,
      matchThreshold: num01('matchThreshold', d.matchThreshold, min: 0.2, max: 0.95),
      matchMargin: num01('matchMargin', d.matchMargin, max: 0.3),
      guestThreshold: num01('guestThreshold', d.guestThreshold, min: 0.2, max: 0.95),
      vadThreshold: num01('vadThreshold', d.vadThreshold, min: 0.2, max: 0.9),
      micGain: num01('micGain', d.micGain, min: 0.5, max: 4.0),
      pauseSeconds: num01('pauseSeconds', d.pauseSeconds, min: 0.3, max: 1.5),
      minSpeechSeconds: num01('minSpeechSeconds', d.minSpeechSeconds, min: 0.1, max: 1.0),
      // Stored as 'asrModel': the old 'speechModel' key held int8 for everyone
      // (the old default), so phones moving up start on fp16 too.
      speechModel: const ['int8', 'fp16'].contains(j['asrModel']) ? j['asrModel']! as String : d.speechModel,
      splitMinSeconds: num01('splitMinSeconds', d.splitMinSeconds, min: 0.8, max: 3.0),
      splitMinWords: intIn('splitMinWords', d.splitMinWords, 1, 5),
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
      multiPatterns: typed('multiPatterns', d.multiPatterns),
      splitSpeakers: typed('splitSpeakers', d.splitSpeakers),
      hearTone: typed('hearTone', d.hearTone),
      clipMode: ClipMode.values.asNameMap()[j['clipMode']] ?? d.clipMode,
      clipPeople: clipPeople,
      clipLimitMb: intIn('clipLimitMb', d.clipLimitMb, 100, 64 * 1024),
      contextTokens: intIn('contextTokens', d.contextTokens, 1024, 131072),
      contextTested: intIn('contextTested', d.contextTested, 0, 131072),
      contextTestNote: typed('contextTestNote', d.contextTestNote),
      reviewChunkTokens: intIn('reviewChunkTokens', d.reviewChunkTokens, 0, 131072),
      customTemplates: templates,
      themeMode: const ['system', 'light', 'dark'].contains(j['themeMode']) ? j['themeMode']! as String : d.themeMode,
      accent: typed('accent', d.accent),
      textScale: num01('textScale', d.textScale, min: 0.85, max: 1.4),
      corners: const ['square', 'soft', 'rounded'].contains(j['corners']) ? j['corners']! as String : d.corners,
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

bool _oldDefaultWake(List<String> p) => p.length == 3 && p[0] == 'hey vox' && p[1] == 'ok vox' && p[2] == 'vox';
