import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:vox_amelior_mobile/app/assistant_client.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/app/token_store.dart';
import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_installer.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/native/gemma_llm_engine.dart';
import 'package:vox_amelior_mobile/native/local_notifier.dart';
import 'package:vox_amelior_mobile/native/voiceprint_worker.dart';
import 'package:vox_amelior_mobile/service/service_controller.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';

/// Composition root: creates and owns everything the app's UI uses.
class AppServices {
  AppServices._({
    required this.supportDir,
    required this.db,
    required this.settingsRepo,
    required this.settings,
    required this.tokens,
    required this.models,
  })  : speakers = SpeakerRepository(db),
        transcripts = TranscriptRepository(db),
        rules = RuleRepository(db),
        outbox = OutboxRepository(db),
        notes = NoteRepository(db),
        requests = AssistantRequestRepository(db),
        reminders = ReminderRepository(db),
        listening = ServiceController();

  final Directory supportDir;
  final AppDatabase db;
  final SettingsRepository settingsRepo;
  final ValueNotifier<AppSettings> settings;
  final TokenStore tokens;
  final ModelStore models;
  final SpeakerRepository speakers;
  final TranscriptRepository transcripts;
  final RuleRepository rules;
  final OutboxRepository outbox;
  final NoteRepository notes;
  final AssistantRequestRepository requests;
  final ReminderRepository reminders;
  final ServiceController listening;

  late final ActionExecutor executor = ActionExecutor(
    outbox: outbox,
    dispatcher: WebhookDispatcher(outbox, DioHttpSender()),
    notes: notes,
    notifier: LocalNotifier.instance,
  );

  /// Used only while the always-on service is off.
  late final GemmaLlmEngine localLlm = GemmaLlmEngine(spec: () {
    final c = LlmConfig.fromStore(models, settings.value.llmAsset);
    return c == null ? null : LlmModelSpec(path: c.path, modelType: c.modelType, supportsTools: c.supportsTools);
  });

  late final AssistantClient assistant = AssistantClient(
    service: listening,
    requests: requests,
    transcripts: transcripts,
    local: AssistantService(
      llm: localLlm,
      transcripts: transcripts,
      speakers: speakers,
      instructions: () => settings.value.instructions,
      agentEnabled: () => settings.value.agentMode,
      toolbox: AgentToolbox(transcripts: transcripts, speakers: speakers, notes: notes, reminders: reminders, rules: rules),
    ),
  );

  late final ModelDownloads downloads = ModelDownloads(
    store: models,
    installer: ModelInstaller(store: models),
    tokenProvider: tokens.huggingFace,
    resolve: _assetById,
    onInstalled: (_) async {
      writeServiceConfig();
      listening.reload();
    },
    onBusyChanged: listening.keepAliveForDownloads,
    onProgressText: (t) => unawaited(listening.updateDownloadNotification(t)),
  );

  /// Bumped whenever people, transcripts or rules change, so screens refresh.
  final ValueNotifier<int> dataVersion = ValueNotifier(0);

  static Future<AppServices> create() async {
    final support = await getApplicationSupportDirectory();
    final settingsRepo = SettingsRepository();
    final services = AppServices._(
      supportDir: support,
      db: AppDatabase.open(p.join(support.path, 'vox.db')),
      settingsRepo: settingsRepo,
      settings: ValueNotifier(await settingsRepo.load()),
      tokens: TokenStore(),
      models: ModelStore(Directory(p.join(support.path, 'models'))),
    );
    services._init();
    return services;
  }

  void _init() {
    // Free space used by models from older versions of the app.
    models.removeExcept({for (final m in ModelCatalog.all) m.id, ModelCatalog.customLlmId});
    listening.initialize();
    writeServiceConfig();
    unawaited(LocalNotifier.instance.initialize());
    unawaited(downloads.resumeInterrupted());
    // The service loads its own copy of the model; don't hold two.
    listening.addListener(() {
      if (listening.isListening) unawaited(localLlm.unload());
    });
  }

  File get serviceConfigFile => File(p.join(supportDir.path, 'service_config.json'));

  bool get speechReady => SpeechModelPaths.fromStore(models) != null;

  bool get assistantReady => models.isInstalled(settings.value.llmAsset);

  VoiceprintWorker? get voiceprints {
    final paths = SpeechModelPaths.fromStore(models);
    return paths == null ? null : VoiceprintWorker(paths.speaker);
  }

  ModelAsset? _assetById(String id) {
    if (id == ModelCatalog.customLlmId) {
      final a = settings.value.llmAsset;
      return a.id == id ? a : null;
    }
    return ModelCatalog.all.where((m) => m.id == id).firstOrNull;
  }

  /// Hands the current settings and model paths to the listening service.
  bool writeServiceConfig() {
    final paths = SpeechModelPaths.fromStore(models);
    if (paths == null) return false;
    ServiceConfig(
      dbPath: db.path,
      paths: paths,
      settings: settings.value,
      queueDir: p.join(supportDir.path, 'speech_queue'),
      llm: LlmConfig.fromStore(models, settings.value.llmAsset),
    ).writeTo(serviceConfigFile);
    return true;
  }

  Future<void> updateSettings(AppSettings next) async {
    final modelChanged = next.llmAsset.id != settings.value.llmAsset.id ||
        next.llmAsset.files.first.url != settings.value.llmAsset.files.first.url;
    settings.value = next;
    await settingsRepo.save(next);
    writeServiceConfig();
    listening.reload();
    if (modelChanged) await localLlm.unload();
  }

  /// Call after people, rules or transcripts change.
  void dataChanged() {
    dataVersion.value++;
    listening.reload();
  }
}
