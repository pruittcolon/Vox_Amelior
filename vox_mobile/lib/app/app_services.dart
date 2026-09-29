import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/app/token_store.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/core/log.dart';
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
  final ServiceController listening;
  final GemmaLlmEngine llm = GemmaLlmEngine();
  late final AssistantService assistant = AssistantService(llm: llm, transcripts: transcripts, speakers: speakers);
  late final ActionExecutor executor = ActionExecutor(
    outbox: outbox,
    dispatcher: WebhookDispatcher(outbox, DioHttpSender()),
    notes: notes,
    notifier: LocalNotifier.instance,
  );
  late final ModelDownloads downloads = ModelDownloads(
    store: models,
    installer: ModelInstaller(store: models),
    tokenProvider: tokens.huggingFace,
    onInstalled: _onModelInstalled,
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
    services.listening.initialize();
    services.writeServiceConfig();
    unawaited(services.activateGemma());
    unawaited(LocalNotifier.instance.initialize());
    return services;
  }

  File get serviceConfigFile => File(p.join(supportDir.path, 'service_config.json'));

  bool get speechReady => SpeechModelPaths.fromStore(models) != null;

  bool get gemmaReady => models.isInstalled(settings.value.gemmaAsset);

  VoiceprintWorker? get voiceprints {
    final paths = SpeechModelPaths.fromStore(models);
    return paths == null ? null : VoiceprintWorker(paths.speaker);
  }

  /// Hands the current settings and model paths to the listening service.
  bool writeServiceConfig() {
    final paths = SpeechModelPaths.fromStore(models);
    if (paths == null) return false;
    ServiceConfig(dbPath: db.path, paths: paths, settings: settings.value).writeTo(serviceConfigFile);
    return true;
  }

  Future<void> updateSettings(AppSettings next) async {
    final modelChanged = next.gemmaAsset.id != settings.value.gemmaAsset.id;
    settings.value = next;
    await settingsRepo.save(next);
    writeServiceConfig();
    listening.reload();
    if (modelChanged) {
      await llm.unload();
      await activateGemma();
    }
  }

  /// Call after people, rules or transcripts change.
  void dataChanged() {
    dataVersion.value++;
    listening.reload();
  }

  /// Registers the downloaded Gemma file with the inference plugin.
  Future<void> activateGemma() async {
    final asset = settings.value.gemmaAsset;
    if (!models.isInstalled(asset)) return;
    try {
      await FlutterGemma.installModel(modelType: ModelType.gemmaIt, fileType: ModelFileType.litertlm)
          .fromFile(models.file(asset, asset.files.first.fileName).path)
          .install();
    } on Object catch (e, st) {
      Log.e('gemma', 'could not register model', e, st);
    }
  }

  Future<void> _onModelInstalled(ModelAsset asset) async {
    if (asset.kind == ModelKind.languageModel) {
      await activateGemma();
    } else {
      writeServiceConfig();
    }
  }
}
