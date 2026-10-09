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
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/context_probe.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_templates.dart';
import 'package:vox_amelior_mobile/assistant/review_worker.dart';
import 'package:vox_amelior_mobile/assistant/time_window.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/data/insights_repository.dart';
import 'package:vox_amelior_mobile/search/embedder_worker.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';
import 'package:vox_amelior_mobile/search/hybrid_search.dart';
import 'package:vox_amelior_mobile/search/vector_store.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_installer.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/native/gemma_llm_engine.dart';
import 'package:vox_amelior_mobile/native/local_notifier.dart';
import 'package:vox_amelior_mobile/native/voiceprint_worker.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';
import 'package:vox_amelior_mobile/service/service_controller.dart';
import 'package:vox_amelior_mobile/service/shared_files.dart';
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
        reviews = ReviewRepository(db),
        clips = ClipStore(db, Directory(voiceClipsDir(supportDir.path))),
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
  final ReviewRepository reviews;
  final ClipStore clips;
  final ServiceController listening;

  /// Statistics over the transcripts (Insights).
  late final InsightsRepository insights = InsightsRepository(db, transcripts);

  // ---- search by meaning (EmbeddingGemma) ---------------------------------------

  /// One compact vector per line.
  late final VectorStore vectors = VectorStore(db);

  /// Words + meaning search, and the lines Gemma reads for a question.
  late final HybridSearch search = HybridSearch(
    db: db,
    transcripts: transcripts,
    vectors: vectors,
    embedder: () => embedder,
    modelId: ModelCatalog.textEmbedder.id,
  );

  /// Embeds new lines in the background.
  late final SearchIndexer indexer = SearchIndexer(store: vectors, embedder: () => embedder, modelId: ModelCatalog.textEmbedder.id);

  AsyncEmbedder? _embedder;
  AsyncEmbedder Function()? _embedderForTests;

  /// The embedder while meaning search is on and its model installed, else null.
  /// The worker isolate only starts on the first embedding.
  AsyncEmbedder? get embedder {
    if (!settings.value.meaningSearch) return null;
    final test = _embedderForTests;
    if (test != null) return _embedder ??= test();
    final m = ModelCatalog.textEmbedder;
    if (!models.isInstalled(m)) return null;
    return _embedder ??= IsolateEmbedder(
      modelPath: models.file(m, 'model_quantized.onnx').path,
      tokenizerPath: models.file(m, 'tokenizer.model').path,
    );
  }

  /// Catches meaning search up with new lines (no-op when it is off).
  void indexForSearch() {
    if (embedder != null) unawaited(indexer.run());
  }

  void _closeEmbedder() {
    indexer.stop();
    final e = _embedder;
    _embedder = null;
    if (e != null) unawaited(e.close());
  }

  /// Bumped whenever a review makes progress, so review screens refresh.
  final ValueNotifier<int> reviewVersion = ValueNotifier(0);

  late final ActionExecutor executor = ActionExecutor(
    outbox: outbox,
    dispatcher: WebhookDispatcher(outbox, DioHttpSender()),
    notes: notes,
    notifier: LocalNotifier.instance,
  );

  /// Used only while the always-on service is off.
  late final GemmaLlmEngine localLlm = GemmaLlmEngine(spec: () {
    final c = LlmConfig.fromStore(models, settings.value.llmAsset);
    return c == null
        ? null
        : LlmModelSpec(
            path: c.path,
            modelType: c.modelType,
            supportsTools: c.supportsTools,
            contextTokens: settings.value.contextTokens,
          );
  });

  ContextBudget get budget => settings.value.budget;

  late final ReviewEngine reviewEngine = ReviewEngine(reviews: reviews, transcripts: transcripts, llm: localLlm);

  /// Works on reviews in the app while Vox is not listening (the listening
  /// service does it otherwise).
  late final ReviewWorker _localReviews = ReviewWorker(
    engine: reviewEngine,
    reviews: reviews,
    owner: 'app',
    schedule: (step) async {
      final more = await step();
      await Future<void>.delayed(const Duration(milliseconds: 300)); // let the UI breathe
      return more;
    },
    canWork: () => !listening.isListening,
    onProgress: (id) {
      reviewVersion.value++;
      final run = reviews.run(id);
      if (run != null) {
        unawaited(listening.updateWorkNotification('Vox is reviewing', '${run.title}: part ${run.doneChunks} of ${run.totalChunks}'));
      }
      if (!reviews.hasActive) unawaited(listening.keepAliveFor('review', false));
    },
  );

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
      budget: () => budget,
      toolbox: AgentToolbox(
        transcripts: transcripts,
        speakers: speakers,
        notes: notes,
        reminders: reminders,
        rules: rules,
        budget: () => budget,
      ),
    ),
    // RAG: lines found by meaning are read alongside the keyword matches.
    hints: (question) => search.hintsFor(question),
  );

  late final ModelDownloads downloads = ModelDownloads(
    store: models,
    installer: ModelInstaller(store: models),
    tokenProvider: tokens.huggingFace,
    resolve: _assetById,
    onInstalled: (asset) async {
      writeServiceConfig();
      listening.reload();
      _fetchChosenSpeechModel();
      if (asset.id == ModelCatalog.textEmbedder.id) indexForSearch();
    },
    onRemoved: _onModelRemoved,
    onBusyChanged: listening.keepAliveForDownloads,
    onProgressText: (t) => unawaited(listening.updateDownloadNotification(t)),
  );

  /// Bumped whenever people, transcripts or rules change, so screens refresh.
  final ValueNotifier<int> dataVersion = ValueNotifier(0);

  /// Services over [db] and [dir] without starting anything (for tests).
  @visibleForTesting
  /// [embedder] stands in for EmbeddingGemma (meaning search on).
  factory AppServices.forTesting({
    required Directory dir,
    required AppDatabase db,
    AppSettings settings = const AppSettings(),
    AsyncEmbedder Function()? embedder,
  }) =>
      AppServices._(
        supportDir: dir,
        db: db,
        settingsRepo: SettingsRepository(),
        settings: ValueNotifier(settings),
        tokens: TokenStore(),
        models: ModelStore(Directory(p.join(dir.path, 'models'))),
      ).._embedderForTests = embedder;

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
    // Lines heard while the app was closed (or before the model arrived).
    indexForSearch();
    dataVersion.addListener(indexForSearch);
    // Phones set up before speaker changes existed get that model too.
    if (speechReady && !models.isInstalled(ModelCatalog.diarizer)) unawaited(downloads.download(ModelCatalog.diarizer));
    _fetchChosenSpeechModel();
    // The service loads its own copy of the model; don't hold two. Reviews
    // move to whichever side runs the model.
    var wasListening = listening.isListening;
    listening.addListener(() {
      final now = listening.isListening;
      if (now == wasListening) return;
      wasListening = now;
      if (now) {
        _localReviews.stop();
        unawaited(localLlm.unload());
        unawaited(listening.keepAliveFor('review', false));
        listening.reviewKick();
      } else {
        _localReviews.restart();
        kickReviews();
      }
    });
    listening.events.listen(_onServiceEvent);
    _collectProbeResults();
    // Only once we know whether the service is running (and holds the model).
    unawaited(listening.refresh().then((_) => kickReviews()));
    if (speakers.ensurePatterns() > 0) dataChanged();
  }

  void _onServiceEvent(Map<Object?, Object?> e) {
    switch (e['type']) {
      case ServiceEvents.review:
        reviewVersion.value++;
      case ServiceEvents.probe:
        if (e['kind'] == 'done') _collectProbeResults();
    }
  }

  /// Applies a phone test that finished (or crashed the app) while no
  /// screen was waiting for it.
  void _collectProbeResults() {
    final crashed = ContextProbe.resultAfterCrash(File(probeMarkerPath(supportDir.path)));
    final finished = takeProbeResult(supportDir.path);
    final result = finished ?? crashed;
    if (result != null) unawaited(applyProbeResult(result));
  }

  // ---- assistant: phone test ------------------------------------------------

  /// Finds the largest context Gemma handles on this phone, using the same
  /// side (service or app) that answers questions. [onStep] reports progress.
  Future<ProbeResult?> testContext(void Function(ProbeStep step) onStep) async {
    if (listening.isListening) {
      final done = Completer<ProbeResult?>();
      Timer? quiet;
      void waitMore() {
        quiet?.cancel();
        // Each size can take minutes on a phone; silence this long means the
        // service is not running the test at all.
        quiet = Timer(const Duration(minutes: 12), () {
          if (!done.isCompleted) done.completeError(const LlmUnavailable('Vox did not respond. Try again.'));
        });
      }

      late StreamSubscription<Map<Object?, Object?>> sub;
      sub = listening.events.listen((e) {
        if (e['type'] != ServiceEvents.probe) return;
        waitMore();
        switch (e['kind']) {
          case 'step':
            final step = ProbeStep.fromJson(e);
            if (step != null) onStep(step);
          case 'done':
            if (!done.isCompleted) done.complete(ProbeResult.fromJson(e));
          case 'error':
            if (!done.isCompleted) done.completeError(LlmUnavailable('${e['message']}'));
        }
      });
      waitMore();
      listening.probe();
      try {
        final result = await done.future;
        takeProbeResult(supportDir.path); // applied just below
        if (result != null) await applyProbeResult(result);
        return result;
      } finally {
        quiet?.cancel();
        await sub.cancel();
      }
    }
    // Reviews wait while the phone is tested (they share the model).
    _localReviews.stop();
    try {
      final result = await ContextProbe(llm: localLlm, markerFile: File(probeMarkerPath(supportDir.path))).run(onStep: onStep);
      await applyProbeResult(result);
      return result;
    } finally {
      _localReviews.restart();
    }
  }

  Future<void> applyProbeResult(ProbeResult r) async {
    final best = r.best > 0 ? r.best : ContextBudget.testSizes.first;
    await updateSettings(settings.value.copyWith(
      contextTokens: best,
      contextTested: r.best,
      contextTestNote: r.note,
      reviewChunkTokens: 0, // back to the recommendation: half of it
    ));
  }

  // ---- reviews ---------------------------------------------------------------

  /// Starts a "go through everything" review. Returns its id, or null when
  /// nothing was said in that period.
  int? startReview({
    required ReviewTemplate template,
    required TimeWindow window,
    List<String> focus = const [],
  }) {
    final id = reviewEngine.start(
      title: template.name,
      prompt: template.prompt,
      format: template.format,
      kind: template.kind,
      periodLabel: window.describe(),
      from: window.from,
      to: window.to,
      focus: focus,
      budget: budget,
    );
    if (id != null) kickReviews();
    reviewVersion.value++;
    return id;
  }

  /// Starts a review of the newest [count] lines said by [speakerIds]
  /// (everyone when empty), optionally only in [emotions] and within [window].
  int? startReviewOfLines({
    required ReviewTemplate template,
    required int count,
    Set<String> speakerIds = const {},
    Set<String> emotions = const {},
    TimeWindow? window,
    String? label,
  }) {
    final id = reviewEngine.startLast(
      title: template.name,
      prompt: template.prompt,
      format: template.format,
      kind: template.kind,
      count: count,
      speakerIds: speakerIds,
      emotions: emotions,
      from: window?.from,
      to: window?.to,
      label: label,
      budget: budget,
    );
    if (id != null) kickReviews();
    reviewVersion.value++;
    return id;
  }

  void pauseReview(int id) {
    reviews.setStatus(id, ReviewStatus.paused);
    reviewVersion.value++;
  }

  void resumeReview(int id) {
    reviews.setStatus(id, ReviewStatus.queued);
    kickReviews();
    reviewVersion.value++;
  }

  void cancelReview(int id) {
    reviews.setStatus(id, ReviewStatus.cancelled);
    reviewVersion.value++;
  }

  void retryReview(int id) {
    reviews.retryFailed(id);
    kickReviews();
    reviewVersion.value++;
  }

  void deleteReview(int id) {
    reviews.delete(id);
    reviewVersion.value++;
  }

  /// Gets queued reviews moving on whichever side runs the model.
  void kickReviews() {
    if (!reviews.hasActive) return;
    if (listening.isListening) {
      listening.reviewKick();
    } else {
      unawaited(listening.keepAliveFor('review', true));
      _localReviews.restart();
    }
  }

  // ---- data --------------------------------------------------------------------

  /// Deletes a conversation and any voice clips saved from it.
  void deleteConversation(int conversationId) {
    try {
      clips.deleteForConversation(conversationId);
    } on Object catch (e) {
      Log.w('clips', 'could not delete clips', e);
    }
    transcripts.deleteConversation(conversationId);
    dataChanged();
  }

  File get serviceConfigFile => File(p.join(supportDir.path, 'service_config.json'));

  bool get speechReady => SpeechModelPaths.fromStore(models, asr: settings.value.asrAsset) != null;

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
    final paths = SpeechModelPaths.fromStore(models, asr: settings.value.asrAsset);
    if (paths == null) {
      // Never leave the service a config that points at deleted models (e.g. after an upgrade).
      try {
        if (serviceConfigFile.existsSync()) serviceConfigFile.deleteSync();
      } on FileSystemException catch (e) {
        Log.w('app', 'could not remove old service config', e);
      }
      return false;
    }
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
    final searchChanged = next.meaningSearch != settings.value.meaningSearch;
    settings.value = next;
    if (searchChanged) next.meaningSearch ? indexForSearch() : _closeEmbedder();
    await settingsRepo.save(next);
    writeServiceConfig();
    listening.reload();
    if (modelChanged) await localLlm.unload();
  }

  /// Chooses the speech model ('fp16', 'int8' or 'fp32'). The chosen model is fetched
  /// if it is missing; whichever one is installed keeps working until it is
  /// ready. Switching away from a model that never finished downloading
  /// stops (and discards) that download.
  Future<void> selectSpeechModel(String model) async {
    await updateSettings(settings.value.copyWith(speechModel: model));
    final chosen = settings.value.asrAsset;
    if (!models.isInstalled(chosen)) await downloads.download(chosen);
    for (final other in ModelCatalog.recognizers) {
      if (other.id == chosen.id || models.isInstalled(other)) continue;
      if (downloads.stateOf(other).isBusy || models.partialBytes(other) > 0) downloads.remove(other);
    }
  }

  /// fp16 is the default speech model and is downloaded first at setup. A
  /// phone that only has the int8 model (set up by an older version) gets
  /// fp16 too, unless int8 was chosen.
  void _fetchChosenSpeechModel() {
    final fp16 = ModelCatalog.parakeetFp16;
    if (settings.value.speechModel != 'fp16' || !speechReady || models.isInstalled(fp16)) return;
    unawaited(downloads.download(fp16));
  }

  /// Deletes a speech model. If it is in use and the other one is
  /// installed, the other one takes over first.
  Future<void> removeSpeechModel(ModelAsset asset) async {
    final other = _otherInstalledRecognizer(asset);
    if (settings.value.asrAsset.id == asset.id && other != null) {
      await updateSettings(settings.value.copyWith(speechModel: _speechModelName(other)));
    }
    downloads.remove(asset);
  }

  ModelAsset? _otherInstalledRecognizer(ModelAsset asset) =>
      ModelCatalog.recognizers.where((m) => m.id != asset.id && models.isInstalled(m)).firstOrNull;

  static String _speechModelName(ModelAsset asset) => ModelCatalog.recognizerName(asset);

  /// A model was deleted: stop pointing the listening service at it.
  void _onModelRemoved(ModelAsset asset) {
    if (asset.id == ModelCatalog.textEmbedder.id) _closeEmbedder();
    final other = _otherInstalledRecognizer(asset);
    if (asset.id == settings.value.asrAsset.id && other != null) {
      unawaited(updateSettings(settings.value.copyWith(speechModel: _speechModelName(other))));
      return;
    }
    writeServiceConfig();
    listening.reload();
  }

  /// Call after people, rules or transcripts change.
  void dataChanged() {
    dataVersion.value++;
    listening.reload();
  }
}
