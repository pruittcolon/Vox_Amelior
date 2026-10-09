import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_tts/flutter_tts.dart';
import 'package:vox_amelior_mobile/assistant/agent_tools.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/assistant_service.dart';
import 'package:vox_amelior_mobile/assistant/context_probe.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/assistant/review_worker.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/automation/segment_handler.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/clip_store.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/location/location_monitor.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/native/gemma_llm_engine.dart';
import 'package:vox_amelior_mobile/native/pcm_source.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/native/sortformer_diarizer.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';
import 'package:vox_amelior_mobile/pipeline/compute_scheduler.dart';
import 'package:vox_amelior_mobile/pipeline/engines.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';
import 'package:vox_amelior_mobile/pipeline/speaker_turns.dart';
import 'package:vox_amelior_mobile/pipeline/speech_capture.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';
import 'package:vox_amelior_mobile/service/shared_files.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

/// Everything that runs while Vox is on: microphone → speech queue →
/// transcription, the assistant (Gemma), automations, reminders and
/// place-based on/off. Lives in the foreground-service isolate.
class ListeningRuntime {
  ListeningRuntime._(this.configFile, this.emit, this.notifier, this._config);

  final File configFile;

  /// Sends an event to the app (ignored if the app is closed).
  final void Function(Map<String, Object?> event) emit;
  final Notifier notifier;
  ServiceConfig _config;

  late final AppDatabase db;
  late final SpeakerRepository speakers;
  late final TranscriptRepository transcripts;
  late final RuleRepository rules;
  late final OutboxRepository outbox;
  late final NoteRepository notes;
  late final AssistantRequestRepository requests;
  late final ReminderRepository reminders;
  late final WebhookDispatcher dispatcher;
  late final ActionExecutor executor;
  late final SegmentHandler handler;
  late final SpeechCapture capture;
  late final SegmentProcessor processor;
  late final ChunkQueue queue;
  late final ComputeScheduler scheduler;
  late final GemmaLlmEngine llm;
  late final AssistantService assistant;
  late final ClipStore clips;
  late final ReviewRepository reviews;
  late final ReviewWorker reviewWorker;
  final Set<int> _cancelled = {};
  final Map<int, void Function()> _stoppers = {};
  bool _probing = false;
  final PcmSource _source = MicrophonePcmSource();
  final FlutterTts _tts = FlutterTts();
  final LocationMonitor _location = const LocationMonitor();
  final LocationPolicy _policy = const LocationPolicy();

  StreamSubscription<Uint8List>? _mic;
  bool _manualPause = false;
  bool _autoPause = false;
  DateTime? _pausedUntil;
  String _reason = 'Starting…';
  String? _placeId;
  int _micFailures = 0;
  Timer? _micRetry;
  int _today = 0;
  DateTime _day = DateTime.now();
  DateTime _lastHousekeeping = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastLocationCheck = DateTime.fromMillisecondsSinceEpoch(0);
  bool _stopped = false;

  bool get isManuallyPaused => _manualPause;

  ListenState get state {
    if (_manualPause || _pausedUntil != null) return ListenState.paused;
    if (_autoPause) return ListenState.autoPaused;
    return _mic == null ? ListenState.starting : ListenState.listening;
  }

  static Future<ListeningRuntime> start({
    required File configFile,
    required void Function(Map<String, Object?> event) emit,
    required Notifier notifier,
  }) async {
    final config = ServiceConfig.readFrom(configFile);
    if (config == null || config.paths.missingFiles().isNotEmpty) {
      throw StateError('Speech models are not set up yet. Open Vox and finish setup.');
    }
    final rt = ListeningRuntime._(configFile, emit, notifier, config);
    await rt._build();
    return rt;
  }

  Future<void> _build() async {
    final c = _config;
    db = AppDatabase.open(c.dbPath);
    speakers = SpeakerRepository(db);
    transcripts = TranscriptRepository(db);
    rules = RuleRepository(db);
    outbox = OutboxRepository(db);
    notes = NoteRepository(db);
    requests = AssistantRequestRepository(db);
    reminders = ReminderRepository(db);
    dispatcher = WebhookDispatcher(outbox, DioHttpSender());
    executor = ActionExecutor(outbox: outbox, dispatcher: dispatcher, notes: notes, notifier: notifier);
    handler = SegmentHandler(
      rules: rules,
      engine: RuleEngine(),
      executor: executor,
      requests: requests,
      wakeParser: WakeCommandParser(c.settings.wakePhrases),
    );

    clips = ClipStore(db, Directory(voiceClipsDir(configFile.parent.path)));
    capture = SpeechCapture(_makeVad(c), gain: c.settings.micGain);
    _asrEncoder = c.paths.encoder;
    processor = SegmentProcessor(
      asr: SherpaParakeetAsr(c.paths),
      embedder: SherpaSpeakerEmbedder(c.paths.speaker),
      identifier: SpeakerIdentifier(
        profiles: speakers.profiles(),
        clusters: speakers.clusters(),
        config: c.settings.identifierConfig,
        newClusterId: SpeakerRepository.newId,
        nextGuestLabel: speakers.nextGuestLabel,
      ),
      transcripts: transcripts,
      speakers: speakers,
      onSaved: (segment, samples) => clips.maybeSave(_config.settings.clipPolicy, segment, samples),
      onReplaced: clips.deleteForSegment,
      diarizer: _loadDiarizer(c.paths.diarizer),
      tone: c.settings.hearTone ? _loadTone(c.paths) : null,
      config: ProcessorConfig(minPartWords: c.settings.splitMinWords),
      turns: SpeakerTurns(minTurnSeconds: c.settings.splitMinSeconds),
    )
      ..splitSpeakers = c.settings.splitSpeakers
      ..hearTone = c.settings.hearTone;
    queue = ChunkQueue(Directory(c.queueDir));
    scheduler = ComputeScheduler(
      queue: queue,
      processor: processor,
      onSegment: _onSegment,
      onActivity: (_, _) => emitStatus(),
    );

    llm = GemmaLlmEngine(spec: () {
      final l = _config.llm;
      return l == null
          ? null
          : LlmModelSpec(
              path: l.path,
              modelType: l.modelType,
              supportsTools: l.supportsTools,
              contextTokens: _config.settings.contextTokens,
            );
    });
    assistant = AssistantService(
      llm: llm,
      transcripts: transcripts,
      speakers: speakers,
      instructions: () => _config.settings.instructions,
      agentEnabled: () => _config.settings.agentMode,
      budget: () => _config.settings.budget,
      toolbox: AgentToolbox(
        transcripts: transcripts,
        speakers: speakers,
        notes: notes,
        reminders: reminders,
        rules: rules,
        hooks: AgentHooks(runAutomation: _runAutomation, pauseListening: pauseFor),
        budget: () => _config.settings.budget,
      ),
    );
    reviews = ReviewRepository(db);
    reviewWorker = ReviewWorker(
      engine: ReviewEngine(reviews: reviews, transcripts: transcripts, llm: llm),
      reviews: reviews,
      owner: 'service',
      schedule: scheduler.runBackground,
      canWork: () => !_stopped,
      onProgress: (id) {
        emit({'type': ServiceEvents.review, 'id': id});
        emitStatus();
      },
    );

    unawaited(_tts.awaitSpeakCompletion(true));
    _housekeeping(force: true);
    await _checkLocation(force: true);
    await _applyListening();
    scheduler.kick(); // anything left over from last time
    // Questions asked while the service was down.
    for (final r in requests.takePending()) {
      unawaited(answer(r.id));
    }
    reviewWorker.kick();
  }

  /// Encoder of the speech model loaded now (to notice a switch of model).
  late String _asrEncoder;

  static SherpaVad _makeVad(ServiceConfig c) => SherpaVad(
        modelPath: c.paths.vad,
        threshold: c.settings.vadThreshold,
        minSilenceSeconds: c.settings.pauseSeconds,
        minSpeechSeconds: c.settings.minSpeechSeconds,
      );

  /// The speaker-change model, or null (then lines are simply not split).
  static DiarizationEngine? _loadDiarizer(String? path) {
    if (path == null || !File(path).existsSync()) return null;
    try {
      return SortformerDiarizer(path);
    } on Object catch (e, st) {
      Log.e('listen', 'speaker-change model could not start', e, st);
      return null;
    }
  }

  /// The tone-of-voice model, or null (then lines simply get no tone).
  static ToneEngine? _loadTone(SpeechModelPaths paths) {
    final model = paths.toneModel;
    final tokens = paths.toneTokens;
    if (model == null || tokens == null || !File(model).existsSync() || !File(tokens).existsSync()) return null;
    try {
      return SherpaSenseVoiceTone(model: model, tokens: tokens);
    } on Object catch (e, st) {
      Log.e('listen', 'tone-of-voice model could not start', e, st);
      return null;
    }
  }

  // ---- microphone ------------------------------------------------------

  bool get _shouldListen => !_manualPause && !_autoPause && _pausedUntil == null && !_stopped;

  Future<void> _applyListening() async {
    if (_shouldListen && _mic == null) {
      await _startMic();
    } else if (!_shouldListen && _mic != null) {
      await _stopMic();
    }
    emitStatus();
  }

  Future<void> _startMic() async {
    _micRetry?.cancel();
    try {
      final stream = await _source.start();
      _mic = stream.listen(
        _onPcm,
        onError: (Object e) => _onMicFailure('Microphone error: $e'),
        onDone: () {
          if (_shouldListen) _onMicFailure('Microphone stopped unexpectedly');
        },
        cancelOnError: true,
      );
      _micFailures = 0;
      _reason = 'Listening';
    } on Object catch (e) {
      _onMicFailure('Could not open the microphone: $e');
    }
  }

  Future<void> _stopMic() async {
    _micRetry?.cancel();
    await _mic?.cancel();
    _mic = null;
    await _source.stop();
    _enqueue(capture.flush());
  }

  void _onMicFailure(String message) {
    Log.w('listen', message);
    _mic = null;
    _micFailures++;
    emit({'type': ServiceEvents.error, 'message': message, 'fatal': false});
    if (!_shouldListen) return;
    // Back off: 2s, 4s, 8s ... up to a minute, and keep trying.
    final delay = Duration(seconds: (1 << _micFailures.clamp(1, 6)).clamp(2, 60));
    _reason = 'Microphone busy — retrying';
    emitStatus();
    _micRetry?.cancel();
    _micRetry = Timer(delay, () => unawaited(_applyListening()));
  }

  /// The app is showing a level meter until then. Screens renew this every
  /// few seconds, so a meter left on by an app that closed switches itself off.
  DateTime? _meterUntil;
  DateTime _lastLevel = DateTime.fromMillisecondsSinceEpoch(0);

  static const Duration meterLease = Duration(seconds: 6);

  /// Starts (or renews) or stops sending [ServiceEvents.level] about 5 times a second.
  void requestLevel(bool on) {
    _meterUntil = on ? DateTime.now().add(meterLease) : null;
    capture.measureLevel = on;
    if (!on) capture.takeLevel();
  }

  void _onPcm(Uint8List bytes) {
    if (!_shouldListen) return;
    _enqueue(capture.addPcm16(bytes));
    if (_meterUntil != null) _emitLevel();
  }

  void _emitLevel() {
    final now = DateTime.now();
    if (now.isAfter(_meterUntil!)) {
      requestLevel(false);
      return;
    }
    if (now.difference(_lastLevel) < const Duration(milliseconds: 200)) return;
    _lastLevel = now;
    final l = capture.takeLevel();
    emit({
      'type': ServiceEvents.level,
      'level': AudioLevel.meter(l.rms),
      'peak': AudioLevel.meter(l.peak),
      'clipping': l.clipping,
    });
  }

  void _enqueue(List<CapturedSpeech> speech) {
    if (speech.isEmpty) return;
    for (final s in speech) {
      queue.push(s.samples, s.startedAt);
    }
    scheduler.kick();
  }

  // ---- utterances and the assistant ------------------------------------

  void _onSegment(SegmentView segment, LineStage stage) {
    final now = DateTime.now();
    if (now.day != _day.day) {
      _day = now;
      _today = 0;
    }
    if (stage != LineStage.finished) _today++;
    emit({'type': ServiceEvents.segment, 'id': segment.id, 'stage': stage.name});
    unawaited(_afterSegment(segment, stage));
  }

  Future<void> _afterSegment(SegmentView segment, LineStage stage) async {
    try {
      final outcome = await handler.handle(segment, stage: stage);
      final id = outcome.assistantRequestId;
      if (id != null) unawaited(answer(id));
    } on Object catch (e, st) {
      Log.e('listen', 'segment follow-up failed', e, st);
    }
  }

  /// Answers request [id] through the scheduler (one heavy job at a time),
  /// streaming progress to the app and storing the result.
  Future<void> answer(int id) => scheduler.runExclusive(() async {
        final request = requests.get(id);
        if (request == null || request.status != RequestStatus.pending) return;
        if (_cancelled.remove(id)) {
          requests.fail(id, 'Stopped.');
          return;
        }
        final buffer = StringBuffer();
        var sources = const <SegmentView>[];
        final finished = Completer<void>();
        final sub = assistant.ask(request.text).listen(
          (e) {
            if (e.sources != null) {
              sources = e.sources!;
              emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'sources', 'ids': [for (final s in sources) s.id]});
            } else if (e.token != null) {
              buffer.write(e.token);
              emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'token', 't': e.token});
            } else if (e.toolName != null) {
              emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'tool', 'name': e.toolName});
            }
          },
          onError: (Object e, StackTrace st) {
            if (!finished.isCompleted) finished.completeError(e, st);
          },
          onDone: () {
            if (!finished.isCompleted) finished.complete();
          },
          cancelOnError: true,
        );
        // The Stop button ends the answer at once (Gemma is told to stop).
        _stoppers[id] = () {
          unawaited(sub.cancel());
          if (!finished.isCompleted) finished.completeError(const _Stopped());
        };
        try {
          await finished.future;
          final text = buffer.toString().trim().isEmpty ? 'Sorry, I have no answer for that.' : buffer.toString().trim();
          requests.answer(id, text, sources: [for (final s in sources) s.id]);
          emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'done'});
          if (request.source == RequestSource.voice) {
            await notifier.show(request.text, text);
            if (_config.settings.speakReplies) await _speak(text);
          }
        } on _Stopped {
          requests.fail(id, 'Stopped.');
          emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'error', 'message': 'Stopped.'});
        } on LlmUnavailable catch (e) {
          requests.fail(id, e.message);
          emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'error', 'message': e.message});
          if (request.source == RequestSource.voice) await notifier.show('Vox cannot answer yet', e.message);
        } on Object catch (e, st) {
          Log.e('assistant', 'answer failed', e, st);
          final message = friendlyLlmError(e);
          requests.fail(id, message);
          emit({'type': ServiceEvents.answer, 'id': id, 'kind': 'error', 'message': message});
          if (request.source == RequestSource.voice) await notifier.show('Vox could not answer', message);
        } finally {
          _stoppers.remove(id);
        }
      });

  /// Stops answering [id] (the app's Stop button), or skips it if it has
  /// not started yet.
  void cancelAnswer(int id) {
    final stop = _stoppers[id];
    if (stop != null) {
      stop();
    } else {
      _cancelled.add(id);
    }
  }

  /// Runs the phone context test with this service's model, streaming
  /// progress to the app. The result is also written to a file so it is not
  /// lost if the app is closed meanwhile.
  Future<void> runProbe() async {
    if (_probing) return;
    _probing = true;
    try {
      await scheduler.runExclusive(() async {
        final probe = ContextProbe(llm: llm, markerFile: File(probeMarkerPath(configFile.parent.path)));
        final result = await probe.run(
          onStep: (s) => emit({'type': ServiceEvents.probe, 'kind': 'step', ...s.toJson()}),
        );
        writeProbeResult(configFile.parent.path, result);
        emit({'type': ServiceEvents.probe, 'kind': 'done', ...result.toJson()});
      });
    } on Object catch (e, st) {
      Log.e('probe', 'context test failed', e, st);
      emit({'type': ServiceEvents.probe, 'kind': 'error', 'message': friendlyLlmError(e)});
    } finally {
      _probing = false;
    }
  }

  Future<void> _speak(String text) async {
    try {
      // Don't transcribe our own voice.
      final wasListening = _mic != null;
      if (wasListening) await _stopMic();
      await _tts.speak(text);
      if (wasListening) await _applyListening();
    } on Object catch (e) {
      Log.w('tts', 'speech output failed', e);
    }
  }

  Future<String> _runAutomation(String name) async {
    final rule = rules.all().where((r) => r.name.toLowerCase() == name.trim().toLowerCase()).firstOrNull;
    if (rule == null) return 'No automation called "$name".';
    if (!rule.enabled) return '"${rule.name}" is turned off.';
    final now = DateTime.now();
    await executor.execute(RuleFire(rule, {
      'text': 'Run by the assistant',
      'speaker': 'Vox',
      'match': rule.name,
      'command': rule.name,
      'time_iso': now.toIso8601String(),
      'date': now.toIso8601String().substring(0, 10),
      'time': now.toIso8601String().substring(11, 16),
      'segment_id': '0',
      'conversation_id': '0',
      'rule': rule.name,
    }));
    rules.markFired(rule.id, now);
    return 'Ran "${rule.name}".';
  }

  // ---- controls --------------------------------------------------------

  Future<void> pause() async {
    _manualPause = true;
    _pausedUntil = null;
    _reason = 'Paused';
    await _applyListening();
  }

  Future<void> resume() async {
    _manualPause = false;
    _pausedUntil = null;
    await _applyListening();
  }

  /// Pauses for [minutes] (0 = until resumed). Used by the assistant.
  Future<void> pauseFor(int minutes) async {
    if (minutes <= 0) return pause();
    _pausedUntil = DateTime.now().add(Duration(minutes: minutes));
    _reason = 'Paused for $minutes min';
    await _applyListening();
  }

  set holdTranscription(bool on) => scheduler.transcriptionPaused = on;

  /// Picks up changes made in the app (people, rules, settings, models).
  Future<void> reload() async {
    final fresh = ServiceConfig.readFrom(configFile);
    if (fresh != null) {
      final llmChanged = fresh.llm?.path != _config.llm?.path;
      final old = _config;
      _config = fresh;
      _applyListeningSettings(old, fresh);
      processor.identifier.config = fresh.settings.identifierConfig;
      processor.splitSpeakers = fresh.settings.splitSpeakers;
      // The speaker-change model may have finished downloading meanwhile.
      if (processor.diarizer == null && fresh.paths.diarizer != null) {
        processor.diarizer = _loadDiarizer(fresh.paths.diarizer);
      }
      // Likewise the tone model. It is only kept in memory while switched on
      // (it takes about 250 MB), and swapped if a new copy was installed.
      processor.hearTone = fresh.settings.hearTone;
      final toneWanted = fresh.settings.hearTone && fresh.paths.toneModel != null;
      if (!toneWanted || fresh.paths.toneModel != old.paths.toneModel) {
        processor.tone?.dispose();
        processor.tone = null;
      }
      if (toneWanted && processor.tone == null) processor.tone = _loadTone(fresh.paths);
      handler.wakeParser = WakeCommandParser(fresh.settings.wakePhrases);
      if (llmChanged) await llm.unload();
    }
    rules.invalidate();
    processor.identifier.updateProfiles(speakers.profiles(), speakers.clusters());
    await _checkLocation(force: true);
    await _applyListening();
    reviewWorker.kick();
  }

  /// Settings that change how sound is heard, applied while running.
  void _applyListeningSettings(ServiceConfig old, ServiceConfig fresh) {
    final a = old.settings;
    final b = fresh.settings;
    capture.gain = b.micGain;
    if (a.vadThreshold != b.vadThreshold || a.pauseSeconds != b.pauseSeconds || a.minSpeechSeconds != b.minSpeechSeconds) {
      try {
        _enqueue(capture.swapVad(_makeVad(fresh)));
      } on Object catch (e, st) {
        Log.e('listen', 'speech detector could not restart; keeping the old settings', e, st);
      }
    }
    processor.config = ProcessorConfig(minPartWords: b.splitMinWords);
    processor.turns = SpeakerTurns(minTurnSeconds: b.splitMinSeconds);
    if (fresh.paths.encoder != _asrEncoder) {
      final missing = fresh.paths.missingFiles();
      if (missing.isNotEmpty) {
        Log.w('listen', 'speech model files missing; keeping the current model: $missing');
        return;
      }
      try {
        processor.swapAsr(SherpaParakeetAsr(fresh.paths));
        _asrEncoder = fresh.paths.encoder;
      } on Object catch (e, st) {
        Log.e('listen', 'speech model could not be switched; keeping the old one', e, st);
      }
    }
  }

  /// Periodic work (every ~30 s).
  Future<void> tick() async {
    if (_pausedUntil != null && DateTime.now().isAfter(_pausedUntil!)) {
      _pausedUntil = null;
      await _applyListening();
    }
    await dispatcher.flush();
    for (final r in reminders.takeDue()) {
      await notifier.show('Reminder', r.text);
      if (_config.settings.speakReplies) await _speak('Reminder: ${r.text}');
    }
    await _checkLocation();
    _housekeeping();
    scheduler.kick();
    reviewWorker.kick();
    emitStatus();
  }

  Future<void> _checkLocation({bool force = false}) async {
    final s = _config.settings;
    if (s.locationMode == LocationMode.off || s.places.isEmpty) {
      if (_autoPause) {
        _autoPause = false;
        await _applyListening();
      }
      return;
    }
    final now = DateTime.now();
    if (!force && now.difference(_lastLocationCheck) < const Duration(minutes: 2)) return;
    _lastLocationCheck = now;
    if (await LocationMonitor.access() != LocationAccess.granted) {
      _reason = 'Location permission needed for place rules';
      return;
    }
    final fix = await _location.current();
    if (fix == null) return; // keep the previous decision
    final d = _policy.decide(
      mode: s.locationMode,
      places: s.places,
      lat: fix.lat,
      lon: fix.lon,
      accuracyM: fix.accuracyM,
      currentPlaceId: _placeId,
    );
    _placeId = d.place?.id;
    final changed = _autoPause == d.listen;
    _autoPause = !d.listen;
    _reason = d.reason;
    if (changed) await _applyListening();
  }

  void _housekeeping({bool force = false}) {
    final now = DateTime.now();
    if (!force && now.difference(_lastHousekeeping) < const Duration(hours: 1)) return;
    _lastHousekeeping = now;
    final days = _config.settings.retentionDays;
    if (days > 0) transcripts.deleteOlderThan(now.subtract(Duration(days: days)));
    outbox.purgeFinishedBefore(now.subtract(const Duration(days: 14)));
  }

  String statusTitle() => switch (state) {
        ListenState.listening => 'Vox is listening',
        ListenState.paused => 'Vox is paused',
        ListenState.autoPaused => 'Vox is paused here',
        ListenState.starting => 'Vox is starting',
        ListenState.error => 'Vox needs attention',
      };

  String statusText() {
    final parts = <String>[_reason];
    if (scheduler.activity == SchedulerActivity.assistant) parts.add('thinking');
    if (scheduler.activity == SchedulerActivity.reviewing || reviewWorker.isRunning) parts.add('reviewing');
    final backlog = scheduler.backlog;
    if (backlog > 0) parts.add('$backlog waiting to transcribe');
    if (_today > 0) parts.add('$_today heard today');
    return parts.join(' · ');
  }

  void emitStatus() => emit({
        'type': ServiceEvents.status,
        'state': state.name,
        'reason': _reason,
        'backlog': scheduler.backlog,
        'activity': scheduler.activity.name,
        'today': _today,
        'errors': processor.stats.errors,
        'assistantReady': _config.llm != null,
      });

  Future<void> stop() async {
    _stopped = true;
    reviewWorker.stop();
    _micRetry?.cancel();
    await _mic?.cancel();
    _mic = null;
    try {
      await _source.stop();
      await _source.dispose();
      _enqueue(capture.flush());
    } on Object catch (e) {
      Log.w('listen', 'stopping microphone failed', e);
    }
    await llm.unload();
    capture.dispose();
    processor.dispose();
    db.close();
  }
}

class _Stopped implements Exception {
  const _Stopped();
}
