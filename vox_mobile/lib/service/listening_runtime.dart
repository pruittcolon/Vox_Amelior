import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/automation/segment_handler.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/data/transcript_repository.dart';
import 'package:vox_amelior_mobile/native/pcm_source.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/pipeline/listening_pipeline.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';
import 'package:vox_amelior_mobile/speakers/speaker_identifier.dart';

/// Messages from the listening service to the app.
abstract final class ServiceEvents {
  static const segment = 'segment';
  static const request = 'request';
  static const status = 'status';
  static const error = 'error';
}

/// Commands from the app to the listening service.
abstract final class ServiceCommands {
  static const pause = 'pause';
  static const resume = 'resume';
  static const reload = 'reload';
  static const ackRequest = 'ack';
}

/// Everything that runs while Vox is listening: microphone, speech pipeline,
/// automations and housekeeping. Lives in the foreground-service isolate.
class ListeningRuntime {
  ListeningRuntime._({
    required this.configFile,
    required this.db,
    required this.speakers,
    required this.transcripts,
    required this.rules,
    required this.outbox,
    required this.dispatcher,
    required this.handler,
    required this.pipeline,
    required this.source,
    required this.emit,
    required this.notifier,
  });

  final File configFile;
  final AppDatabase db;
  final SpeakerRepository speakers;
  final TranscriptRepository transcripts;
  final RuleRepository rules;
  final OutboxRepository outbox;
  final WebhookDispatcher dispatcher;
  final SegmentHandler handler;
  final ListeningPipeline pipeline;
  final PcmSource source;
  final Notifier notifier;

  /// Sends an event to the app (if it is running).
  final void Function(Map<String, Object?> event) emit;

  StreamSubscription<Uint8List>? _mic;
  bool _paused = false;
  int _segmentsToday = 0;
  DateTime _day = DateTime.now();
  DateTime _lastHousekeeping = DateTime.fromMillisecondsSinceEpoch(0);
  final Set<int> _acked = {};

  bool get isPaused => _paused;

  static Future<ListeningRuntime> start({
    required File configFile,
    required void Function(Map<String, Object?> event) emit,
    required Notifier notifier,
  }) async {
    final config = ServiceConfig.readFrom(configFile);
    if (config == null) {
      throw StateError('Speech models are not set up yet. Open Vox and finish Setup.');
    }
    final db = AppDatabase.open(config.dbPath);
    final speakers = SpeakerRepository(db);
    final transcripts = TranscriptRepository(db);
    final rules = RuleRepository(db);
    final outbox = OutboxRepository(db);
    final dispatcher = WebhookDispatcher(outbox, DioHttpSender());
    final handler = SegmentHandler(
      rules: rules,
      engine: RuleEngine(),
      executor: ActionExecutor(outbox: outbox, dispatcher: dispatcher, notes: NoteRepository(db), notifier: notifier),
      requests: AssistantRequestRepository(db),
      wakeParser: WakeCommandParser(config.settings.wakePhrases),
    );

    late ListeningRuntime runtime;
    final pipeline = ListeningPipeline(
      vad: SherpaVad(modelPath: config.paths.vad, threshold: config.settings.vadThreshold),
      asr: SherpaParakeetAsr(config.paths),
      embedder: SherpaSpeakerEmbedder(config.paths.speaker),
      identifier: SpeakerIdentifier(
        profiles: speakers.profiles(),
        clusters: speakers.clusters(),
        config: config.settings.identifierConfig,
        newClusterId: SpeakerRepository.newId,
        nextGuestLabel: speakers.nextGuestLabel,
      ),
      transcripts: transcripts,
      speakers: speakers,
      onSegment: (s) => runtime._onSegment(s),
    );
    runtime = ListeningRuntime._(
      configFile: configFile,
      db: db,
      speakers: speakers,
      transcripts: transcripts,
      rules: rules,
      outbox: outbox,
      dispatcher: dispatcher,
      handler: handler,
      pipeline: pipeline,
      source: MicrophonePcmSource(),
      emit: emit,
      notifier: notifier,
    );
    await runtime._startMic();
    runtime._housekeeping(force: true);
    return runtime;
  }

  Future<void> _startMic() async {
    final stream = await source.start();
    _mic = stream.listen(
      (bytes) {
        if (!_paused) pipeline.addPcm16(bytes);
      },
      onError: (Object e) {
        Log.w('listen', 'microphone error', e);
        emit({'type': ServiceEvents.error, 'message': 'Microphone error: $e'});
      },
    );
  }

  Future<void> pause() async {
    if (_paused) return;
    _paused = true;
    await _mic?.cancel();
    _mic = null;
    await source.stop();
    pipeline.flush();
    emit(_status());
  }

  Future<void> resume() async {
    if (!_paused) return;
    _paused = false;
    await _startMic();
    emit(_status());
  }

  /// Picks up changes made in the app (people, rules, settings).
  void reload() {
    final config = ServiceConfig.readFrom(configFile);
    rules.invalidate();
    pipeline.identifier.updateProfiles(speakers.profiles(), speakers.clusters());
    if (config != null) {
      pipeline.identifier.config = config.settings.identifierConfig;
      handler.wakeParser = WakeCommandParser(config.settings.wakePhrases);
    }
  }

  void acknowledge(int requestId) => _acked.add(requestId);

  /// Periodic work: deliver webhooks, apply retention, refresh status.
  Future<void> tick() async {
    await dispatcher.flush();
    _housekeeping();
    emit(_status());
  }

  String statusText() {
    if (_paused) return 'Paused';
    return _segmentsToday == 0 ? 'Listening' : 'Listening · $_segmentsToday things heard today';
  }

  Future<void> stop() async {
    await _mic?.cancel();
    await source.stop();
    await source.dispose();
    try {
      pipeline.flush();
    } on Object catch (e) {
      Log.w('listen', 'flush on stop failed', e);
    }
    pipeline.dispose();
    db.close();
  }

  void _onSegment(SegmentView segment) {
    final now = DateTime.now();
    if (now.day != _day.day) {
      _day = now;
      _segmentsToday = 0;
    }
    _segmentsToday++;
    emit({'type': ServiceEvents.segment, 'id': segment.id});
    unawaited(_afterSegment(segment));
  }

  Future<void> _afterSegment(SegmentView segment) async {
    try {
      final outcome = await handler.handle(segment);
      final requestId = outcome.assistantRequestId;
      if (requestId == null) return;
      emit({'type': ServiceEvents.request, 'id': requestId});
      // If the app does not pick the question up, tell the user.
      await Future<void>.delayed(const Duration(seconds: 3));
      if (!_acked.remove(requestId)) {
        await notifier.show('Vox heard a question', '"${outcome.wakeCommand}" — tap to open Vox and get the answer.');
      }
    } on Object catch (e, st) {
      Log.e('listen', 'segment follow-up failed', e, st);
    }
  }

  void _housekeeping({bool force = false}) {
    final now = DateTime.now();
    if (!force && now.difference(_lastHousekeeping) < const Duration(hours: 1)) return;
    _lastHousekeeping = now;
    final config = ServiceConfig.readFrom(configFile);
    final days = config?.settings.retentionDays ?? 0;
    if (days > 0) transcripts.deleteOlderThan(now.subtract(Duration(days: days)));
    outbox.purgeFinishedBefore(now.subtract(const Duration(days: 14)));
  }

  Map<String, Object?> _status() => {
        'type': ServiceEvents.status,
        'paused': _paused,
        'today': _segmentsToday,
        'errors': pipeline.stats.errors,
        'lastMs': pipeline.stats.lastProcessing.inMilliseconds,
      };
}
