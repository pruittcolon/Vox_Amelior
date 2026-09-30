import 'dart:async';
import 'dart:collection';

import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/log.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/pipeline/chunk_queue.dart';
import 'package:vox_amelior_mobile/pipeline/segment_processor.dart';

enum SchedulerActivity { idle, transcribing, assistant, reviewing }

/// Runs heavy work one job at a time so transcription and the assistant
/// never compete for the phone's CPU/GPU and memory bandwidth.
///
/// Assistant jobs (someone is waiting for an answer) go first; transcription
/// drains the speech queue in between. Background jobs (review steps) run
/// only when nothing else is waiting, one small step at a time, so live
/// speech never falls far behind. Captured audio waits safely on disk.
class ComputeScheduler {
  ComputeScheduler({required this.queue, required this.processor, this.onSegment, this.onActivity, this.clock = systemClock});

  final ChunkQueue queue;
  final SegmentProcessor processor;

  /// Every line twice: [LineStage.fast] as soon as its sentence is
  /// transcribed, [LineStage.finished] after the chunk pass.
  final void Function(SegmentView segment, LineStage stage)? onSegment;
  final Clock clock;
  Timer? _wake;
  final void Function(SchedulerActivity activity, int backlog)? onActivity;

  final Queue<Future<void> Function()> _jobs = Queue();
  final Queue<Future<void> Function()> _background = Queue();
  bool _running = false;
  bool _transcriptionPaused = false;
  SchedulerActivity _activity = SchedulerActivity.idle;

  SchedulerActivity get activity => _activity;
  int get backlog => queue.length;

  /// Stops transcribing (audio keeps queueing) — e.g. while enrolling voices.
  set transcriptionPaused(bool value) {
    _transcriptionPaused = value;
    if (!value) kick();
  }

  /// Queues exclusive work (an assistant answer) and returns its result.
  Future<T> runExclusive<T>(Future<T> Function() job) {
    final done = Completer<T>();
    _jobs.add(() async {
      try {
        done.complete(await job());
      } catch (e, st) {
        done.completeError(e, st);
      }
    });
    kick();
    return done.future;
  }

  /// Queues low-priority work (one review step); runs when idle.
  Future<T> runBackground<T>(Future<T> Function() job) {
    final done = Completer<T>();
    _background.add(() async {
      try {
        done.complete(await job());
      } catch (e, st) {
        done.completeError(e, st);
      }
    });
    kick();
    return done.future;
  }

  /// Starts working through queued jobs and speech if not already running.
  void kick() {
    if (!_running) unawaited(_pump());
  }

  Future<void> _pump() async {
    _running = true;
    try {
      while (true) {
        if (_jobs.isNotEmpty) {
          _set(SchedulerActivity.assistant);
          await _jobs.removeFirst()();
          continue;
        }
        // Held (e.g. while voice samples are recorded): background waits too.
        if (_transcriptionPaused) break;
        final chunk = queue.take();
        if (chunk == null) {
          if (processor.refineDue(clock())) {
            _refine();
            continue;
          }
          if (_background.isEmpty) break;
          _set(SchedulerActivity.reviewing);
          await _background.removeFirst()();
          continue;
        }
        _set(SchedulerActivity.transcribing);
        try {
          for (final segment in processor.transcribe(chunk.read(), chunk.startedAt)) {
            onSegment?.call(segment, LineStage.fast);
          }
        } on Object catch (e, st) {
          processor.stats.errors++;
          Log.e('scheduler', 'transcription failed', e, st);
        } finally {
          queue.done(chunk);
        }
        // A complete chunk is finished right away, even with speech waiting.
        if (processor.hasReadyChunk) _refine(readyOnly: true);
        // Let microphone data and messages from the app through.
        await Future<void>.delayed(Duration.zero);
      }
    } finally {
      _running = false;
      _set(SchedulerActivity.idle);
      // Lines waiting for the pause that ends their chunk: look again then.
      _wake?.cancel();
      if (processor.hasPending) _wake = Timer(processor.config.chunkPause + const Duration(milliseconds: 250), kick);
    }
  }

  void _refine({bool readyOnly = false}) {
    _set(SchedulerActivity.transcribing);
    try {
      for (final segment in processor.refine(clock(), readyOnly: readyOnly)) {
        onSegment?.call(segment, LineStage.finished);
      }
    } on Object catch (e, st) {
      processor.stats.errors++;
      Log.e('scheduler', 'finishing lines failed', e, st);
    }
  }

  void _set(SchedulerActivity a) {
    _activity = a;
    onActivity?.call(a, backlog);
  }
}
