import 'dart:async';

import 'package:vox_amelior_mobile/assistant/review_engine.dart';
import 'package:vox_amelior_mobile/assistant/review_repository.dart';
import 'package:vox_amelior_mobile/core/log.dart';

/// Works through queued reviews one small step at a time.
///
/// Two copies exist: one in the listening service (steps go through its work
/// queue, after live questions and transcription) and one in the app (used
/// only while Vox is not listening). A lease in the database makes sure only
/// one of them works on a review at a time, and lets the other take over.
class ReviewWorker {
  ReviewWorker({
    required this.engine,
    required this.reviews,
    required this.owner,
    required this.schedule,
    this.canWork,
    this.onProgress,
  });

  final ReviewEngine engine;
  final ReviewRepository reviews;

  /// 'service' or 'app'.
  final String owner;

  /// Runs one step the way this host wants (e.g. through the work queue).
  final Future<bool> Function(Future<bool> Function() step) schedule;
  final bool Function()? canWork;

  /// Called after every step with the review's id.
  final void Function(int runId)? onProgress;

  bool _running = false;
  bool _stopped = false;

  bool get isRunning => _running;

  /// Starts working if there is anything to do.
  void kick() {
    if (_running || _stopped) return;
    unawaited(_loop());
  }

  Future<void> _loop() async {
    _running = true;
    try {
      while (!_stopped && (canWork?.call() ?? true)) {
        final run = reviews.nextRunnable(owner);
        if (run == null) break;
        if (!reviews.claim(run.id, owner)) break;
        var more = false;
        try {
          more = await schedule(() => engine.step(run.id));
        } on Object catch (e, st) {
          Log.e('review', 'step failed', e, st);
          reviews.setStatus(run.id, ReviewStatus.paused, error: 'Paused after an error: $e');
        }
        onProgress?.call(run.id);
        if (!more) reviews.release(run.id, owner);
      }
    } finally {
      _running = false;
      if (!_stopped && !(canWork?.call() ?? true)) reviews.releaseAll(owner);
    }
  }

  /// Stops after the current step and hands the reviews back.
  void stop() {
    _stopped = true;
    reviews.releaseAll(owner);
  }

  void restart() {
    _stopped = false;
    kick();
  }
}
