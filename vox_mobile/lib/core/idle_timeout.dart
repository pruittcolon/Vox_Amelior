import 'dart:async';

/// Ends [source] with a [TimeoutException] when it produces nothing for
/// [idle] (the source is cancelled, which also stops the model).
///
/// Unlike [Stream.timeout] this completes promptly in test zones too.
Stream<T> withIdleTimeout<T>(Stream<T> source, Duration idle) {
  late final StreamController<T> out;
  StreamSubscription<T>? sub;
  Timer? timer;

  void arm() {
    timer?.cancel();
    timer = Timer(idle, () {
      final s = sub;
      sub = null;
      unawaited(s?.cancel());
      out
        ..addError(TimeoutException('No output for ${idle.inSeconds} s', idle))
        ..close();
    });
  }

  out = StreamController<T>(
    onListen: () {
      arm();
      sub = source.listen(
        (e) {
          arm();
          out.add(e);
        },
        onError: (Object e, StackTrace st) {
          arm();
          out.addError(e, st);
        },
        onDone: () {
          timer?.cancel();
          sub = null;
          out.close();
        },
      );
    },
    onPause: () => sub?.pause(),
    onResume: () => sub?.resume(),
    onCancel: () {
      timer?.cancel();
      final s = sub;
      sub = null;
      return s?.cancel();
    },
  );
  return out.stream;
}
