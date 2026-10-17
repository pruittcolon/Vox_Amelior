/// Live listening: renders ahead of the speaker, follows along sentence by
/// sentence, and saves where you stopped.
library;

import 'dart:async';
import 'dart:collection';

import 'package:flutter/foundation.dart';

import '../app_state.dart';
import '../render/audio.dart';
import '../render/renderer.dart';
import 'native.dart';

enum PlayState { idle, buffering, playing, paused, ended, error }

class Player extends ChangeNotifier {
  Player(this.app, this.native) {
    _mediaSub = native.mediaActions.listen(_onMedia);
  }

  final AppState app;
  final NativeApi native;
  late final StreamSubscription<String> _mediaSub;

  /// How far ahead of the speaker to render.
  static const aheadSeconds = 25.0;

  Session? session;
  PlayState state = PlayState.idle;
  int current = 0;
  String? error;

  /// Seconds of audio produced per second of work (above 1 keeps up).
  double speedRatio = 0;

  int _gen = 0;
  Renderer? _renderer;
  final _queue = Queue<RenderChunk>();
  int _queuedFrames = 0;
  int _written = 0;
  bool _producerDone = false;
  final _timeline = <(int frame, int utterance)>[];
  Timer? _poll;
  DateTime _lastSave = DateTime.now();
  int _renderedFrames = 0;
  final _renderWatch = Stopwatch();

  bool get active => state == PlayState.playing || state == PlayState.buffering || state == PlayState.paused;
  bool get isPlaying => state == PlayState.playing || state == PlayState.buffering;

  Future<void> play(Session s, {int? from}) async {
    final start = (from ?? s.doc.position).clamp(0, s.plan.utterances.isEmpty ? 0 : s.plan.utterances.length - 1);
    await _halt();
    final gen = ++_gen;
    session = s;
    current = start;
    error = null;
    state = PlayState.buffering;
    notifyListeners();
    try {
      await native.playerOpen(outputRate);
    } catch (e) {
      _fail('Could not open the speaker: $e');
      return;
    }
    _renderer = Renderer(
      plan: s.plan,
      synth: app.synth,
      voiceFor: s.voiceFor,
      theme: s.theme,
      startUtterance: start,
      onError: (u, e) => debugPrint('sentence $u failed: $e'),
    );
    _queue.clear();
    _queuedFrames = 0;
    _written = 0;
    _producerDone = false;
    _timeline.clear();
    _renderedFrames = 0;
    _renderWatch
      ..reset()
      ..start();
    unawaited(_service());
    unawaited(_produce(gen));
    unawaited(_feed(gen));
    _poll = Timer.periodic(const Duration(milliseconds: 120), (_) => _tick(gen));
  }

  Future<void> _produce(int gen) async {
    final r = _renderer!;
    while (gen == _gen) {
      final position = await native.playerPosition();
      if (gen != _gen) return;
      final ahead = (_written + _queuedFrames - position) / outputRate;
      if (ahead > aheadSeconds) {
        await Future<void>.delayed(const Duration(milliseconds: 250));
        continue;
      }
      RenderChunk? chunk;
      try {
        chunk = await r.next();
      } catch (e) {
        if (gen == _gen) _fail(e.toString());
        return;
      }
      if (gen != _gen) return;
      if (chunk == null) {
        _producerDone = true;
        _renderWatch.stop();
        return;
      }
      _renderedFrames += chunk.samples.length;
      final secs = _renderWatch.elapsedMilliseconds / 1000;
      if (secs > 0.5) speedRatio = _renderedFrames / outputRate / secs;
      _queue.add(chunk);
      _queuedFrames += chunk.samples.length;
    }
  }

  Future<void> _feed(int gen) async {
    while (gen == _gen) {
      if (_queue.isEmpty) {
        if (_producerDone) return;
        await Future<void>.delayed(const Duration(milliseconds: 40));
        continue;
      }
      final chunk = _queue.removeFirst();
      _queuedFrames -= chunk.samples.length;
      if (chunk.utterance != null) _timeline.add((_written, chunk.utterance!));
      _written += chunk.samples.length;
      try {
        await native.playerWrite(toInt16Bytes(chunk.samples));
      } catch (e) {
        if (gen == _gen) _fail('Audio output failed: $e');
        return;
      }
    }
  }

  Future<void> _tick(int gen) async {
    if (gen != _gen || state == PlayState.paused) return;
    final pos = await native.playerPosition();
    if (gen != _gen || state == PlayState.paused) return;
    // Which sentence is audible now.
    var u = current;
    for (var i = _timeline.length - 1; i >= 0; i--) {
      if (_timeline[i].$1 <= pos) {
        u = _timeline[i].$2;
        break;
      }
    }
    final starving = !_producerDone && pos >= _written - outputRate ~/ 20;
    final finished = _producerDone && _queue.isEmpty && pos >= _written - outputRate ~/ 50;
    var changed = false;
    if (u != current) {
      final chapterBefore = session?.plan.chapterOf(current);
      current = u;
      changed = true;
      _saveSoon();
      if (session?.plan.chapterOf(u) != chapterBefore) unawaited(_service());
    }
    final next = finished ? PlayState.ended : (starving ? PlayState.buffering : PlayState.playing);
    if (next != state) {
      state = next;
      changed = true;
      if (finished) {
        final doc = session?.doc;
        if (doc != null) {
          // Finished: next time it starts from the top.
          doc.position = 0;
          await app.library.changed(doc);
        }
        await _halt();
        await native.serviceStop();
      } else {
        unawaited(_service());
      }
    }
    if (changed) notifyListeners();
  }

  Future<void> pause() async {
    if (!isPlaying) return;
    await native.playerPause();
    state = PlayState.paused;
    notifyListeners();
    await _persist();
    await _service();
  }

  Future<void> resume() async {
    if (state == PlayState.paused) {
      await native.playerResume();
      state = PlayState.playing;
      notifyListeners();
      await _service();
    } else if (session != null && (state == PlayState.ended || state == PlayState.idle || state == PlayState.error)) {
      await play(session!, from: state == PlayState.ended ? 0 : current);
    }
  }

  Future<void> toggle() => isPlaying ? pause() : resume();

  Future<void> jump(int utterance) async {
    final s = session;
    if (s == null) return;
    final target = utterance.clamp(0, s.plan.utterances.length - 1);
    final wasPaused = state == PlayState.paused;
    await play(s, from: target);
    if (wasPaused) await pause();
  }

  Future<void> next() => jump(current + 1);
  Future<void> previous() => jump(current - 1);

  /// Headphone and notification skip: by chapter when the episode has them.
  /// Back restarts the chapter unless you're at its start.
  Future<void> nextChapter() async {
    final plan = session?.plan;
    if (plan == null || !plan.hasChapters) return next();
    final here = plan.chapterOf(current)!;
    if (here.index + 1 < plan.chapters.length) await jump(plan.chapters[here.index + 1].first);
  }

  Future<void> previousChapter() async {
    final plan = session?.plan;
    if (plan == null || !plan.hasChapters) return previous();
    final here = plan.chapterOf(current)!;
    final back = current > here.first + 1 || here.index == 0;
    await jump(back ? here.first : plan.chapters[here.index - 1].first);
  }

  /// Restarts from the current sentence with a rebuilt session (new voices,
  /// speed or style).
  Future<void> reload(Session s) async {
    final wasActive = active;
    final at = current;
    session = s;
    if (wasActive) {
      await play(s, from: at.clamp(0, s.plan.utterances.length - 1));
    } else {
      notifyListeners();
    }
  }

  Future<void> stop() async {
    // Silence first, then save: stopping should feel instant.
    await _halt();
    state = PlayState.idle;
    notifyListeners();
    await _persist();
    await native.serviceStop();
  }

  Future<void> _halt() async {
    _gen++;
    _poll?.cancel();
    _poll = null;
    _queue.clear();
    _queuedFrames = 0;
    try {
      await native.playerStop();
    } catch (_) {}
  }

  void _fail(String message) {
    error = message;
    state = PlayState.error;
    notifyListeners();
    unawaited(_halt());
  }

  void _saveSoon() {
    if (DateTime.now().difference(_lastSave).inSeconds >= 5) unawaited(_persist());
  }

  Future<void> _persist() async {
    final s = session;
    if (s == null) return;
    _lastSave = DateTime.now();
    s.doc.position = current;
    await app.library.changed(s.doc);
  }

  Future<void> _service() async {
    final s = session;
    if (s == null) return;
    try {
      await native.serviceShow(
        kind: 'play',
        title: s.doc.title,
        text: s.plan.hasChapters
            ? 'Chapter ${s.plan.chapterOf(current)!.index + 1} of ${s.plan.chapters.length} · ${s.plan.chapterOf(current)!.title}'
            : '${s.style.name} · sentence ${current + 1} of ${s.plan.utterances.length}',
        playing: state != PlayState.paused,
      );
    } catch (_) {
      // Notifications are a nicety; playback continues without them.
    }
  }

  void _onMedia(String action) {
    switch (action) {
      case 'play':
        unawaited(resume());
      case 'pause':
        unawaited(pause());
      case 'next':
        unawaited(nextChapter());
      case 'previous':
        unawaited(previousChapter());
      case 'stop':
        unawaited(stop());
    }
  }

  @override
  void dispose() {
    _mediaSub.cancel();
    _poll?.cancel();
    super.dispose();
  }
}
