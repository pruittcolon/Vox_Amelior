import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../log.dart';

/// Something the phone's voice did ('start', 'done', 'error', 'ready').
class SpeakerEvent {
  final String type;
  final String id;
  final String message;
  const SpeakerEvent(this.type, {this.id = '', this.message = ''});

  factory SpeakerEvent.fromMap(Map m) =>
      SpeakerEvent('${m['type']}', id: '${m['id'] ?? ''}', message: '${m['message'] ?? ''}');
}

/// A voice offered by the phone's text-to-speech engine.
class TtsVoice {
  final String name;
  final String locale;
  final int quality; // Android: 500 very high .. 100 very low
  final bool network; // needs the internet
  const TtsVoice(this.name, this.locale, this.quality, this.network);

  factory TtsVoice.fromMap(Map m) => TtsVoice('${m['name']}', '${m['locale'] ?? ''}',
      (m['quality'] as num?)?.toInt() ?? 300, m['network'] == true);

  String get label {
    final q = quality >= 500 ? 'Best' : (quality >= 400 ? 'High' : 'Normal');
    return '$locale · $q${network ? ' · online' : ''} · $name';
  }
}

class TtsEngineInfo {
  final String name; // package
  final String label;
  const TtsEngineInfo(this.name, this.label);
}

/// Reads text aloud.
abstract class Speaker {
  Stream<SpeakerEvent> get events;

  /// Queue [text] (or replace what is playing when [flush]).
  Future<void> speak(String text, String id, {bool flush = false});
  Future<void> stop();

  /// [engine] '' = the phone's default; [voice] '' = the engine's default.
  /// [rate] and [pitch]: 1.0 = normal.
  Future<void> configure({String engine = '', String voice = '', double rate = 1, double pitch = 1});
  Future<List<TtsVoice>> voices();
  Future<List<TtsEngineInfo>> engines();
}

/// Android's built-in TextToSpeech through the app's own channel (no SDK).
class PlatformSpeaker implements Speaker {
  static const _methods = MethodChannel('suite/tts');
  static final Stream<SpeakerEvent> _events = const EventChannel('suite/ttsEvents')
      .receiveBroadcastStream()
      .map((e) => SpeakerEvent.fromMap(e as Map));

  @override
  Stream<SpeakerEvent> get events => _events;

  Future<T?> _call<T>(String method, [Object? args]) async {
    try {
      return await _methods.invokeMethod<T>(method, args);
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }

  @override
  Future<void> speak(String text, String id, {bool flush = false}) =>
      _call('speak', {'text': text, 'id': id, 'flush': flush});

  @override
  Future<void> stop() => _call('stop');

  @override
  Future<void> configure({String engine = '', String voice = '', double rate = 1, double pitch = 1}) =>
      _call('configure', {'engine': engine, 'voice': voice, 'rate': rate, 'pitch': pitch});

  @override
  Future<List<TtsVoice>> voices() async =>
      [for (final v in await _call<List>('voices') ?? const []) TtsVoice.fromMap(v as Map)];

  @override
  Future<List<TtsEngineInfo>> engines() async => [
        for (final e in await _call<List>('engines') ?? const [])
          TtsEngineInfo('${(e as Map)['name']}', '${e['label']}'),
      ];
}

/// Text made for listening: no Markdown, symbols, links or code read out.
class SpeechText {
  SpeechText._();

  static final _fence = RegExp(r'```[\s\S]*?(```|$)');
  static final _inlineCode = RegExp(r'`([^`]{1,40})`');
  static final _longCode = RegExp(r'`[^`]*`');
  static final _url = RegExp(r'https?://\S+');
  static final _link = RegExp(r'\[([^\]]+)\]\([^)]+\)');
  static final _marks = RegExp(r'[*_#>|~^]+');
  static final _bullet = RegExp(r'^\s*(?:[-•]|\d+[.)])\s+', multiLine: true);
  static final _emoji = RegExp(r'[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}]', unicode: true);

  static String clean(String text) {
    var t = text.replaceAll(_fence, ' (code on screen) ');
    t = t.replaceAllMapped(_link, (m) => m[1]!);
    t = t.replaceAll(_url, 'a link');
    t = t.replaceAllMapped(_inlineCode, (m) => m[1]!);
    t = t.replaceAll(_longCode, '');
    t = t.replaceAll(_bullet, '');
    t = t.replaceAll(_marks, '');
    t = t.replaceAll(_emoji, '');
    t = t.replaceAll(' -> ', ' to ').replaceAll('&', ' and ');
    return t.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  static final _end = RegExp(r'([.!?]+["”’)]*)(\s+|$)|\n\s*\n|\n(?=\s*(?:[-•]|\d+[.)])\s)');

  /// Whole sentences in [text] after [from]: (pieces to say, where the next
  /// piece starts). Short sentences are grouped so the voice flows; with
  /// [flush] the unfinished rest is included too.
  static (List<String>, int) sentences(String text, int from, {bool flush = false, int minChars = 40}) {
    final out = <String>[];
    var start = from.clamp(0, text.length);
    var pieceStart = start;
    for (final m in _end.allMatches(text, start)) {
      // A sentence end at the very end of streaming text may still grow ("3." -> "3.5").
      if (!flush && m.end >= text.length && m.group(2) != null && m.group(2)!.isEmpty) break;
      final piece = text.substring(pieceStart, m.end).trim();
      if (piece.length < minChars && m.end < text.length) continue; // keep growing this piece
      if (piece.isNotEmpty) out.add(piece);
      pieceStart = m.end;
    }
    if (flush && pieceStart < text.length) {
      final rest = text.substring(pieceStart).trim();
      if (rest.isNotEmpty) out.add(rest);
      pieceStart = text.length;
    }
    return (out, pieceStart);
  }
}

/// Speaks Claude's answer as it streams in: whole sentences go to the voice
/// while the rest is still being written, so it starts talking early and
/// flows like a podcast. [onDone] fires once everything of a finished answer
/// has been said (then the app listens for a follow-up).
class Narrator extends ChangeNotifier {
  Narrator(this.speaker) {
    _sub = speaker.events.listen(_onEvent);
  }

  final Speaker speaker;
  bool enabled = true;
  VoidCallback? onDone;

  StreamSubscription<SpeakerEvent>? _sub;
  String _msg = '';
  int _block = -1;
  int _offset = 0;
  String _live = '';
  bool _saidAny = false;
  bool _finished = false;
  int _n = 0;
  final Set<String> _queued = {};
  Timer? _safety;
  int _queuedChars = 0;

  bool get speaking => _queued.isNotEmpty;

  /// The answer so far ([live] is the text block number [block] being written).
  void feed(String msgId, int block, String live) {
    if (!enabled) return;
    if (msgId != _msg) _begin(msgId);
    if (block != _block) {
      _block = block;
      _offset = 0;
    }
    _live = live;
    final (pieces, next) = SpeechText.sentences(live, _offset);
    _offset = next;
    pieces.forEach(_say);
  }

  /// The finished answer: say whatever has not been said yet.
  void finish(String msgId, int block, String text) {
    if (!enabled) {
      onDone?.call();
      return;
    }
    if (msgId != _msg) _begin(msgId);
    String rest;
    if (!_saidAny) {
      rest = text;
    } else if (block == _block && text.startsWith(_live.substring(0, _offset.clamp(0, _live.length)))) {
      rest = text.substring(_offset.clamp(0, text.length));
    } else {
      rest = ''; // already said in other words while it was being written
    }
    final (pieces, _) = SpeechText.sentences(rest, 0, flush: true);
    pieces.forEach(_say);
    _finished = true;
    if (_queued.isEmpty) {
      _complete();
    } else {
      _armSafety();
    }
    notifyListeners();
  }

  /// Say something on its own (a short notice).
  void say(String text) {
    if (!enabled) return;
    _say(text);
  }

  Future<void> stop() async {
    final was = _queued.isNotEmpty;
    _queued.clear();
    _finished = false;
    _safety?.cancel();
    _msg = '';
    if (was) notifyListeners();
    await speaker.stop();
  }

  void _begin(String msgId) {
    _msg = msgId;
    _block = -1;
    _offset = 0;
    _live = '';
    _saidAny = false;
    _finished = false;
  }

  void _say(String piece) {
    final text = SpeechText.clean(piece);
    if (text.isEmpty) return;
    final id = 'u${_n++}';
    _queued.add(id);
    _saidAny = true;
    _queuedChars += text.length;
    speaker.speak(text, id);
    notifyListeners();
  }

  void _onEvent(SpeakerEvent e) {
    if (e.type == 'error') EventLog.add('voice', 'ttsError', {'message': e.message});
    if (e.type != 'done' && e.type != 'error') return;
    if (!_queued.remove(e.id)) return;
    if (_queued.isEmpty) {
      _queuedChars = 0;
      if (_finished) _complete();
    }
    notifyListeners();
  }

  /// If the engine never reports back, do not wait forever for it.
  void _armSafety() {
    _safety?.cancel();
    final secs = 5 + _queuedChars ~/ 12;
    _safety = Timer(Duration(seconds: secs), () {
      if (_queued.isEmpty) return;
      _queued.clear();
      if (_finished) _complete();
      notifyListeners();
    });
  }

  void _complete() {
    _finished = false;
    _safety?.cancel();
    onDone?.call();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _safety?.cancel();
    super.dispose();
  }
}
