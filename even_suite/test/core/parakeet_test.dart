import 'dart:io';
import 'dart:typed_data';

import 'package:even_suite/core/voice/parakeet_model.dart';
import 'package:even_suite/core/voice/parakeet_speech.dart';
import 'package:even_suite/core/voice/speech.dart';
import 'package:flutter_test/flutter_test.dart';

import '../helpers/fakes.dart';

/// Talking = any loud sample; a stretch ends after [quietChunks] quiet chunks.
class _FakeVad implements SpeechDetector {
  static const quietChunks = 3;
  final _cur = <double>[];
  final _done = <Float32List>[];
  int _quiet = 0;
  bool _talking = false;

  @override
  void accept(Float32List s) {
    final loud = s.any((v) => v.abs() > 0.1);
    if (loud) {
      _talking = true;
      _quiet = 0;
      _cur.addAll(s);
    } else if (_talking) {
      _quiet++;
      if (_quiet >= quietChunks) {
        _done.add(Float32List.fromList(_cur));
        _cur.clear();
        _talking = false;
      }
    }
  }

  @override
  bool get talking => _talking;

  @override
  List<Float32List> take() {
    final out = List.of(_done);
    _done.clear();
    return out;
  }

  @override
  void flush() {
    if (_cur.isNotEmpty) _done.add(Float32List.fromList(_cur));
    _cur.clear();
    _talking = false;
  }

  @override
  void reset() {
    _cur.clear();
    _done.clear();
    _talking = false;
  }
}

/// "word" for every 0.2 s of sound.
class _FakeAsr implements Transcriber {
  int calls = 0;
  @override
  String transcribe(Float32List s) {
    calls++;
    return List.filled(s.length ~/ 3200, 'word').join(' ');
  }
}

Float32List _chunk(bool loud) => Float32List.fromList(List.filled(1600, loud ? 0.3 : 0.0));

void main() {
  test('PCM bytes to samples, and a loudness on Android\'s scale', () {
    final bytes = Uint8List.fromList([0x00, 0x40, 0x00, 0xC0]); // 16384, -16384
    final s = pcmToFloat(bytes);
    expect(s, [0.5, -0.5]);
    expect(loudness(Float32List(1600)), -2);
    expect(loudness(Float32List.fromList(List.filled(1600, 0.3))), greaterThan(6));
  });

  group('live loop', () {
    test('words while you talk, the final text when you pause', () {
      final asr = _FakeAsr();
      final loop = UtteranceLoop(asr, _FakeVad(), partialEvery: 0.3);
      final events = <Map<String, Object>>[];
      for (var i = 0; i < 10; i++) {
        events.addAll(loop.onSamples(_chunk(true), clock: Stopwatch()));
      }
      final partials = events.where((e) => e['type'] == 'partial').toList();
      expect(events.first['type'], 'level');
      expect(events.any((e) => e['type'] == 'begin'), isTrue);
      expect(partials.length, greaterThanOrEqualTo(2), reason: 'the words so far, more than once');
      expect(events.any((e) => e['type'] == 'final'), isFalse, reason: 'still talking');
      for (var i = 0; i < 3; i++) {
        events.addAll(loop.onSamples(_chunk(false), clock: Stopwatch()));
      }
      final fin = events.where((e) => e['type'] == 'final').single;
      expect('${fin['text']}'.split(' ').length, 5, reason: '1 s of speech');
      expect(fin['finishing'], isFalse);
    });

    test('finish: what was said so far, or an empty end', () {
      final loop = UtteranceLoop(_FakeAsr(), _FakeVad());
      for (var i = 0; i < 4; i++) {
        loop.onSamples(_chunk(true), clock: Stopwatch());
      }
      final out = loop.finish();
      expect(out.single['type'], 'final');
      expect(out.single['finishing'], isTrue);
      expect(loop.finish().single, {'type': 'end', 'finishing': true});
    });

    test('quiet sound is never decoded', () {
      final asr = _FakeAsr();
      final loop = UtteranceLoop(asr, _FakeVad());
      for (var i = 0; i < 20; i++) {
        loop.onSamples(_chunk(false));
      }
      expect(asr.calls, 0);
    });
  });

  test('switching engines: events only from the one in use', () async {
    final android = FakeSpeech();
    final parakeet = FakeSpeech();
    final sw = SwitchableSpeech(android, parakeet);
    final seen = <String>[];
    sw.events.listen((e) => seen.add('${e.type}:${e.text}'));
    await sw.start();
    expect(android.running, isTrue);
    android.partial('from android');
    sw.useParakeet = true;
    await sw.start();
    expect(android.running, isFalse, reason: 'the old one is stopped');
    expect(parakeet.running, isTrue);
    android.partial('stray');
    parakeet.partial('from parakeet');
    await Future<void>.delayed(Duration.zero);
    expect(seen.where((s) => s.startsWith('partial')), ['partial:from android', 'partial:from parakeet']);
    await sw.finish();
    expect(parakeet.finishes, 1);
  });

  group('model download', () {
    late HttpServer server;
    late Directory dir;
    final big = Uint8List.fromList(List.generate(300000, (i) => i % 251));
    final small = Uint8List.fromList(List.generate(1234, (i) => i % 7));
    var rangeSeen = '';

    setUp(() async {
      dir = await Directory.systemTemp.createTemp('model');
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((req) async {
        final res = req.response;
        if (req.uri.path == '/redirect/big.bin') {
          res.statusCode = 302;
          res.headers.set('location', '/files/big.bin');
          await res.close();
          return;
        }
        final data = req.uri.path.endsWith('big.bin') ? big : small;
        final range = req.headers.value('range');
        if (range != null) {
          rangeSeen = range;
          final from = int.parse(range.substring(6, range.length - 1));
          res.statusCode = 206;
          res.add(data.sublist(from));
        } else {
          res.add(data);
        }
        await res.close();
      });
    });

    tearDown(() async {
      await server.close(force: true);
      await dir.delete(recursive: true);
    });

    SpeechModel model() {
      final base = 'http://127.0.0.1:${server.port}';
      return SpeechModel(() async => dir.path, folder: 'm', files: [
        ModelFile('big.bin', '$base/redirect/big.bin', big.length),
        ModelFile('small.txt', '$base/files/small.txt', small.length),
      ]);
    }

    test('downloads every file (through a redirect) and is ready', () async {
      final m = model();
      await m.check();
      expect(m.state, ModelState.missing);
      await m.download();
      expect(m.state, ModelState.ready, reason: m.error ?? '');
      expect(File('${dir.path}/m/big.bin').readAsBytesSync(), big);
      expect(m.progress, 1.0);
      final again = model();
      await again.check();
      expect(again.ready, isTrue);
      await again.delete();
      expect(again.state, ModelState.missing);
      expect(Directory('${dir.path}/m').existsSync(), isFalse);
    });

    test('picks up where a cut download stopped', () async {
      await Directory('${dir.path}/m').create(recursive: true);
      File('${dir.path}/m/big.bin.part').writeAsBytesSync(big.sublist(0, 100000));
      final m = model();
      await m.check();
      expect(m.received, 100000);
      await m.download();
      expect(m.ready, isTrue, reason: m.error ?? '');
      expect(rangeSeen, 'bytes=100000-', reason: 'the Range survives the redirect');
      expect(File('${dir.path}/m/big.bin').readAsBytesSync(), big);
    });

    test('a wrong size is an error, not a broken model', () async {
      final base = 'http://127.0.0.1:${server.port}';
      final m = SpeechModel(() async => dir.path,
          folder: 'm', files: [ModelFile('small.txt', '$base/files/small.txt', small.length + 5)]);
      await m.download();
      expect(m.state, ModelState.failed);
      expect(File('${dir.path}/m/small.txt').existsSync(), isFalse);
    });
  });

  test('no Parakeet until it is downloaded', () async {
    final m = SpeechModel(() async => null);
    final p = ParakeetSpeech(m);
    expect(await p.available(), isFalse);
    final errors = <SpeechEvent>[];
    p.events.listen(errors.add);
    await p.start();
    await Future<void>.delayed(Duration.zero);
    expect(errors.single.type, 'error');
    expect(errors.single.message, contains('Download'));
  });
}
