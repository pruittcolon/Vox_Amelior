import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/search/embedder_worker.dart';
import 'package:vox_amelior_mobile/search/embedding.dart';

import 'support/fakes.dart';

/// Stands in for EmbeddingGemma in the worker isolate. The "model path" is
/// a folder: each time the model is released a line is added to a file
/// there, so the test (in another isolate) can see it happen. The
/// "tokenizer path" says how loading should go.
TextEmbedder _load(String dir, String how) {
  if (how == 'broken') throw StateError('not a model');
  if (how == 'slow') sleep(const Duration(milliseconds: 500));
  return _Released(dir);
}

class _Released extends FakeTextEmbedder {
  _Released(this.dir);

  final String dir;

  @override
  void dispose() {
    super.dispose();
    File('$dir/released').writeAsStringSync('released\n', mode: FileMode.append);
  }
}

/// The worker that runs EmbeddingGemma off the UI isolate.
void main() {
  late Directory dir;

  setUp(() => dir = Directory.systemTemp.createTempSync('vox_embedder_'));
  tearDown(() => dir.deleteSync(recursive: true));

  IsolateEmbedder worker({String how = 'ok', Duration idle = const Duration(minutes: 2)}) =>
      IsolateEmbedder(modelPath: dir.path, tokenizerPath: how, load: _load, idle: idle);

  int releases() {
    final f = File('${dir.path}/released');
    return f.existsSync() ? f.readAsLinesSync().length : 0;
  }

  Future<void> until(bool Function() done) async {
    final watch = Stopwatch()..start();
    while (!done()) {
      if (watch.elapsed > const Duration(seconds: 10)) fail('timed out');
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
  }

  test('embeds in the worker, each request getting its own answer', () async {
    final e = worker();
    final texts = ['pay the electric bill', 'the puppy chewed my shoe', 'dentist on Tuesday'];
    final results = await Future.wait([for (final t in texts) e.embed([t], task: EmbedTask.document)]);
    for (var i = 0; i < texts.length; i++) {
      expect(results[i].single, FakeTextEmbedder.vector(texts[i]));
    }
    expect(await e.embed(const [], task: EmbedTask.query), isEmpty);
    await e.close();
  });

  test('closing releases the model, and the next use loads it again', () async {
    final e = worker();
    await e.embed(['one'], task: EmbedTask.document);
    expect(releases(), 0);
    await e.close();
    expect(releases(), 1, reason: 'the model is released before close returns');
    expect((await e.embed(['two'], task: EmbedTask.query)).single, FakeTextEmbedder.vector('two'));
    await e.close();
    expect(releases(), 2);
    await e.close(); // closing twice is fine
  });

  test('releases the model by itself after a while without work', () async {
    final e = worker(idle: const Duration(milliseconds: 100));
    await e.embed(['one'], task: EmbedTask.document);
    await until(() => releases() == 1);
    expect((await e.embed(['two'], task: EmbedTask.document)).single, FakeTextEmbedder.vector('two'));
    await e.close();
  });

  test('closed while the model is still loading: no wait forever, and the model is still released', () async {
    final e = worker(how: 'slow');
    final pending = expectLater(e.embed(['one'], task: EmbedTask.document), throwsStateError);
    await Future<void>.delayed(const Duration(milliseconds: 100));
    await e.close();
    await pending;
    expect(releases(), 1, reason: 'released as soon as it finished loading');
  });

  test('a model that will not load is reported, and tried again next time', () async {
    final e = worker(how: 'broken');
    await expectLater(
      e.embed(['one'], task: EmbedTask.document),
      throwsA(isA<StateError>().having((s) => s.message, 'message', contains('could not start'))),
    );
    await expectLater(e.embed(['one'], task: EmbedTask.document), throwsStateError);
    await e.close();
    expect(releases(), 0);
  });
}
