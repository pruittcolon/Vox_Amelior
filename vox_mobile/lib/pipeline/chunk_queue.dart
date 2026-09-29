import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/log.dart';

/// A speech chunk waiting to be transcribed.
class QueuedChunk {
  const QueuedChunk(this.file, this.startedAt);

  final File file;
  final DateTime startedAt;

  Float32List read() {
    final bytes = file.readAsBytesSync();
    return Uint8List.fromList(bytes).buffer.asFloat32List(0, bytes.length ~/ 4);
  }
}

/// Speech waiting for transcription, kept on disk so nothing is lost while
/// the assistant is busy, or if Android restarts the service.
///
/// A chunk being processed is renamed to `.working`; if the process dies on
/// it, it is set aside as `.failed` on the next start instead of crashing
/// again in a loop.
class ChunkQueue {
  ChunkQueue(this.dir, {this.maxBytes = 300 * 1024 * 1024}) {
    dir.createSync(recursive: true);
    for (final f in dir.listSync().whereType<File>()) {
      if (f.path.endsWith('.working')) f.renameSync(f.path.replaceAll('.working', '.failed'));
      if (f.path.endsWith('.tmp')) f.deleteSync();
    }
  }

  final Directory dir;

  /// Oldest audio is dropped when the backlog grows beyond this.
  final int maxBytes;
  int _seq = 0;

  int get length => _pending().length;

  void push(Float32List samples, DateTime startedAt) {
    final name = '${startedAt.millisecondsSinceEpoch.toString().padLeft(15, '0')}_${(_seq++).toString().padLeft(6, '0')}';
    final tmp = File(p.join(dir.path, '$name.tmp'))
      ..writeAsBytesSync(samples.buffer.asUint8List(samples.offsetInBytes, samples.lengthInBytes), flush: true);
    tmp.renameSync(p.join(dir.path, '$name.f32'));
    _enforceLimit();
  }

  /// The oldest waiting chunk, marked as being worked on.
  QueuedChunk? take() {
    final pending = _pending();
    if (pending.isEmpty) return null;
    final f = pending.first;
    final working = f.renameSync('${f.path}.working');
    final ms = int.tryParse(p.basename(f.path).split('_').first) ?? DateTime.now().millisecondsSinceEpoch;
    return QueuedChunk(working, DateTime.fromMillisecondsSinceEpoch(ms));
  }

  void done(QueuedChunk c) {
    if (c.file.existsSync()) c.file.deleteSync();
  }

  List<File> _pending() {
    final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.f32')).toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    return files;
  }

  void _enforceLimit() {
    final files = _pending();
    var total = files.fold<int>(0, (a, f) => a + f.lengthSync());
    for (final f in files) {
      if (total <= maxBytes) break;
      total -= f.lengthSync();
      f.deleteSync();
      Log.w('queue', 'speech backlog too large; dropped oldest chunk');
    }
  }
}
