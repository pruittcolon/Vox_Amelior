import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:archive/archive_io.dart';
import 'package:path/path.dart' as p;

/// Extracts selected files from a `.tar.bz2` without loading the archive into
/// memory. Model archives are hundreds of MB, so everything streams via disk
/// and runs in a background isolate to keep the UI responsive.
class ArchiveExtractor {
  const ArchiveExtractor();

  /// Extracts the entries whose *file name* is in [wanted] into [destDir],
  /// flattening any directory structure. Only base names are used, so a
  /// malicious archive cannot write outside [destDir].
  ///
  /// Throws [StateError] if any wanted file is missing from the archive.
  ///
  /// [onProgress] reports each stage as it goes: decompressing the archive
  /// (by far the slowest part), then copying out the wanted files.
  Future<List<File>> extractTarBz2(
    File archive,
    Directory destDir, {
    required Set<String> wanted,
    void Function(ExtractProgress progress)? onProgress,
  }) async {
    final archivePath = archive.path;
    final destPath = destDir.path;
    final wantedCopy = Set<String>.of(wanted);
    final port = ReceivePort();
    final send = port.sendPort;
    final sub = port.listen((m) {
      final (stage, done, total, file) = m as (int, int, int, String?);
      onProgress?.call(ExtractProgress(ExtractStage.values[stage], done, total, file));
    });
    try {
      final paths = await _runExtract(archivePath, destPath, wantedCopy, send);
      return paths.map(File.new).toList();
    } finally {
      // Let progress messages already sent arrive before closing.
      await Future<void>.delayed(Duration.zero);
      await sub.cancel();
      port.close();
    }
  }
}

/// Top level so the isolate's closure captures only these sendable values
/// (a closure inside [ArchiveExtractor.extractTarBz2] would also capture the
/// progress callback and everything it references, e.g. an HTTP client).
Future<List<String>> _runExtract(String archivePath, String destPath, Set<String> wanted, SendPort progress) =>
    Isolate.run(() => _extractSync(archivePath, destPath, wanted, progress));

enum ExtractStage { decompressing, copying }

class ExtractProgress {
  const ExtractProgress(this.stage, this.doneBytes, this.totalBytes, [this.fileName]);

  final ExtractStage stage;
  final int doneBytes;
  final int totalBytes;

  /// While copying: the file being written.
  final String? fileName;

  double? get fraction => totalBytes > 0 ? (doneBytes / totalBytes).clamp(0.0, 1.0) : null;
}

/// Reports how far [input] has been read each time the output buffer is
/// written to disk (about every MB), without touching the per-byte path.
class _ProgressOutput extends OutputFileStream {
  _ProgressOutput(String path, this.onFlush) : super.withFileHandle(FileHandle(path, mode: FileAccess.write));

  final void Function() onFlush;

  @override
  void flush() {
    super.flush();
    onFlush();
  }
}

List<String> _extractSync(String archivePath, String destPath, Set<String> wanted, SendPort progress) {
  final dest = Directory(destPath)..createSync(recursive: true);
  final tarFile = File(p.join(destPath, '.extract.tar'));
  final written = <String>[];
  var lastSent = DateTime.fromMillisecondsSinceEpoch(0);
  void report(ExtractStage stage, int done, int total, [String? file, bool force = false]) {
    final now = DateTime.now();
    if (!force && now.difference(lastSent).inMilliseconds < 250) return;
    lastSent = now;
    progress.send((stage.index, done, total, file));
  }

  try {
    final archiveBytes = File(archivePath).lengthSync();
    final input = InputFileStream(archivePath);
    final output = _ProgressOutput(tarFile.path, () => report(ExtractStage.decompressing, input.position, archiveBytes));
    report(ExtractStage.decompressing, 0, archiveBytes, null, true);
    try {
      BZip2Decoder().decodeStream(input, output);
    } finally {
      input.closeSync();
      output.closeSync();
    }
    report(ExtractStage.decompressing, archiveBytes, archiveBytes, null, true);

    final tarInput = InputFileStream(tarFile.path);
    try {
      final tar = TarDecoder().decodeStream(tarInput);
      bool isWanted(ArchiveFile e) => e.isFile && wanted.contains(p.basename(e.name));
      final tarBytes = tar.where(isWanted).fold<int>(0, (a, e) => a + e.size);
      var copied = 0;
      for (final entry in tar) {
        if (!isWanted(entry)) continue;
        final name = p.basename(entry.name);
        final target = p.join(dest.path, name);
        final base = copied;
        report(ExtractStage.copying, base, tarBytes, name, true);
        late final _ProgressOutput out;
        out = _ProgressOutput(target, () => report(ExtractStage.copying, base + out.length, tarBytes, name));
        try {
          entry.writeContent(out);
        } finally {
          out.closeSync();
        }
        copied += entry.size;
        written.add(target);
      }
      report(ExtractStage.copying, tarBytes, tarBytes, null, true);
    } finally {
      tarInput.closeSync();
    }
  } finally {
    if (tarFile.existsSync()) tarFile.deleteSync();
  }

  final missing = wanted.difference(written.map(p.basename).toSet());
  if (missing.isNotEmpty) {
    throw StateError('Archive is missing expected files: ${missing.join(', ')}');
  }
  return written;
}
