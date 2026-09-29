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
  Future<List<File>> extractTarBz2(
    File archive,
    Directory destDir, {
    required Set<String> wanted,
  }) {
    final archivePath = archive.path;
    final destPath = destDir.path;
    final wantedCopy = Set<String>.of(wanted);
    return Isolate.run(() => _extractSync(archivePath, destPath, wantedCopy))
        .then((paths) => paths.map(File.new).toList());
  }
}

List<String> _extractSync(String archivePath, String destPath, Set<String> wanted) {
  final dest = Directory(destPath)..createSync(recursive: true);
  final tarFile = File(p.join(destPath, '.extract.tar'));
  final written = <String>[];
  try {
    final input = InputFileStream(archivePath);
    final output = OutputFileStream(tarFile.path);
    try {
      BZip2Decoder().decodeStream(input, output);
    } finally {
      input.closeSync();
      output.closeSync();
    }

    final tarInput = InputFileStream(tarFile.path);
    try {
      final tar = TarDecoder().decodeStream(tarInput);
      for (final entry in tar) {
        if (!entry.isFile) continue;
        final name = p.basename(entry.name);
        if (!wanted.contains(name)) continue;
        final target = p.join(dest.path, name);
        final out = OutputFileStream(target);
        try {
          entry.writeContent(out);
        } finally {
          out.closeSync();
        }
        written.add(target);
      }
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
