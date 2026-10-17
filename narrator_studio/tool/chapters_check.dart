// Writes chapters into an M4A with the app's code and reads them back.
//
//   dart run tool/chapters_check.dart --m4a FILE --chapters FILE.json [--only quicktime|nero]
//
// CI then asks ffprobe, an independent reader, to list them too.
import 'dart:convert';
import 'dart:io';

import 'package:narrator_studio/core/audio/mp4_chapters.dart';

Future<void> main(List<String> args) async {
  String? arg(String name) {
    final i = args.indexOf('--$name');
    return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
  }

  final m4a = arg('m4a')!;
  final only = arg('only');
  final list = (jsonDecode(await File(arg('chapters')!).readAsString()) as List).cast<Map<String, Object?>>();
  final marks = [
    for (final c in list) ChapterMark(Duration(milliseconds: (c['ms']! as num).toInt()), c['title']! as String),
  ];
  final before = await File(m4a).length();
  final ok = await writeMp4Chapters(m4a, marks, quicktime: only != 'nero', nero: only != 'quicktime');
  if (!ok) {
    stderr.writeln('chapters were not written to $m4a');
    exit(1);
  }
  final (qt, nero) = await readMp4Chapters(m4a);
  stdout.writeln('$m4a: $before -> ${await File(m4a).length()} bytes');
  stdout.writeln('QuickTime: ${qt.join(' | ')}');
  stdout.writeln('Nero:      ${nero.join(' | ')}');
  final want = only == 'nero' ? 0 : marks.length;
  if (qt.length != want || nero.length != (only == 'quicktime' ? 0 : marks.length)) {
    stderr.writeln('read back the wrong number of chapters');
    exit(1);
  }
}
