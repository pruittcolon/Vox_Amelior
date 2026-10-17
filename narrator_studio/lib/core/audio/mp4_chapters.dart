/// Chapter marks inside an .m4a, so podcast players show the episode's
/// chapters: a QuickTime chapter track (what Apple Podcasts, Overcast and
/// Pocket Casts read) and a Nero `chpl` list (VLC, foobar2000, ffmpeg). The
/// audio itself is never touched; only the `moov` index grows.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

class ChapterMark {
  const ChapterMark(this.start, this.title);
  final Duration start;
  final String title;

  @override
  String toString() => '${start.inMilliseconds}ms $title';
}

/// Adds [chapters] to the MP4 at [path], in place (through a temporary file).
/// Returns false and leaves the file as it was when it is not an MP4 this
/// can handle, already has chapters, or there are fewer than two chapters.
Future<bool> writeMp4Chapters(
  String path,
  List<ChapterMark> chapters, {
  bool quicktime = true,
  bool nero = true,
}) async {
  final marks = _tidy(chapters);
  if (marks.length < 2 || (!quicktime && !nero)) return false;
  final file = File(path);
  final tmp = File('$path.chapters.tmp');
  final raf = await file.open();
  try {
    final length = await raf.length();
    final top = await _topLevel(raf, length);
    final moovs = top.where((b) => b.type == 'moov').toList();
    if (moovs.length != 1 || !top.any((b) => b.type == 'mdat')) return false;
    final oldMoov = moovs.single;
    await raf.setPosition(oldMoov.start);
    final moov = await raf.read(oldMoov.size);
    final built = _addChapters(moov, marks, quicktime: quicktime, nero: nero);
    if (built == null) return false;
    final delta = built.moov.length - oldMoov.size;
    // Media after the old moov moves by delta; the chapter text goes at the end.
    final textAt = length + delta + 8;
    if (!_patchOffsets(built.moov, oldMoov.start, delta, built.trackId, textAt)) return false;

    final out = await tmp.open(mode: FileMode.write);
    try {
      await _copy(raf, out, 0, oldMoov.start);
      await out.writeFrom(built.moov);
      await _copy(raf, out, oldMoov.end, length);
      if (built.text.isNotEmpty) {
        await out.writeFrom(_u32(8 + built.text.length));
        await out.writeFrom(ascii.encode('mdat'));
        await out.writeFrom(built.text);
      }
    } finally {
      await out.close();
    }
  } on FormatException {
    if (tmp.existsSync()) await tmp.delete();
    return false;
  } finally {
    await raf.close();
  }
  await tmp.rename(path);
  return true;
}

/// Chapters found in the MP4 at [path]: (QuickTime track, Nero list).
Future<(List<ChapterMark>, List<ChapterMark>)> readMp4Chapters(String path) async {
  final raf = await File(path).open();
  try {
    final length = await raf.length();
    final top = await _topLevel(raf, length);
    final m = top.firstWhere((b) => b.type == 'moov', orElse: () => throw const FormatException('no moov'));
    await raf.setPosition(m.start);
    final moov = await raf.read(m.size);
    final d = ByteData.sublistView(moov);
    final kids = _children(moov, 8, moov.length);

    final nero = <ChapterMark>[];
    final udta = kids.where((b) => b.type == 'udta').firstOrNull;
    final chpl = udta == null ? null : _children(moov, udta.body, udta.end).where((b) => b.type == 'chpl').firstOrNull;
    if (chpl != null) {
      var at = chpl.body + 4 + (moov[chpl.body] > 0 ? 4 : 0);
      final count = moov[at++];
      for (var i = 0; i < count; i++) {
        final start = d.getUint64(at);
        final len = moov[at + 8];
        nero.add(ChapterMark(Duration(microseconds: start ~/ 10), utf8.decode(moov.sublist(at + 9, at + 9 + len))));
        at += 9 + len;
      }
    }

    final quicktime = <ChapterMark>[];
    final traks = kids.where((b) => b.type == 'trak').toList();
    final chapterIds = <int>{};
    for (final t in traks) {
      final tref = _child(moov, t, 'tref');
      final chap = tref == null ? null : _child(moov, tref, 'chap');
      if (chap == null) continue;
      for (var at = chap.body; at + 4 <= chap.end; at += 4) {
        chapterIds.add(d.getUint32(at));
      }
    }
    for (final t in traks) {
      if (!chapterIds.contains(_trackId(moov, t))) continue;
      final mdia = _child(moov, t, 'mdia')!;
      final mdhd = _child(moov, mdia, 'mdhd')!;
      final scale = d.getUint32(mdhd.body + (moov[mdhd.body] == 1 ? 20 : 12));
      final stbl = _child(moov, _child(moov, mdia, 'minf')!, 'stbl')!;
      final stts = _child(moov, stbl, 'stts')!;
      final stsz = _child(moov, stbl, 'stsz')!;
      final stco = _child(moov, stbl, 'stco');
      final co64 = _child(moov, stbl, 'co64');
      final durations = <int>[];
      for (var i = 0, n = d.getUint32(stts.body + 4); i < n; i++) {
        final count = d.getUint32(stts.body + 8 + i * 8);
        final delta = d.getUint32(stts.body + 12 + i * 8);
        durations.addAll(List.filled(count, delta));
      }
      final sizes = [for (var i = 0, n = d.getUint32(stsz.body + 8); i < n; i++) d.getUint32(stsz.body + 12 + i * 4)];
      // One chunk holding every sample (as written here and by ffmpeg).
      var offset = stco != null ? d.getUint32(stco.body + 8) : d.getUint64(co64!.body + 8);
      var time = 0;
      for (var i = 0; i < sizes.length; i++) {
        await raf.setPosition(offset);
        final sample = await raf.read(sizes[i]);
        final len = ByteData.sublistView(sample).getUint16(0);
        quicktime.add(ChapterMark(Duration(microseconds: time * 1000000 ~/ scale), utf8.decode(sample.sublist(2, 2 + len))));
        offset += sizes[i];
        if (i < durations.length) time += durations[i];
      }
    }
    return (quicktime, nero);
  } finally {
    await raf.close();
  }
}

/// Starts in order, the first at zero, titles cleaned (Nero allows 255 bytes).
List<ChapterMark> _tidy(List<ChapterMark> chapters) {
  final out = <ChapterMark>[];
  for (final c in chapters) {
    final start = out.isEmpty ? Duration.zero : c.start;
    if (out.isNotEmpty && start <= out.last.start) continue;
    var title = c.title.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (title.isEmpty) title = 'Chapter ${out.length + 1}';
    while (utf8.encode(title).length > 255) {
      title = title.substring(0, title.length - 1);
    }
    out.add(ChapterMark(start, title));
  }
  return out;
}

class _Box {
  const _Box(this.type, this.start, this.size, this.header);
  final String type;
  final int start;
  final int size;
  final int header;
  int get end => start + size;
  int get body => start + header;
}

Future<List<_Box>> _topLevel(RandomAccessFile raf, int length) async {
  final out = <_Box>[];
  var at = 0;
  while (at + 8 <= length) {
    await raf.setPosition(at);
    final head = await raf.read(16);
    if (head.length < 8) break;
    final d = ByteData.sublistView(head);
    var size = d.getUint32(0);
    var header = 8;
    if (size == 1) {
      if (head.length < 16) throw const FormatException('short box');
      size = d.getUint64(8);
      header = 16;
    } else if (size == 0) {
      size = length - at;
    }
    if (size < header || at + size > length) throw const FormatException('bad box');
    out.add(_Box(String.fromCharCodes(head.sublist(4, 8)), at, size, header));
    at += size;
  }
  if (out.isEmpty || out.first.type != 'ftyp') throw const FormatException('not an MP4');
  return out;
}

List<_Box> _children(Uint8List b, int from, int to) {
  final d = ByteData.sublistView(b);
  final out = <_Box>[];
  var at = from;
  while (at + 8 <= to) {
    var size = d.getUint32(at);
    var header = 8;
    if (size == 1) {
      size = d.getUint64(at + 8);
      header = 16;
    } else if (size == 0) {
      size = to - at;
    }
    if (size < header || at + size > to) throw const FormatException('bad box');
    out.add(_Box(String.fromCharCodes(b.sublist(at + 4, at + 8)), at, size, header));
    at += size;
  }
  return out;
}

_Box? _child(Uint8List b, _Box parent, String type) =>
    _children(b, parent.body, parent.end).where((c) => c.type == type).firstOrNull;

int _trackId(Uint8List b, _Box trak) {
  final tkhd = _child(b, trak, 'tkhd');
  if (tkhd == null) return 0;
  return ByteData.sublistView(b).getUint32(tkhd.body + (b[tkhd.body] == 1 ? 20 : 12));
}

String? _handler(Uint8List b, _Box trak) {
  final mdia = _child(b, trak, 'mdia');
  final hdlr = mdia == null ? null : _child(b, mdia, 'hdlr');
  return hdlr == null ? null : String.fromCharCodes(b.sublist(hdlr.body + 8, hdlr.body + 12));
}

class _Built {
  const _Built(this.moov, this.text, this.trackId);
  final Uint8List moov;
  final Uint8List text;
  final int trackId;
}

_Built? _addChapters(Uint8List moov, List<ChapterMark> marks, {required bool quicktime, required bool nero}) {
  final d = ByteData.sublistView(moov);
  final kids = _children(moov, 8, moov.length);
  final mvhd = kids.where((b) => b.type == 'mvhd').firstOrNull;
  final traks = kids.where((b) => b.type == 'trak').toList();
  if (mvhd == null || traks.isEmpty) return null;
  final udta = kids.where((b) => b.type == 'udta').firstOrNull;
  if (udta != null && _children(moov, udta.body, udta.end).any((b) => b.type == 'chpl')) return null;
  if (traks.any((t) => _child(moov, t, 'tref') != null)) return null;
  final audio = traks.where((t) => _handler(moov, t) == 'soun').toList();
  if (audio.isEmpty) return null;

  final v1 = moov[mvhd.body] == 1;
  final scale = d.getUint32(mvhd.body + (v1 ? 20 : 12));
  final duration = v1 ? d.getUint64(mvhd.body + 24) : d.getUint32(mvhd.body + 16);
  final nextIdAt = mvhd.body + (v1 ? 108 : 96);
  if (scale == 0 || nextIdAt + 4 > mvhd.end) return null;
  final totalMs = duration * 1000 ~/ scale;
  var trackId = d.getUint32(nextIdAt);
  final used = traks.map((t) => _trackId(moov, t)).toSet();
  if (trackId == 0 || trackId == 0xFFFFFFFF || used.contains(trackId)) trackId = used.reduce(math.max) + 1;

  final ms = [for (final m in marks) m.start.inMilliseconds];
  if (totalMs <= ms.last) return null; // chapters past the end of the audio

  final out = BytesBuilder(copy: false);
  final text = BytesBuilder(copy: false);
  for (final k in kids) {
    final bytes = moov.sublist(k.start, k.end);
    switch (k.type) {
      case 'mvhd':
        if (quicktime) ByteData.sublistView(bytes).setUint32(nextIdAt - k.start, trackId + 1);
        out.add(bytes);
      case 'trak' when quicktime && audio.contains(k):
        // tref/chap right after tkhd points the audio at the chapter track.
        final parts = _children(moov, k.body, k.end);
        final body = BytesBuilder(copy: false);
        for (final p in parts) {
          body.add(moov.sublist(p.start, p.end));
          if (p.type == 'tkhd') body.add(_box('tref', [_box('chap', [_u32(trackId)])]));
        }
        out.add(_box('trak', [body.takeBytes()]));
      case 'udta' when nero:
        out.add(_box('udta', [moov.sublist(k.body, k.end), _chpl(marks)]));
      default:
        out.add(bytes);
    }
  }
  if (nero && udta == null) out.add(_box('udta', [_chpl(marks)]));
  if (quicktime) {
    final sizes = <int>[];
    for (final m in marks) {
      final t = utf8.encode(m.title);
      final sample = [..._u16(t.length), ...t, ..._encd];
      sizes.add(sample.length);
      text.add(sample);
    }
    final durations = [for (var i = 0; i < ms.length; i++) (i + 1 < ms.length ? ms[i + 1] : totalMs) - ms[i]];
    out.add(_chapterTrak(trackId, durations, sizes, movieScale: scale));
  }
  return _Built(_box('moov', [out.takeBytes()]), text.takeBytes(), quicktime ? trackId : -1);
}

/// Moves every chunk offset that points past the old moov by [delta]; the
/// chapter track's single chunk goes to [textAt].
bool _patchOffsets(Uint8List moov, int moovStart, int delta, int chapterTrack, int textAt) {
  final d = ByteData.sublistView(moov);
  for (final trak in _children(moov, 8, moov.length).where((b) => b.type == 'trak')) {
    final mdia = _child(moov, trak, 'mdia');
    final minf = mdia == null ? null : _child(moov, mdia, 'minf');
    final stbl = minf == null ? null : _child(moov, minf, 'stbl');
    if (stbl == null) continue;
    final mine = _trackId(moov, trak) == chapterTrack;
    for (final box in _children(moov, stbl.body, stbl.end)) {
      if (box.type != 'stco' && box.type != 'co64') continue;
      final wide = box.type == 'co64';
      final n = d.getUint32(box.body + 4);
      for (var i = 0; i < n; i++) {
        final at = box.body + 8 + i * (wide ? 8 : 4);
        final old = wide ? d.getUint64(at) : d.getUint32(at);
        final value = mine ? textAt : (old > moovStart ? old + delta : old);
        if (wide) {
          d.setUint64(at, value);
        } else {
          if (value > 0xFFFFFFFF) return false;
          d.setUint32(at, value);
        }
      }
    }
  }
  return true;
}

Uint8List _chapterTrak(int id, List<int> durationsMs, List<int> sizes, {required int movieScale}) {
  final totalMs = durationsMs.fold(0, (a, b) => a + b);
  final n = durationsMs.length;
  const identity = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
  final tkhd = _full('tkhd', 0, 0x000002, [
    // In the movie but not enabled: players use it for chapters, never show it.
    _u32(0), _u32(0), _u32(id), _u32(0), _u32(totalMs * movieScale ~/ 1000),
    _u32(0), _u32(0), _u16(0), _u16(0), _u16(0), _u16(0),
    for (final v in identity) _u32(v),
    _u32(0), _u32(0),
  ]);
  final mdhd = _full('mdhd', 0, 0, [_u32(0), _u32(0), _u32(1000), _u32(totalMs), _u16(0x55C4), _u16(0)]);
  final hdlr = _full('hdlr', 0, 0, [_u32(0), ascii.encode('text'), _u32(0), _u32(0), _u32(0), ascii.encode('Chapters\x00')]);
  // The generic media header Apple's chapter tracks carry (as ffmpeg writes it).
  final gmhd = _box('gmhd', [
    _box('gmin', [_u32(0), _u16(0x40), _u16(0x8000), _u16(0x8000), _u16(0x8000), _u16(0), _u16(0)]),
    _box('text', [_u16(1), _u32(0), _u32(0), _u32(0), _u32(1), _u32(0), _u32(0), _u32(0), _u32(0x4000), _u16(0)]),
  ]);
  final dinf = _box('dinf', [
    _full('dref', 0, 0, [_u32(1), _full('url ', 0, 1, const [])]),
  ]);
  final textEntry = _box('text', [Uint8List(6), _u16(1), _textSampleEntry]);
  final stbl = _box('stbl', [
    _full('stsd', 0, 0, [_u32(1), textEntry]),
    _full('stts', 0, 0, [_u32(n), for (final ms in durationsMs) ...[_u32(1), _u32(ms)]]),
    _full('stsz', 0, 0, [_u32(0), _u32(n), for (final s in sizes) _u32(s)]),
    _full('stsc', 0, 0, [_u32(1), _u32(1), _u32(n), _u32(1)]),
    _full('stco', 0, 0, [_u32(1), _u32(0)]), // set by _patchOffsets
  ]);
  return _box('trak', [
    tkhd,
    _box('mdia', [mdhd, hdlr, _box('minf', [gmhd, dinf, stbl])]),
  ]);
}

Uint8List _chpl(List<ChapterMark> marks) {
  final b = BytesBuilder(copy: false)
    ..add(_u32(0))
    ..addByte(math.min(marks.length, 255));
  for (final m in marks.take(255)) {
    final t = utf8.encode(m.title);
    b
      ..add(_u64(m.start.inMicroseconds * 10)) // 100 ns units
      ..addByte(t.length)
      ..add(t);
  }
  return _full('chpl', 1, 0, [b.takeBytes()]);
}

/// The text sample description ffmpeg gives chapter tracks.
final _textSampleEntry = Uint8List.fromList([
  0, 0, 0, 1, // display flags
  0, 0, // justification
  0, 0, 0, 0, // background colour
  0, 0, 0, 0, 0, 0, 0, 0, // default text box
  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // style record
  0, 0, 0, 13, 0x66, 0x74, 0x61, 0x62, 0, 1, 0, 1, 0, // font table: one font, no name
]);

/// Marks each chapter title as UTF-8.
const _encd = [0, 0, 0, 12, 0x65, 0x6E, 0x63, 0x64, 0, 0, 1, 0];

Uint8List _box(String type, List<List<int>> parts) {
  final size = 8 + parts.fold(0, (a, p) => a + p.length);
  final b = BytesBuilder(copy: false)
    ..add(_u32(size))
    ..add(ascii.encode(type));
  for (final p in parts) {
    b.add(p);
  }
  return b.takeBytes();
}

Uint8List _full(String type, int version, int flags, List<List<int>> parts) =>
    _box(type, [_u32((version << 24) | flags), ...parts]);

Uint8List _u16(int v) => Uint8List(2)..buffer.asByteData().setUint16(0, v);
Uint8List _u32(int v) => Uint8List(4)..buffer.asByteData().setUint32(0, v);
Uint8List _u64(int v) => Uint8List(8)..buffer.asByteData().setUint64(0, v);

Future<void> _copy(RandomAccessFile from, RandomAccessFile to, int start, int end) async {
  await from.setPosition(start);
  var left = end - start;
  while (left > 0) {
    final chunk = await from.read(math.min(left, 1 << 20));
    if (chunk.isEmpty) throw const FormatException('file ended early');
    await to.writeFrom(chunk);
    left -= chunk.length;
  }
}
