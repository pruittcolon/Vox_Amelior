import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';

/// Whose speech is saved as audio clips.
enum ClipMode { off, everyone, chosen }

/// What to save and how much room it may take.
class ClipPolicy {
  const ClipPolicy({this.mode = ClipMode.off, this.people = const [], this.limitBytes = 2 * 1024 * 1024 * 1024});

  final ClipMode mode;

  /// Person ids for [ClipMode.chosen].
  final List<String> people;
  final int limitBytes;

  bool allows(String? speakerId) => switch (mode) {
        ClipMode.off => false,
        ClipMode.everyone => true,
        ClipMode.chosen => speakerId != null && people.contains(speakerId),
      };
}

/// Clips saved for one person (or unlabelled ones, [speakerId] null).
class ClipGroup {
  const ClipGroup({required this.speakerId, required this.name, required this.count, required this.bytes, required this.duration});

  final String? speakerId;
  final String name;
  final int count;
  final int bytes;
  final Duration duration;
}

class ClipStats {
  const ClipStats({required this.count, required this.bytes, required this.duration, required this.groups});

  final int count;
  final int bytes;
  final Duration duration;
  final List<ClipGroup> groups;
}

/// Audio of what was said (16 kHz mono WAV) next to its text, kept on the
/// phone so the voice and speech models can be trained on it later.
class ClipStore {
  ClipStore(this._db, this.dir, {this.clock = systemClock});

  final AppDatabase _db;
  final Directory dir;
  final Clock clock;

  int totalBytes() =>
      (_db.raw.select('SELECT COALESCE(SUM(bytes), 0) AS b FROM voice_clips').first['b']! as int);

  /// Saves the audio of [segment] if [policy] allows it and there is room.
  /// Returns true when a clip was written.
  bool maybeSave(ClipPolicy policy, SegmentView segment, Float32List samples, {int sampleRate = 16000}) {
    if (!policy.allows(segment.speakerId) || samples.isEmpty) return false;
    final bytes = 44 + samples.length * 2;
    if (totalBytes() + bytes > policy.limitBytes) return false;
    final month = segment.startedAt.toIso8601String().substring(0, 7);
    final relative = p.join(month, '${segment.startedAt.millisecondsSinceEpoch}_${segment.id}.wav');
    final file = File(p.join(dir.path, relative));
    file.parent.createSync(recursive: true);
    file.writeAsBytesSync(wavBytes(samples, sampleRate), flush: false);
    _db.raw.execute(
      'INSERT OR REPLACE INTO voice_clips(segment_id, speaker_id, text, path, bytes, duration_ms, started_at, created_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      [
        segment.id,
        segment.speakerId,
        segment.text,
        relative,
        bytes,
        (samples.length * 1000 / sampleRate).round(),
        segment.startedAt.millisecondsSinceEpoch,
        clock().millisecondsSinceEpoch,
      ],
    );
    return true;
  }

  ClipStats stats() {
    final rows = _db.raw.select('''
SELECT c.speaker_id AS sid, sp.name AS name, COUNT(*) AS n, SUM(c.bytes) AS b, SUM(c.duration_ms) AS d
FROM voice_clips c LEFT JOIN speakers sp ON sp.id = c.speaker_id
GROUP BY c.speaker_id ORDER BY n DESC''');
    final groups = [
      for (final r in rows)
        ClipGroup(
          speakerId: r['sid'] as String?,
          name: (r['name'] as String?) ?? 'Not named',
          count: r['n']! as int,
          bytes: r['b']! as int,
          duration: Duration(milliseconds: r['d']! as int),
        ),
    ];
    return ClipStats(
      count: groups.fold(0, (a, g) => a + g.count),
      bytes: groups.fold(0, (a, g) => a + g.bytes),
      duration: groups.fold(Duration.zero, (a, g) => a + g.duration),
      groups: groups,
    );
  }

  /// Deletes every clip. Returns how many were removed.
  int deleteAll() {
    final n = _deleteWhere('1 = 1', const []);
    if (dir.existsSync()) {
      for (final e in dir.listSync()) {
        e.deleteSync(recursive: true);
      }
    }
    return n;
  }

  /// Deletes one person's clips ([speakerId] null = clips without a name).
  int deleteForSpeaker(String? speakerId) =>
      speakerId == null ? _deleteWhere('speaker_id IS NULL', const []) : _deleteWhere('speaker_id = ?', [speakerId]);

  int deleteForSegment(int segmentId) => _deleteWhere('segment_id = ?', [segmentId]);

  void deleteForConversation(int conversationId) => _deleteWhere(
        'segment_id IN (SELECT id FROM segments WHERE conversation_id = ?)',
        [conversationId],
      );

  /// Writes a NeMo-style training manifest (one JSON object per line) next
  /// to the clips and returns it.
  File writeManifest() {
    final out = File(p.join(dir.path, 'manifest.jsonl'));
    dir.createSync(recursive: true);
    final rows = _db.raw.select('''
SELECT c.path, c.duration_ms, c.text, sp.name AS name
FROM voice_clips c LEFT JOIN speakers sp ON sp.id = c.speaker_id ORDER BY c.started_at''');
    final b = StringBuffer();
    for (final r in rows) {
      b.writeln(jsonEncode({
        'audio_filepath': r['path'],
        'duration': (r['duration_ms']! as int) / 1000,
        'text': r['text'],
        'speaker': r['name'],
      }));
    }
    out.writeAsStringSync(b.toString(), flush: true);
    return out;
  }

  int _deleteWhere(String where, List<Object?> args) {
    final rows = _db.raw.select('SELECT id, path FROM voice_clips WHERE $where', args);
    for (final r in rows) {
      final f = File(p.join(dir.path, r['path']! as String));
      if (f.existsSync()) f.deleteSync();
    }
    _db.raw.execute('DELETE FROM voice_clips WHERE $where', args);
    return rows.length;
  }

  /// 16-bit PCM mono WAV.
  static Uint8List wavBytes(Float32List samples, int sampleRate) {
    final data = ByteData(44 + samples.length * 2);
    void ascii(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        data.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    ascii(0, 'RIFF');
    data.setUint32(4, 36 + samples.length * 2, Endian.little);
    ascii(8, 'WAVE');
    ascii(12, 'fmt ');
    data
      ..setUint32(16, 16, Endian.little)
      ..setUint16(20, 1, Endian.little)
      ..setUint16(22, 1, Endian.little)
      ..setUint32(24, sampleRate, Endian.little)
      ..setUint32(28, sampleRate * 2, Endian.little)
      ..setUint16(32, 2, Endian.little)
      ..setUint16(34, 16, Endian.little);
    ascii(36, 'data');
    data.setUint32(40, samples.length * 2, Endian.little);
    for (var i = 0; i < samples.length; i++) {
      data.setInt16(44 + i * 2, (samples[i].clamp(-1.0, 1.0) * 32767).round(), Endian.little);
    }
    return data.buffer.asUint8List();
  }
}
