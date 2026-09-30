import 'dart:math' as math;
import 'dart:typed_data';

/// Who is talking in each short frame of a stretch of audio, as produced by
/// NVIDIA's Sortformer-family diarizer: a 0–1 activity per speaker slot
/// ("slot 0" = the first voice heard in this audio, "slot 1" the second...).
class SpeakerActivity {
  SpeakerActivity(this.probs, {required this.frames, required this.speakers, required this.frameSeconds})
      : assert(probs.length >= frames * speakers);

  final Float32List probs;
  final int frames;
  final int speakers;
  final double frameSeconds;

  double at(int frame, int speaker) => probs[frame * speakers + speaker];
}

/// One person's stretch of talk inside the audio.
class SpeakerTurn {
  const SpeakerTurn({required this.start, required this.end, required this.slot, this.overlap = false});

  /// Seconds from the start of the audio.
  final double start;
  final double end;

  /// Diarizer slot of the main speaker.
  final int slot;

  /// Someone else talked at the same time during part of this turn.
  final bool overlap;

  double get duration => end - start;

  SpeakerTurn copyWith({double? start, double? end, int? slot, bool? overlap}) => SpeakerTurn(
        start: start ?? this.start,
        end: end ?? this.end,
        slot: slot ?? this.slot,
        overlap: overlap ?? this.overlap,
      );

  @override
  String toString() => 'Turn(${start.toStringAsFixed(2)}–${end.toStringAsFixed(2)} slot $slot${overlap ? ' +overlap' : ''})';
}

/// Turns frame-level speaker activity into clean speaker turns.
class SpeakerTurns {
  const SpeakerTurns({
    this.threshold = 0.5,
    this.minBurstSeconds = 0.24,
    this.minGapSeconds = 0.24,
    this.minTurnSeconds = 0.8,
    this.minOverlapSeconds = 0.24,
  });

  /// Activity above this counts as talking.
  final double threshold;

  /// Shorter blips of activity are ignored.
  final double minBurstSeconds;

  /// Shorter pauses inside one person's talk are filled in.
  final double minGapSeconds;

  /// Shorter turns are merged into a neighbour (too short to transcribe well).
  final double minTurnSeconds;

  /// Two voices must overlap at least this long to be marked.
  final double minOverlapSeconds;

  /// Turns in time order. Covers the whole audio: pauses belong to the turn
  /// around them, so no speech falls between turns.
  List<SpeakerTurn> turns(SpeakerActivity a, {required double totalSeconds}) {
    if (a.frames == 0 || a.speakers == 0) return [SpeakerTurn(start: 0, end: totalSeconds, slot: 0)];
    final fs = a.frameSeconds;
    final burst = math.max(1, (minBurstSeconds / fs).round());
    final gap = math.max(1, (minGapSeconds / fs).round());

    // Per speaker: threshold, fill short gaps, drop short bursts.
    final active = List.generate(a.speakers, (s) {
      final row = List<bool>.generate(a.frames, (t) => a.at(t, s) > threshold);
      _fillGaps(row, gap);
      _dropBursts(row, burst);
      return row;
    });

    // Frame labels: -1 silence, slot index for one voice, -2 for overlap.
    final label = List<int>.filled(a.frames, -1);
    final loudest = List<int>.filled(a.frames, -1);
    for (var t = 0; t < a.frames; t++) {
      var count = 0;
      var best = -1;
      var bestP = -1.0;
      for (var s = 0; s < a.speakers; s++) {
        if (!active[s][t]) continue;
        count++;
        if (a.at(t, s) > bestP) {
          bestP = a.at(t, s);
          best = s;
        }
      }
      label[t] = count == 0 ? -1 : (count == 1 ? best : -2);
      loudest[t] = best;
    }
    final overlapFrames = math.max(1, (minOverlapSeconds / fs).round());
    // Short overlaps are just a hand-over: treat as the louder voice.
    var t = 0;
    while (t < a.frames) {
      if (label[t] != -2) {
        t++;
        continue;
      }
      var e = t;
      while (e < a.frames && label[e] == -2) {
        e++;
      }
      if (e - t < overlapFrames) {
        for (var i = t; i < e; i++) {
          label[i] = loudest[i];
        }
      }
      t = e;
    }

    // Runs of single-voice frames become turns; overlap and silence join the
    // current turn.
    final out = <SpeakerTurn>[];
    int? slot;
    var start = 0;
    var overlap = false;
    for (var i = 0; i < a.frames; i++) {
      final l = label[i];
      if (l >= 0 && l != slot) {
        if (slot != null) {
          out.add(SpeakerTurn(start: start * fs, end: i * fs, slot: slot, overlap: overlap));
          overlap = false;
        }
        slot = l;
        start = out.isEmpty ? 0 : i;
      } else if (l == -2) {
        overlap = true;
      }
    }
    if (slot == null) return [SpeakerTurn(start: 0, end: totalSeconds, slot: 0, overlap: overlap)];
    out.add(SpeakerTurn(start: start * fs, end: totalSeconds, slot: slot, overlap: overlap));
    return _mergeShort(out);
  }

  /// Merges turns shorter than [minTurnSeconds] into a neighbour, then joins
  /// neighbours that ended up with the same speaker.
  List<SpeakerTurn> _mergeShort(List<SpeakerTurn> turns) {
    final list = List<SpeakerTurn>.of(turns);
    var changed = true;
    while (changed && list.length > 1) {
      changed = false;
      for (var i = 0; i < list.length; i++) {
        if (list[i].duration >= minTurnSeconds) continue;
        // Into the longer neighbour.
        final left = i > 0 ? list[i - 1] : null;
        final right = i + 1 < list.length ? list[i + 1] : null;
        final intoLeft = right == null || (left != null && left.duration >= right.duration);
        if (intoLeft) {
          list[i - 1] = left!.copyWith(end: list[i].end, overlap: left.overlap || list[i].overlap);
        } else {
          list[i + 1] = right.copyWith(start: list[i].start, overlap: right.overlap || list[i].overlap);
        }
        list.removeAt(i);
        changed = true;
        break;
      }
    }
    return joinSame(list, (t) => t.slot);
  }

  /// Joins neighbouring turns that belong to the same person according to
  /// [key] (e.g. after naming two diarizer slots as the same person).
  static List<SpeakerTurn> joinSame<K>(List<SpeakerTurn> turns, K Function(SpeakerTurn) key) {
    final out = <SpeakerTurn>[];
    for (final t in turns) {
      if (out.isNotEmpty && key(out.last) == key(t)) {
        out[out.length - 1] = out.last.copyWith(end: t.end, overlap: out.last.overlap || t.overlap);
      } else {
        out.add(t);
      }
    }
    return out;
  }

  /// Sample ranges where only [slot] is talking (clean audio for a voiceprint).
  static List<(int, int)> soloRanges(SpeakerActivity a, int slot, {required int sampleRate, double threshold = 0.5}) {
    final ranges = <(int, int)>[];
    int? from;
    for (var t = 0; t <= a.frames; t++) {
      var solo = false;
      if (t < a.frames && a.at(t, slot) > threshold) {
        solo = true;
        for (var s = 0; s < a.speakers; s++) {
          if (s != slot && a.at(t, s) > threshold) solo = false;
        }
      }
      if (solo && from == null) from = t;
      if (!solo && from != null) {
        ranges.add(((from * a.frameSeconds * sampleRate).round(), (t * a.frameSeconds * sampleRate).round()));
        from = null;
      }
    }
    return ranges;
  }

  static void _fillGaps(List<bool> row, int maxGap) {
    var i = 0;
    while (i < row.length) {
      if (row[i]) {
        i++;
        continue;
      }
      var e = i;
      while (e < row.length && !row[e]) {
        e++;
      }
      if (i > 0 && e < row.length && e - i <= maxGap) {
        for (var k = i; k < e; k++) {
          row[k] = true;
        }
      }
      i = e;
    }
  }

  static void _dropBursts(List<bool> row, int minBurst) {
    var i = 0;
    while (i < row.length) {
      if (!row[i]) {
        i++;
        continue;
      }
      var e = i;
      while (e < row.length && row[e]) {
        e++;
      }
      if (e - i < minBurst) {
        for (var k = i; k < e; k++) {
          row[k] = false;
        }
      }
      i = e;
    }
  }
}
