/// Turns a [Script] into the exact sequence the listener hears: sentences,
/// gaps between them, and music / sound cues, according to a reading style.
library;

import '../script/script.dart';
import '../script/sentences.dart';
import '../script/text_cleaner.dart';
import 'styles.dart';

/// One sentence (or heading) the voice says; [start]/[end] index into the
/// block's on-screen text for highlighting.
class Utterance {
  const Utterance({
    required this.index,
    required this.block,
    required this.start,
    required this.end,
    required this.speech,
    this.speaker,
    this.heading = false,
  });

  final int index;
  final int block;
  final int start;
  final int end;
  final String speech;
  final String? speaker;
  final bool heading;

  int get words => RegExp(r'\S+').allMatches(speech).length;
}

sealed class Step {
  const Step();
}

class SpeakStep extends Step {
  const SpeakStep(this.utterance);
  final int utterance;
  @override
  String toString() => 'Say($utterance)';
}

class GapStep extends Step {
  const GapStep(this.ms);
  final int ms;
  @override
  String toString() => 'Gap($ms)';
}

/// `music:intro`, `music:outro`, `music:stop`, `sound:<name>`, or `section`.
class CueStep extends Step {
  const CueStep(this.cue);
  final String cue;
  @override
  String toString() => 'Cue($cue)';
}

class Plan {
  Plan(this.utterances, this.steps) : stepOfUtterance = _index(utterances.length, steps);

  static List<int> _index(int count, List<Step> steps) {
    final at = List<int>.filled(count, 0);
    for (var i = 0; i < steps.length; i++) {
      final s = steps[i];
      if (s is SpeakStep) at[s.utterance] = i;
    }
    return at;
  }

  final List<Utterance> utterances;
  final List<Step> steps;

  /// Index into [steps] where each utterance is spoken.
  final List<int> stepOfUtterance;

  /// Utterances of each block, for the reader screen.
  Map<int, List<Utterance>> get byBlock {
    final m = <int, List<Utterance>>{};
    for (final u in utterances) {
      (m[u.block] ??= []).add(u);
    }
    return m;
  }

  int get words => utterances.fold(0, (a, u) => a + u.words);

  /// Rough listening time at [speed] (155 words/minute at 1.0x, plus gaps).
  Duration estimate(double speed) {
    var ms = words / (155 * speed) * 60000;
    for (final s in steps) {
      if (s is GapStep) ms += s.ms;
    }
    return Duration(milliseconds: ms.round());
  }
}

Plan buildPlan(Script script, ReadingStyle style, {bool music = true}) {
  final utterances = <Utterance>[];
  final steps = <Step>[];
  final blocks = script.blocks;
  final hasIntroCue = blocks.any((b) => b.cue == 'music:intro');
  final hasOutroCue = blocks.any((b) => b.cue == 'music:outro');
  final autoMusic = music && (script.music ?? style.music) != 'off';

  if (autoMusic && !hasIntroCue) steps.add(const CueStep('music:intro'));

  String? lastSpeaker;
  var spokenSomething = false;
  // A [pause] written in the script is never shortened by automatic gaps.
  var pinned = false;

  for (var bi = 0; bi < blocks.length; bi++) {
    final b = blocks[bi];
    switch (b.kind) {
      case BlockKind.heading:
        if (style.readHeadings) {
          if (spokenSomething) _gap(steps, style.headingBeforeMs);
          steps.add(const CueStep('section'));
          final speech = speechText(b.text);
          if (isSpeakable(speech)) {
            final u = Utterance(
              index: utterances.length,
              block: bi,
              start: 0,
              end: b.text.length,
              speech: _endWithStop(speech),
              heading: true,
            );
            utterances.add(u);
            steps.add(SpeakStep(u.index));
            _gap(steps, style.headingAfterMs);
            spokenSomething = true;
          }
        } else {
          steps.add(const CueStep('section'));
          if (spokenSomething && music && style.sectionSound != null) {
            _gap(steps, 250);
            steps.add(CueStep('sound:${style.sectionSound}'));
            _gap(steps, 300);
          } else if (spokenSomething) {
            _gap(steps, style.headingBeforeMs);
          }
        }
        lastSpeaker = null;
      case BlockKind.line:
        if (spokenSomething && !pinned && lastSpeaker != null && (b.speaker ?? '') != lastSpeaker) {
          // A change of voice: replace the paragraph gap with the speaker gap.
          _replaceTrailingGap(steps, style.speakerGapMs);
        }
        final ranges = splitSentences(b.text);
        for (var si = 0; si < ranges.length; si++) {
          final r = ranges[si];
          final speech = speechText(b.text.substring(r.start, r.end));
          if (!isSpeakable(speech)) continue;
          final u = Utterance(
            index: utterances.length,
            block: bi,
            start: r.start,
            end: r.end,
            speech: speech,
            speaker: b.speaker,
          );
          utterances.add(u);
          steps.add(SpeakStep(u.index));
          _gap(steps, si == ranges.length - 1 ? style.paragraphGapMs : style.sentenceGapMs);
          spokenSomething = true;
          pinned = false;
        }
        lastSpeaker = b.speaker ?? '';
      case BlockKind.pause:
        _replaceTrailingGap(steps, (b.seconds * 1000).round(), keepLonger: true);
        pinned = true;
      case BlockKind.cue:
        // Explicit cues play whenever music & sounds are on, even in a style
        // without default music (the renderer then uses Curious Minds).
        if (music) steps.add(CueStep(b.cue!));
    }
  }

  // Trailing gap is pointless; the outro (or the end) follows.
  while (steps.isNotEmpty && steps.last is GapStep) {
    steps.removeLast();
  }

  if (autoMusic && !hasOutroCue && utterances.length > 3) {
    // Bring the music back in under the last sentence.
    final lastSay = steps.lastIndexWhere((s) => s is SpeakStep);
    if (lastSay > 0) steps.insert(lastSay, const CueStep('music:outro'));
  }

  return Plan(utterances, steps);
}

/// Adds a gap; back-to-back gaps merge into the longer one.
void _gap(List<Step> steps, int ms) {
  if (ms <= 0) return;
  if (steps.isNotEmpty && steps.last is GapStep) {
    final old = (steps.last as GapStep).ms;
    if (ms > old) steps[steps.length - 1] = GapStep(ms);
    return;
  }
  steps.add(GapStep(ms));
}

/// Sets the gap at the end of [steps] to [ms] (adds one if there is none).
void _replaceTrailingGap(List<Step> steps, int ms, {bool keepLonger = false}) {
  if (steps.isNotEmpty && steps.last is GapStep) {
    final old = (steps.last as GapStep).ms;
    steps[steps.length - 1] = GapStep(keepLonger ? (ms > old ? ms : old) : ms);
  } else {
    _gap(steps, ms);
  }
}

String _endWithStop(String s) => RegExp(r'[.!?]$').hasMatch(s) ? s : '$s.';
