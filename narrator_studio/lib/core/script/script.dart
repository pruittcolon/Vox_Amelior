/// A parsed document or script: what to say, who says it, and where the
/// pauses, music and sound cues go.
library;

enum BlockKind { heading, line, pause, cue }

class Block {
  const Block._(this.kind, {this.text = '', this.speaker, this.level = 0, this.seconds = 0, this.cue});

  const Block.heading(String text, {int level = 2}) : this._(BlockKind.heading, text: text, level: level);

  const Block.line(String text, {String? speaker}) : this._(BlockKind.line, text: text, speaker: speaker);

  const Block.pause(double seconds) : this._(BlockKind.pause, seconds: seconds);

  /// [cue] is `music:intro`, `music:outro`, `music:stop` or `sound:<name>`.
  const Block.cue(String cue) : this._(BlockKind.cue, cue: cue);

  final BlockKind kind;

  /// What is shown on screen (markdown removed); sentences are cut from this.
  final String text;

  /// Upper-case speaker tag such as `HOST`, or null for the narrator.
  final String? speaker;
  final int level;
  final double seconds;
  final String? cue;

  bool get isSpoken => kind == BlockKind.line || kind == BlockKind.heading;

  @override
  String toString() => switch (kind) {
        BlockKind.heading => 'H$level($text)',
        BlockKind.line => speaker == null ? 'L($text)' : 'L[$speaker]($text)',
        BlockKind.pause => 'P($seconds)',
        BlockKind.cue => 'C($cue)',
      };
}

class Script {
  const Script({
    required this.title,
    required this.blocks,
    this.style,
    this.music,
    this.voices = const {},
    this.description,
  });

  final String title;
  final List<Block> blocks;

  /// Style id from the front matter (`style: podcast`), if any.
  final String? style;

  /// Music theme from the front matter (`music: cosmic` / `music: off`), if any.
  final String? music;

  /// Speaker tag -> voice or character name, from the front matter.
  final Map<String, String> voices;
  final String? description;

  /// Speakers in order of first appearance.
  List<String> get speakers {
    final seen = <String>{};
    for (final b in blocks) {
      if (b.speaker != null) seen.add(b.speaker!);
    }
    return seen.toList();
  }

  int get wordCount {
    var n = 0;
    for (final b in blocks) {
      if (b.isSpoken) n += RegExp(r'\S+').allMatches(b.text).length;
    }
    return n;
  }
}
