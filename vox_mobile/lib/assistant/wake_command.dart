/// Detects a wake phrase such as "Hey Vox, ..." at the start of an utterance.
class WakeCommandParser {
  WakeCommandParser(List<String> phrases) : _regex = _build(phrases);

  final RegExp? _regex;

  static RegExp? _build(List<String> phrases) {
    final cleaned = phrases
        .map((p) => p.trim().toLowerCase())
        .where((p) => p.isNotEmpty)
        .map((p) => p.split(RegExp(r'\s+')).map(RegExp.escape).join(r'[\s,.!?-]+'))
        .toList()
      ..sort((a, b) => b.length.compareTo(a.length));
    if (cleaned.isEmpty) return null;
    return RegExp(
      r'^\s*(?:(?:hey|hi|ok|okay)[\s,]+)?(?:' + cleaned.join('|') + r')(?![\p{L}\p{N}])[\s,:.!?-]*(.*)$',
      caseSensitive: false,
      dotAll: true,
      unicode: true,
    );
  }

  /// Returns the command after the wake phrase (possibly empty), or null if
  /// the utterance does not start with a wake phrase.
  String? extract(String utterance) {
    final m = _regex?.firstMatch(utterance);
    if (m == null) return null;
    return m.group(1)!.trim();
  }
}
