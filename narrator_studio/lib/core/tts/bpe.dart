/// The Qwen2 tokenizer (byte-level BPE), as VibeVoice reads text.
///
/// Loads a Hugging Face `tokenizer.json`: the vocabulary and merge ranks.
/// Text is split with Qwen2's pre-tokenizer pattern, each piece's UTF-8
/// bytes are mapped to printable characters (GPT-2's byte encoder), and
/// adjacent symbols are merged lowest rank first. tool/vibevoice checks the
/// ids against the Python tokenizer.
library;

import 'dart:convert';
import 'dart:io';

class Bpe {
  Bpe._(this._vocab, this._ranks);

  factory Bpe.fromFile(String path) {
    final j = jsonDecode(File(path).readAsStringSync()) as Map<String, Object?>;
    final model = j['model']! as Map<String, Object?>;
    final vocab = (model['vocab']! as Map<String, Object?>).map((k, v) => MapEntry(k, (v! as num).toInt()));
    final ranks = <String, int>{};
    final merges = model['merges']! as List<Object?>;
    for (var i = 0; i < merges.length; i++) {
      final m = merges[i];
      // Either "a b" or ["a", "b"], depending on the tokenizers version.
      final key = m is String ? m : (m! as List<Object?>).join(' ');
      ranks.putIfAbsent(key, () => i);
    }
    return Bpe._(vocab, ranks);
  }

  final Map<String, int> _vocab;
  final Map<String, int> _ranks;
  final _cache = <String, List<int>>{};

  /// Qwen2's split pattern; `(?i:...)` spelled out (Dart has no inline flags).
  static final _split = RegExp(
    r"""'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    unicode: true,
  );

  static final List<String> _byteChars = _byteEncoder();

  /// GPT-2's map from bytes to characters that are never whitespace or controls.
  static List<String> _byteEncoder() {
    final bs = <int>[
      for (var b = 0x21; b <= 0x7E; b++) b,
      for (var b = 0xA1; b <= 0xAC; b++) b,
      for (var b = 0xAE; b <= 0xFF; b++) b,
    ];
    final out = List<String>.filled(256, '');
    var n = 0;
    for (var b = 0; b < 256; b++) {
      out[b] = String.fromCharCode(bs.contains(b) ? b : 256 + n++);
    }
    return out;
  }

  List<int> encode(String text) {
    final ids = <int>[];
    var at = 0;
    void piece(String p) => ids.addAll(_cache[p] ??= _word(p));
    for (final m in _split.allMatches(text)) {
      if (m.start > at) piece(text.substring(at, m.start)); // kept, as the Split does
      piece(m[0]!);
      at = m.end;
    }
    if (at < text.length) piece(text.substring(at));
    return ids;
  }

  List<int> _word(String piece) {
    final symbols = [for (final b in utf8.encode(piece)) _byteChars[b]];
    while (symbols.length > 1) {
      var best = -1;
      var bestRank = 1 << 62;
      for (var i = 0; i + 1 < symbols.length; i++) {
        final r = _ranks['${symbols[i]} ${symbols[i + 1]}'];
        if (r != null && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      symbols[best] = symbols[best] + symbols[best + 1];
      symbols.removeAt(best + 1);
    }
    return [
      for (final s in symbols) _vocab[s] ?? (throw StateError('token "$s" is not in the vocabulary')),
    ];
  }
}
