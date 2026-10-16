/// Voice mixes: new Kokoro voices made by blending existing ones
/// (e.g. 60% George + 40% Onyx = an older, deeper professor).
///
/// Kokoro describes each voice as a table of style vectors in voices.bin.
/// A mix is a weighted average of those tables, written into one of the
/// spare slots (see [kokoroMixSlots]) of the app's own copy of the file.
library;

import 'dart:io';
import 'dart:typed_data';

import 'voices.dart';

/// Floats per voice in voices.bin: 510 positions x 256 style dims.
const kokoroVoiceFloats = 510 * 256;

class VoiceMix {
  const VoiceMix({required this.id, required this.name, required this.parts, required this.slot});

  final String id;
  final String name;

  /// Kokoro voice name (e.g. `bm_george`) -> weight.
  final Map<String, double> parts;

  /// Index into [kokoroMixSlots].
  final int slot;

  int get sid => kokoroSpeakerNames.indexOf(kokoroMixSlots[slot]);

  String get voiceId => 'mix:$id';

  String get recipe {
    final total = parts.values.fold<double>(0, (a, b) => a + b);
    return parts.entries
        .map((e) => '${(e.value / total * 100).round()}% ${kokoroVoices.firstWhere((v) => v.id == 'kokoro:${e.key}').name}')
        .join(' + ');
  }

  Voice toVoice() => Voice(
        id: voiceId,
        name: name,
        engine: Engine.kokoro,
        sid: sid,
        group: 'Mixes',
        blurb: 'Your mix: $recipe',
        license: 'Apache-2.0',
      );

  Map<String, Object?> toJson() => {'id': id, 'name': name, 'parts': parts, 'slot': slot};

  static VoiceMix fromJson(Map<String, Object?> j) => VoiceMix(
        id: j['id']! as String,
        name: j['name']! as String,
        parts: (j['parts']! as Map).map((k, v) => MapEntry(k as String, (v as num).toDouble())),
        slot: (j['slot']! as num).toInt(),
      );
}

/// Blends [parts] (voice index -> weight) out of [table].
Float32List blendVoices(Float32List table, Map<int, double> parts) {
  final total = parts.values.fold<double>(0, (a, b) => a + b);
  final out = Float32List(kokoroVoiceFloats);
  for (final e in parts.entries) {
    final w = e.value / total;
    final base = e.key * kokoroVoiceFloats;
    for (var i = 0; i < kokoroVoiceFloats; i++) {
      out[i] += table[base + i] * w;
    }
  }
  return out;
}

/// Writes [original] with every mix in its slot to [out]; returns the path
/// Kokoro should load ([original] itself when there are no mixes).
Future<String> writeMixedVoices({required String original, required String out, required List<VoiceMix> mixes}) async {
  if (mixes.isEmpty) return original;
  final bytes = await File(original).readAsBytes();
  final table = bytes.buffer.asFloat32List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 4);
  final copy = Float32List.fromList(table);
  for (final m in mixes) {
    final parts = <int, double>{};
    m.parts.forEach((name, w) {
      final i = kokoroSpeakerNames.indexOf(name);
      if (i >= 0 && w > 0) parts[i] = w;
    });
    if (parts.isEmpty) continue;
    final blended = blendVoices(table, parts);
    copy.setRange(m.sid * kokoroVoiceFloats, (m.sid + 1) * kokoroVoiceFloats, blended);
  }
  final tmp = File('$out.tmp');
  await tmp.writeAsBytes(copy.buffer.asUint8List(), flush: true);
  await tmp.rename(out);
  return out;
}
