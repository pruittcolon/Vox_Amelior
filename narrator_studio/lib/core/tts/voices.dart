/// Every voice the app can speak with, and the character presets built on them.
library;

enum Engine { kokoro, supertonic, pocket, vibevoice }

extension EngineInfo on Engine {
  /// Voice pack that must be downloaded for this engine.
  String get pack => switch (this) {
        Engine.kokoro => 'kokoro',
        Engine.supertonic => 'supertonic',
        Engine.pocket => 'pocket',
        Engine.vibevoice => 'vibevoice',
      };
}

class Voice {
  const Voice({
    required this.id,
    required this.name,
    required this.engine,
    required this.blurb,
    this.sid = 0,
    this.group = '',
    this.refPath,
    this.license = '',
    this.tone = '',
  });

  /// `kokoro:af_heart`, `supertonic:3`, `gallery:<id>`, `clone:<id>`, `mix:<id>`.
  final String id;
  final String name;
  final Engine engine;
  final String blurb;

  /// Speaker id inside the model (Kokoro, Supertonic, voice mixes).
  final int sid;

  /// Filter group shown in the voice browser, e.g. "American", "British".
  final String group;

  /// Reference clip for Pocket TTS voices (absolute path once downloaded).
  final String? refPath;
  final String license;

  /// deep / low / medium / high, when known.
  final String tone;

  Voice copyWith({String? refPath, String? name}) => Voice(
        id: id,
        name: name ?? this.name,
        engine: engine,
        blurb: blurb,
        sid: sid,
        group: group,
        refPath: refPath ?? this.refPath,
        license: license,
        tone: tone,
      );
}

/// Kokoro v1.0 speakers, in the model's speaker-id order (checked against the
/// model's `speaker_names` metadata in CI).
const kokoroSpeakerNames = <String>[
  'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river',
  'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx',
  'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george',
  'bm_lewis', 'ef_dora', 'em_alex', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara',
  'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex',
  'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia',
  'zm_yunyang', 'em_santa',
];

/// Kokoro slots that voice mixes may overwrite in the app's private copy of
/// voices.bin (the Mandarin-trained voices; they are also the least clear in
/// English in the CI listening test).
const kokoroMixSlots = <String>[
  'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang',
];

const _kokoroInfo = <String, (String, String, String)>{
  // name, group, blurb
  'af_heart': ('Heart', 'American', 'Warm, natural female narrator. The best all-rounder.'),
  'af_bella': ('Bella', 'American', 'Bright, expressive female voice.'),
  'af_nicole': ('Nicole', 'American', 'Soft, breathy and close. Bedtime stories.'),
  'af_aoede': ('Aoede', 'American', 'Smooth, even female voice.'),
  'af_kore': ('Kore', 'American', 'Clear, confident female voice.'),
  'af_sarah': ('Sarah', 'American', 'Friendly, conversational female voice.'),
  'af_nova': ('Nova', 'American', 'Crisp, modern female voice.'),
  'af_sky': ('Sky', 'American', 'Light, youthful female voice.'),
  'af_alloy': ('Alloy', 'American', 'Neutral, steady female voice.'),
  'af_jessica': ('Jessica', 'American', 'Casual, chatty female voice.'),
  'af_river': ('River', 'American', 'Relaxed, easygoing female voice.'),
  'am_michael': ('Michael', 'American', 'Warm, steady male voice. Great host.'),
  'am_fenrir': ('Fenrir', 'American', 'Deep, strong male voice.'),
  'am_puck': ('Puck', 'American', 'Lively, playful male voice.'),
  'am_echo': ('Echo', 'American', 'Calm, measured male voice.'),
  'am_eric': ('Eric', 'American', 'Plain-spoken male voice.'),
  'am_liam': ('Liam', 'American', 'Young male voice.'),
  'am_onyx': ('Onyx', 'American', 'Deep, gravelly male voice.'),
  'am_adam': ('Adam', 'American', 'Rough, casual male voice.'),
  'am_santa': ('Santa', 'American', 'Jolly, booming old man.'),
  'bf_emma': ('Emma', 'British', 'Polished British female voice.'),
  'bf_isabella': ('Isabella', 'British', 'Warm British female voice.'),
  'bf_alice': ('Alice', 'British', 'Bright British female voice.'),
  'bf_lily': ('Lily', 'British', 'Soft British female voice.'),
  'bm_george': ('George', 'British', 'Distinguished, older British gentleman.'),
  'bm_fable': ('Fable', 'British', 'British storyteller.'),
  'bm_lewis': ('Lewis', 'British', 'Deep British male voice.'),
  'bm_daniel': ('Daniel', 'British', 'Calm British male voice.'),
  'ef_dora': ('Dora', 'Accented', 'English with a Spanish flavor (female).'),
  'em_alex': ('Alex', 'Accented', 'English with a Spanish flavor (male).'),
  'em_santa': ('Santa (ES)', 'Accented', 'Jolly old man, Spanish flavor.'),
  'ff_siwis': ('Siwis', 'Accented', 'English with a French flavor (female).'),
  'hf_alpha': ('Alpha', 'Accented', 'English with a Hindi flavor (female).'),
  'hf_beta': ('Beta', 'Accented', 'English with a Hindi flavor (female).'),
  'hm_omega': ('Omega', 'Accented', 'English with a Hindi flavor (male).'),
  'hm_psi': ('Psi', 'Accented', 'English with a Hindi flavor (male).'),
  'if_sara': ('Sara', 'Accented', 'English with an Italian flavor (female).'),
  'im_nicola': ('Nicola', 'Accented', 'English with an Italian flavor (male).'),
  'jf_alpha': ('Alpha (JA)', 'Accented', 'English with a Japanese flavor (female).'),
  'jf_gongitsune': ('Gongitsune', 'Accented', 'English with a Japanese flavor (female).'),
  'jf_nezumi': ('Nezumi', 'Accented', 'English with a Japanese flavor (female).'),
  'jf_tebukuro': ('Tebukuro', 'Accented', 'English with a Japanese flavor (female).'),
  'jm_kumo': ('Kumo', 'Accented', 'English with a Japanese flavor (male).'),
  'pf_dora': ('Dora (PT)', 'Accented', 'English with a Portuguese flavor (female).'),
  'pm_alex': ('Alex (PT)', 'Accented', 'English with a Portuguese flavor (male).'),
  'pm_santa': ('Santa (PT)', 'Accented', 'Jolly old man, Portuguese flavor.'),
};

final List<Voice> kokoroVoices = [
  for (var i = 0; i < kokoroSpeakerNames.length; i++)
    if (_kokoroInfo.containsKey(kokoroSpeakerNames[i]))
      Voice(
        id: 'kokoro:${kokoroSpeakerNames[i]}',
        name: _kokoroInfo[kokoroSpeakerNames[i]]!.$1,
        group: _kokoroInfo[kokoroSpeakerNames[i]]!.$2,
        blurb: _kokoroInfo[kokoroSpeakerNames[i]]!.$3,
        engine: Engine.kokoro,
        sid: i,
        license: 'Apache-2.0',
      ),
];

/// Supertonic 3 voice styles in speaker-id order (ids 0-4 measured at
/// 160-196 Hz, 5-9 at 87-138 Hz in the CI samples).
const supertonicStyles = <String>['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'];

const _supertonicInfo = <String, (String, String)>{
  'F1': ('Fiona', 'Clear, mid-pitched female voice.'),
  'F2': ('Faye', 'Bright female voice.'),
  'F3': ('Freya', 'Warm, lower female voice.'),
  'F4': ('Flora', 'Light, high female voice.'),
  'F5': ('Frankie', 'Calm, low female voice.'),
  'M1': ('Miles', 'Light, youthful male voice.'),
  'M2': ('Marcus', 'Deep male voice.'),
  'M3': ('Martin', 'Deep, steady male voice.'),
  'M4': ('Morgan', 'Mid-range male voice.'),
  'M5': ('Max', 'Very deep male voice.'),
};

final List<Voice> supertonicVoices = [
  for (var i = 0; i < supertonicStyles.length; i++)
    Voice(
      id: 'supertonic:${supertonicStyles[i]}',
      name: _supertonicInfo[supertonicStyles[i]]!.$1,
      group: 'Studio',
      blurb: '${_supertonicInfo[supertonicStyles[i]]!.$2} Studio quality, full 44 kHz.',
      engine: Engine.supertonic,
      sid: i,
      license: 'OpenRAIL',
    ),
];

/// VibeVoice-Realtime's English voices (Microsoft's presets, MIT).
const vibevoiceSpeakers = <String, (String, String)>{
  'carter': ('Carter', 'man'),
  'davis': ('Davis', 'man'),
  'emma': ('Emma', 'woman'),
  'frank': ('Frank', 'man'),
  'grace': ('Grace', 'woman'),
  'mike': ('Mike', 'man'),
};

final List<Voice> vibevoiceVoices = [
  for (final e in vibevoiceSpeakers.entries)
    Voice(
      id: 'vibevoice:${e.key}',
      name: e.value.$1,
      group: 'Lifelike',
      blurb: 'Lifelike ${e.value.$2}\'s voice (VibeVoice): the most human-sounding in the app.',
      engine: Engine.vibevoice,
      license: 'MIT',
    ),
];

/// 'f' or 'm' for a voice, when its name says so (Kokoro af_/am_..., Supertonic
/// F1/M1..., VibeVoice's presets).
String? genderOf(String voiceId) {
  final i = voiceId.indexOf(':');
  final engine = i < 0 ? '' : voiceId.substring(0, i);
  final name = voiceId.substring(i + 1);
  return switch (engine) {
    'kokoro' when name.length > 2 && name[2] == '_' => name[1] == 'm' ? 'm' : 'f',
    'supertonic' when name.isNotEmpty => name[0] == 'M' ? 'm' : 'f',
    'vibevoice' => vibevoiceSpeakers[name]?.$2 == 'woman' ? 'f' : 'm',
    _ => null,
  };
}

/// The voices of one set's engine, by gender: what a speaker gets in that
/// set when nothing more specific is known.
const setVoices = <String, Map<String, List<String>>>{
  'natural': {
    'f': ['kokoro:af_heart', 'kokoro:af_bella', 'kokoro:bf_emma', 'kokoro:af_nicole'],
    'm': ['kokoro:am_michael', 'kokoro:bm_george', 'kokoro:am_fenrir', 'kokoro:am_puck'],
  },
  'studio': {
    'f': ['supertonic:F1', 'supertonic:F2', 'supertonic:F3', 'supertonic:F4', 'supertonic:F5'],
    'm': ['supertonic:M1', 'supertonic:M2', 'supertonic:M3', 'supertonic:M4', 'supertonic:M5'],
  },
  'lifelike': {
    'f': ['vibevoice:emma', 'vibevoice:grace'],
    'm': ['vibevoice:carter', 'vibevoice:davis', 'vibevoice:frank', 'vibevoice:mike'],
  },
};

/// The engine behind each voice set.
const setEngines = {'natural': Engine.kokoro, 'studio': Engine.supertonic, 'lifelike': Engine.vibevoice};

/// A character: a voice plus how it speaks. Scripts refer to these by name
/// (`GUEST: The Professor`) and styles pick them for common roles.
class Character {
  const Character({
    required this.id,
    required this.name,
    required this.voiceId,
    required this.blurb,
    this.speed = 1.0,
    this.fallbackVoiceId,
    this.studioVoiceId,
    this.lifelikeVoiceId,
  });

  final String id;
  final String name;

  /// The natural voice: the most human-sounding one for the part.
  final String voiceId;
  final String blurb;
  final double speed;

  /// Used when [voiceId]'s pack (or clip) isn't downloaded.
  final String? fallbackVoiceId;

  /// The part's voice in the Studio set (Supertonic: crisper, flatter).
  final String? studioVoiceId;

  /// The part's voice in the Lifelike set (VibeVoice).
  final String? lifelikeVoiceId;

  Map<String, Object?> toJson() => {
        'id': id,
        'name': name,
        'voiceId': voiceId,
        'blurb': blurb,
        'speed': speed,
      };

  static Character fromJson(Map<String, Object?> j) => Character(
        id: j['id']! as String,
        name: j['name']! as String,
        voiceId: j['voiceId']! as String,
        blurb: (j['blurb'] as String?) ?? '',
        speed: ((j['speed'] as num?) ?? 1.0).toDouble(),
      );
}

/// The voice sets: "natural" (Kokoro, human-sounding and fast; the default),
/// "studio" (Supertonic HD: crisper, flatter) or "lifelike" (VibeVoice: the
/// most human-sounding, and the slowest to make).
const voiceSets = ['natural', 'studio', 'lifelike'];

/// Kokoro voices with the highest grades from its authors (af_heart A,
/// af_bella A-, bf_emma and af_nicole B-, am_michael and am_fenrir C+) carry
/// the main parts; each has a Studio stand-in.
const builtInCharacters = <Character>[
  Character(
    id: 'narrator',
    name: 'The Narrator',
    voiceId: 'kokoro:af_heart',
    studioVoiceId: 'supertonic:F1',
    lifelikeVoiceId: 'vibevoice:emma',
    blurb: 'Warm and clear. Reads books beautifully.',
    speed: 0.97,
  ),
  Character(
    id: 'storyteller',
    name: 'The Storyteller',
    voiceId: 'kokoro:bm_fable',
    studioVoiceId: 'supertonic:M2',
    lifelikeVoiceId: 'vibevoice:davis',
    blurb: 'A British storyteller by the fire.',
    speed: 0.93,
  ),
  Character(
    id: 'professor',
    name: 'The Professor',
    voiceId: 'kokoro:bm_george',
    studioVoiceId: 'supertonic:M3',
    lifelikeVoiceId: 'vibevoice:frank',
    blurb: 'Distinguished and thoughtful. Explains big ideas slowly.',
    speed: 0.92,
  ),
  Character(
    id: 'sage',
    name: 'The Old Sage',
    voiceId: 'kokoro:am_onyx',
    studioVoiceId: 'supertonic:M5',
    lifelikeVoiceId: 'vibevoice:davis',
    blurb: 'Deep, unhurried, a little gravelly. For a wise guest.',
    speed: 0.88,
  ),
  Character(
    id: 'host',
    name: 'Radio Host',
    voiceId: 'kokoro:am_michael',
    studioVoiceId: 'supertonic:M1',
    lifelikeVoiceId: 'vibevoice:carter',
    blurb: 'Friendly, upbeat podcast host.',
    speed: 1.02,
  ),
  Character(
    id: 'cohost',
    name: 'Co-Host',
    voiceId: 'kokoro:af_bella',
    studioVoiceId: 'supertonic:F2',
    lifelikeVoiceId: 'vibevoice:grace',
    blurb: 'Bright and curious; asks the questions listeners would.',
    speed: 1.02,
  ),
  Character(
    id: 'explorer',
    name: 'The Explorer',
    voiceId: 'kokoro:am_fenrir',
    studioVoiceId: 'supertonic:M4',
    lifelikeVoiceId: 'vibevoice:mike',
    blurb: 'Bold and adventurous. Documentaries and expeditions.',
    speed: 0.98,
  ),
  Character(
    id: 'anchor',
    name: 'News Anchor',
    voiceId: 'kokoro:bf_emma',
    studioVoiceId: 'supertonic:F5',
    lifelikeVoiceId: 'vibevoice:emma',
    blurb: 'Polished and precise.',
    speed: 1.04,
  ),
  Character(
    id: 'reporter',
    name: 'Field Reporter',
    voiceId: 'kokoro:am_puck',
    lifelikeVoiceId: 'vibevoice:mike',
    studioVoiceId: 'supertonic:M4',
    blurb: 'Lively, on-the-scene energy.',
    speed: 1.04,
  ),
  Character(
    id: 'night',
    name: 'Night Reader',
    voiceId: 'kokoro:af_nicole',
    studioVoiceId: 'supertonic:F3',
    lifelikeVoiceId: 'vibevoice:emma',
    blurb: 'Soft and close. For winding down.',
    speed: 0.9,
  ),
  Character(
    id: 'student',
    name: 'The Student',
    voiceId: 'kokoro:af_sky',
    studioVoiceId: 'supertonic:F4',
    lifelikeVoiceId: 'vibevoice:grace',
    blurb: 'Young and curious.',
    speed: 1.0,
  ),
  Character(
    id: 'maestro',
    name: 'Il Maestro',
    voiceId: 'kokoro:im_nicola',
    studioVoiceId: 'supertonic:M2',
    lifelikeVoiceId: 'vibevoice:davis',
    blurb: 'An Italian-flavored teacher of art and history.',
    speed: 0.93,
  ),
  Character(
    id: 'philosophe',
    name: 'La Philosophe',
    voiceId: 'kokoro:ff_siwis',
    studioVoiceId: 'supertonic:F4',
    lifelikeVoiceId: 'vibevoice:grace',
    blurb: 'A French-flavored philosopher.',
    speed: 0.94,
  ),
  Character(
    id: 'santa',
    name: 'Jolly Old Man',
    voiceId: 'kokoro:am_santa',
    studioVoiceId: 'supertonic:M5',
    lifelikeVoiceId: 'vibevoice:frank',
    blurb: 'Booming and merry.',
    speed: 0.95,
  ),
  Character(
    id: 'librarian',
    name: 'The Librarian',
    voiceId: 'gallery:voice-zero-caro-davy',
    fallbackVoiceId: 'kokoro:bf_isabella',
    studioVoiceId: 'supertonic:F3',
    lifelikeVoiceId: 'vibevoice:emma',
    blurb: 'A real LibriVox audiobook narrator (CC0). Needs the cloning pack.',
    speed: 1.0,
  ),
  Character(
    id: 'announcer',
    name: 'The Announcer',
    voiceId: 'gallery:alba-mackenna-announcer',
    fallbackVoiceId: 'kokoro:af_kore',
    studioVoiceId: 'supertonic:F5',
    lifelikeVoiceId: 'vibevoice:grace',
    blurb: 'Big announcer energy (voice-acted, CC-BY). Needs the cloning pack.',
    speed: 1.0,
  ),
];
