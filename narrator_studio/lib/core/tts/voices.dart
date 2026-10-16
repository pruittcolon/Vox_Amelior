/// Every voice the app can speak with, and the character presets built on them.
library;

enum Engine { kokoro, supertonic, pocket }

extension EngineInfo on Engine {
  /// Voice pack that must be downloaded for this engine.
  String get pack => switch (this) {
        Engine.kokoro => 'kokoro',
        Engine.supertonic => 'supertonic',
        Engine.pocket => 'pocket',
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
      group: 'Fast',
      blurb: '${_supertonicInfo[supertonicStyles[i]]!.$2} Very fast to generate.',
      engine: Engine.supertonic,
      sid: i,
      license: 'OpenRAIL',
    ),
];

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
  });

  final String id;
  final String name;
  final String voiceId;
  final String blurb;
  final double speed;

  /// Used when [voiceId]'s pack (or clip) isn't downloaded.
  final String? fallbackVoiceId;

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

const builtInCharacters = <Character>[
  Character(
    id: 'narrator',
    name: 'The Narrator',
    voiceId: 'kokoro:af_heart',
    blurb: 'Warm and clear. Reads books beautifully.',
    speed: 0.97,
  ),
  Character(
    id: 'storyteller',
    name: 'The Storyteller',
    voiceId: 'kokoro:bm_fable',
    blurb: 'A British storyteller by the fire.',
    speed: 0.93,
  ),
  Character(
    id: 'professor',
    name: 'The Professor',
    voiceId: 'kokoro:bm_george',
    blurb: 'Distinguished and thoughtful. Explains big ideas slowly.',
    speed: 0.92,
  ),
  Character(
    id: 'sage',
    name: 'The Old Sage',
    voiceId: 'kokoro:am_onyx',
    blurb: 'Deep, unhurried, a little gravelly. For a wise guest.',
    speed: 0.88,
  ),
  Character(
    id: 'host',
    name: 'Radio Host',
    voiceId: 'kokoro:am_michael',
    blurb: 'Friendly, upbeat podcast host.',
    speed: 1.02,
  ),
  Character(
    id: 'cohost',
    name: 'Co-Host',
    voiceId: 'kokoro:af_bella',
    blurb: 'Bright and curious; asks the questions listeners would.',
    speed: 1.02,
  ),
  Character(
    id: 'explorer',
    name: 'The Explorer',
    voiceId: 'kokoro:am_fenrir',
    blurb: 'Bold and adventurous. Documentaries and expeditions.',
    speed: 0.98,
  ),
  Character(
    id: 'anchor',
    name: 'News Anchor',
    voiceId: 'kokoro:bf_emma',
    blurb: 'Polished and precise.',
    speed: 1.04,
  ),
  Character(
    id: 'reporter',
    name: 'Field Reporter',
    voiceId: 'kokoro:am_puck',
    blurb: 'Lively, on-the-scene energy.',
    speed: 1.04,
  ),
  Character(
    id: 'night',
    name: 'Night Reader',
    voiceId: 'kokoro:af_nicole',
    blurb: 'Soft and close. For winding down.',
    speed: 0.9,
  ),
  Character(
    id: 'student',
    name: 'The Student',
    voiceId: 'kokoro:af_sky',
    blurb: 'Young and curious.',
    speed: 1.0,
  ),
  Character(
    id: 'maestro',
    name: 'Il Maestro',
    voiceId: 'kokoro:im_nicola',
    blurb: 'An Italian-flavored teacher of art and history.',
    speed: 0.93,
  ),
  Character(
    id: 'philosophe',
    name: 'La Philosophe',
    voiceId: 'kokoro:ff_siwis',
    blurb: 'A French-flavored philosopher.',
    speed: 0.94,
  ),
  Character(
    id: 'santa',
    name: 'Jolly Old Man',
    voiceId: 'kokoro:am_santa',
    blurb: 'Booming and merry.',
    speed: 0.95,
  ),
  Character(
    id: 'librarian',
    name: 'The Librarian',
    voiceId: 'gallery:voice-zero-caro-davy',
    fallbackVoiceId: 'kokoro:bf_isabella',
    blurb: 'A real LibriVox audiobook narrator (CC0). Needs the cloning pack.',
    speed: 1.0,
  ),
  Character(
    id: 'announcer',
    name: 'The Announcer',
    voiceId: 'gallery:alba-mackenna-announcer',
    fallbackVoiceId: 'kokoro:af_kore',
    blurb: 'Big announcer energy (voice-acted, CC-BY). Needs the cloning pack.',
    speed: 1.0,
  ),
];
