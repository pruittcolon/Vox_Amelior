/// Ties everything together: packs, voices, library, settings, the speech
/// engine, and turning a document into something playable.
library;

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import 'audio/native.dart';
import 'library/library.dart';
import 'library/settings.dart';
import 'models/model_store.dart';
import 'render/music.dart';
import 'render/plan.dart';
import 'render/styles.dart';
import 'script/script.dart';
import 'script/script_parser.dart';
import 'tts/synth.dart';
import 'tts/voice_mix.dart';
import 'tts/voices.dart';

class NoVoicesError implements Exception {
  @override
  String toString() => 'Download a voice pack first (Voices tab).';
}

/// A document ready to play or export.
class Session {
  Session({
    required this.doc,
    required this.script,
    required this.style,
    required this.plan,
    required this.theme,
    required this.music,
    required this.cast,
    required this.castRefs,
  });

  final Doc doc;
  final Script script;
  final ReadingStyle style;
  final Plan plan;
  final MusicTheme? theme;
  final bool music;

  /// Speaker (null = narrator) -> voice and speed.
  final Map<String?, VoiceSpec> cast;

  /// Speaker -> the choice behind it (`char:host`, `kokoro:af_heart`...).
  final Map<String?, String> castRefs;

  VoiceSpec voiceFor(String? speaker) => cast[speaker] ?? cast[null]!;
}

class AppState extends ChangeNotifier {
  AppState({
    required this.supportDir,
    required this.documentsDir,
    required this.native,
    required Synth Function(AppState app) synth,
    ModelStore? models,
  })  : models = models ?? ModelStore(Directory(p.join(supportDir.path, 'models'))),
        library = Library(Directory(p.join(documentsDir.path, 'library'))),
        settings = Settings(File(p.join(supportDir.path, 'settings.json'))) {
    this.synth = synth(this);
    this.models.addListener(notifyListeners);
    library.addListener(notifyListeners);
    settings.addListener(notifyListeners);
  }

  final Directory supportDir;
  final Directory documentsDir;
  final NativeApi native;
  final ModelStore models;
  final Library library;
  final Settings settings;
  late final Synth synth;

  Future<void> init() async {
    await settings.load();
    await library.load();
    if (settings.mixes.isNotEmpty && !File(mixedVoicesPath).existsSync() && hasKokoro) {
      await rebuildMixes();
    }
  }

  // ---- Paths ---------------------------------------------------------------

  String get mixedVoicesPath => p.join(supportDir.path, 'kokoro-voices-mixed.bin');

  /// The installed Kokoro variant: the fast one if present, else compact.
  String get kokoroPack => models.isInstalled('kokoro') || !models.isInstalled('kokoro-lite') ? 'kokoro' : 'kokoro-lite';

  bool get hasKokoro => models.isInstalled('kokoro') || models.isInstalled('kokoro-lite');

  /// True once any voice pack is ready.
  bool get hasAnyVoices => hasKokoro || models.isInstalled('supertonic');

  /// The Clarity upscaler's folder, or '' when it is off or not downloaded.
  String get clarityDir => settings.clarity && models.isInstalled('clarity') ? models.packDir('clarity').path : '';

  String get kokoroVoicesFile {
    final original = models.path(kokoroPack, 'voices.bin');
    if (settings.mixes.isEmpty) return original;
    return File(mixedVoicesPath).existsSync() ? mixedVoicesPath : original;
  }

  String cloneFile(ClonedVoice c) => p.join(supportDir.path, 'clones', c.file);
  Directory get clonesDir => Directory(p.join(supportDir.path, 'clones'));
  Directory get exportsDir => Directory(p.join(documentsDir.path, 'exports'));
  Directory get cacheDir => Directory(p.join(supportDir.path, 'cache'));

  // ---- Voices --------------------------------------------------------------

  List<Character> get characters => [...builtInCharacters, ...settings.characters];

  /// Every voice you could pick, downloaded or not.
  List<Voice> get allVoices => [
        ...settings.mixes.map((m) => m.toVoice()),
        ...settings.clones.map(_cloneVoice),
        ...settings.gallery.map(_galleryVoice),
        ...kokoroVoices,
        ...supertonicVoices,
      ];

  Voice _cloneVoice(ClonedVoice c) => Voice(
        id: c.voiceId,
        name: c.name,
        engine: Engine.pocket,
        group: 'Cloned',
        blurb: 'Cloned from your clip',
        refPath: cloneFile(c),
        license: 'Yours',
      );

  Voice _galleryVoice(SavedGalleryVoice g) => Voice(
        id: g.voiceId,
        name: g.name,
        engine: Engine.pocket,
        group: 'Gallery',
        blurb: g.description,
        refPath: models.path('gallery', g.asset),
        license: g.license,
      );

  /// Gallery voices used by built-in characters, so they work after the clip
  /// is downloaded even if you never added them to your list.
  Voice? _builtInGalleryVoice(String id) {
    final clip = id.substring('gallery:'.length);
    final asset = 'voice--$clip.wav';
    return Voice(
      id: id,
      name: clip,
      engine: Engine.pocket,
      group: 'Gallery',
      blurb: '',
      refPath: models.path('gallery', asset),
    );
  }

  Voice? voiceById(String id) {
    for (final v in allVoices) {
      if (v.id == id) return v;
    }
    if (id.startsWith('gallery:')) return _builtInGalleryVoice(id);
    if (id.startsWith('kokoro:')) {
      // Accept bare names too: kokoro:george
      final name = id.substring(7).toLowerCase();
      for (final v in kokoroVoices) {
        if (v.name.toLowerCase() == name) return v;
      }
    }
    return null;
  }

  bool isAvailable(Voice v) {
    if (v.engine == Engine.kokoro) return hasKokoro;
    if (!models.isInstalled(v.engine.pack)) return false;
    if (v.engine == Engine.pocket) return v.refPath != null && File(v.refPath!).existsSync();
    return true;
  }

  Character? characterById(String id) {
    for (final c in characters) {
      if (c.id == id) return c;
    }
    return null;
  }

  /// Display name for a cast choice.
  String refName(String ref) {
    if (ref.startsWith('char:')) return characterById(ref.substring(5))?.name ?? ref;
    return voiceById(ref)?.name ?? ref;
  }

  /// Finds a character or voice by what a script calls it:
  /// "The Professor", "professor", "George", "bm_george", "kokoro:bm_george".
  String? matchRef(String name) {
    String norm(String s) => s.toLowerCase().replaceFirst(RegExp(r'^the\s+'), '').replaceAll(RegExp(r'[^a-z0-9]'), '');
    final n = norm(name);
    if (n.isEmpty) return null;
    for (final c in characters) {
      if (norm(c.name) == n || norm(c.id) == n) return 'char:${c.id}';
    }
    for (final v in allVoices) {
      final bare = v.id.contains(':') ? v.id.split(':').last : v.id;
      if (norm(v.id) == n || norm(bare) == n || norm(v.name) == n || norm('${v.engine.name} ${v.name}') == n) {
        return v.id;
      }
    }
    return null;
  }

  /// Turns a cast choice into a concrete, downloaded voice and speed.
  VoiceSpec resolve(String ref, {double extraSpeed = 1.0}) {
    var speed = 1.0;
    String voiceId = ref;
    String? fallback;
    if (ref.startsWith('char:')) {
      final c = characterById(ref.substring(5)) ?? builtInCharacters.first;
      speed = c.speed;
      voiceId = c.voiceId;
      fallback = c.fallbackVoiceId;
    }
    final candidates = [voiceId, ?fallback, 'kokoro:af_heart', 'supertonic:F1', 'supertonic:M1'];
    for (final id in candidates) {
      final v = voiceById(id);
      if (v != null && isAvailable(v)) {
        return VoiceSpec(v, (speed * extraSpeed).clamp(0.5, 2.0).toDouble());
      }
    }
    throw NoVoicesError();
  }

  /// Voices given to speakers nobody assigned, so each sounds different.
  static const _rotation = ['professor', 'cohost', 'explorer', 'anchor', 'storyteller', 'sage', 'reporter', 'student'];

  /// Which character/voice reads [speaker] (null: the narrator).
  Map<String?, String> castRefs(Doc doc, Script script, ReadingStyle style) {
    final refs = <String?, String>{};
    refs[null] = doc.narrator ?? 'char:${style.narrator}';
    final used = <String>{refs[null]!};
    for (final speaker in script.speakers) {
      String? ref = doc.cast[speaker];
      if (ref == null && script.voices[speaker] != null) ref = matchRef(script.voices[speaker]!);
      if (ref == null && speaker == 'NARRATOR') ref = refs[null];
      if (ref == null && style.defaultCast[speaker] != null) ref = 'char:${style.defaultCast[speaker]}';
      ref ??= matchRef(speaker); // "PROFESSOR:" -> The Professor
      if (ref == null) {
        for (final c in _rotation) {
          if (!used.contains('char:$c')) {
            ref = 'char:$c';
            break;
          }
        }
      }
      ref ??= refs[null]!;
      refs[speaker] = ref;
      used.add(ref);
    }
    return refs;
  }

  Future<Session> openSession(Doc doc) async {
    final text = await library.readText(doc);
    final script = parseScript(text, fallbackTitle: doc.title);
    final styleId = doc.style ??
        script.style ??
        guessStyle(hasSpeakers: script.speakers.isNotEmpty, speakerCount: script.speakers.length, isPdf: doc.kind == 'pdf');
    final style = styleById(styleId);
    final musicId = doc.music ?? script.music ?? style.music;
    final music = settings.music && musicId != 'off';
    final plan = buildPlan(script, style, music: music);
    final refs = castRefs(doc, script, style);
    final userSpeed = doc.speed ?? settings.speed;
    final cast = <String?, VoiceSpec>{
      for (final e in refs.entries) e.key: resolve(e.value, extraSpeed: style.speed * userSpeed),
    };
    if (doc.utterances != plan.utterances.length || doc.words != plan.words) {
      doc.utterances = plan.utterances.length;
      doc.words = plan.words;
      if (doc.position >= plan.utterances.length) doc.position = 0;
      await library.changed(doc);
    }
    return Session(
      doc: doc,
      script: script,
      style: style,
      plan: plan,
      theme: themeById(musicId) ?? (music ? musicThemes.first : null),
      music: music,
      cast: cast,
      castRefs: refs,
    );
  }

  // ---- Voice mixes ---------------------------------------------------------

  Future<void> rebuildMixes() async {
    if (!hasKokoro) return;
    await writeMixedVoices(
      original: models.path(kokoroPack, 'voices.bin'),
      out: mixedVoicesPath,
      mixes: settings.mixes,
    );
    await synth.reloadKokoroVoices();
    notifyListeners();
  }

  int? freeMixSlot() {
    final used = settings.mixes.map((m) => m.slot).toSet();
    for (var i = 0; i < kokoroMixSlots.length; i++) {
      if (!used.contains(i)) return i;
    }
    return null;
  }
}
