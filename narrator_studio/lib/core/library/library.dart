/// Your documents: their text, chosen style and voices, and where you stopped.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

class Doc {
  Doc({
    required this.id,
    required this.title,
    required this.kind,
    required this.created,
    DateTime? opened,
    this.style,
    this.music,
    Map<String, String>? cast,
    this.narrator,
    this.speed,
    this.position = 0,
    this.utterances = 0,
    this.words = 0,
    this.summary,
    List<String>? topics,
    this.chapters = 0,
  })  : opened = opened ?? created,
        cast = cast ?? {},
        topics = topics ?? [];

  final String id;
  String title;

  /// txt, md, pdf, pasted, shared, sample
  final String kind;
  final DateTime created;
  DateTime opened;

  /// Style override (null: from the script / a guess).
  String? style;

  /// Music theme override, or 'off'.
  String? music;

  /// Speaker tag -> voice id or `char:<character id>`.
  final Map<String, String> cast;

  /// Narrator override (voice id or `char:<id>`).
  String? narrator;

  /// Your speed multiplier for this document (null: the global setting).
  double? speed;

  /// Utterance index to resume from.
  int position;
  int utterances;
  int words;

  /// From the script's front matter (`summary:`, `topics:`) and its index;
  /// kept here so the library can show and sort by them without parsing.
  String? summary;
  List<String> topics;
  int chapters;

  double get progress => utterances == 0 ? 0 : math.min(1, position / utterances);

  Map<String, Object?> toJson() => {
        'id': id,
        'title': title,
        'kind': kind,
        'created': created.toIso8601String(),
        'opened': opened.toIso8601String(),
        'style': style,
        'music': music,
        'cast': cast,
        'narrator': narrator,
        'speed': speed,
        'position': position,
        'utterances': utterances,
        'words': words,
        'summary': summary,
        'topics': topics,
        'chapters': chapters,
      };

  static Doc fromJson(Map<String, Object?> j) => Doc(
        id: j['id']! as String,
        title: j['title']! as String,
        kind: (j['kind'] as String?) ?? 'txt',
        created: DateTime.parse(j['created']! as String),
        opened: DateTime.tryParse((j['opened'] as String?) ?? ''),
        style: j['style'] as String?,
        music: j['music'] as String?,
        cast: (j['cast'] as Map?)?.map((k, v) => MapEntry(k as String, v as String)),
        narrator: j['narrator'] as String?,
        speed: (j['speed'] as num?)?.toDouble(),
        position: (j['position'] as num?)?.toInt() ?? 0,
        utterances: (j['utterances'] as num?)?.toInt() ?? 0,
        words: (j['words'] as num?)?.toInt() ?? 0,
        summary: j['summary'] as String?,
        topics: (j['topics'] as List?)?.whereType<String>().toList(),
        chapters: (j['chapters'] as num?)?.toInt() ?? 0,
      );
}

class Library extends ChangeNotifier {
  Library(this.dir);

  /// `<documents>/library`
  final Directory dir;
  final List<Doc> _docs = [];
  bool _loaded = false;

  File get _index => File(p.join(dir.path, 'library.json'));
  File textFile(String id) => File(p.join(dir.path, 'texts', '$id.txt'));

  /// Most recently opened first.
  List<Doc> get docs => List.unmodifiable(_docs..sort((a, b) => b.opened.compareTo(a.opened)));

  Doc? byId(String id) {
    for (final d in _docs) {
      if (d.id == id) return d;
    }
    return null;
  }

  Future<void> load() async {
    if (_loaded) return;
    _loaded = true;
    if (!_index.existsSync()) return;
    try {
      final list = jsonDecode(await _index.readAsString()) as List<Object?>;
      _docs
        ..clear()
        ..addAll(list.cast<Map<String, Object?>>().map(Doc.fromJson));
    } catch (_) {
      // A damaged index: keep a copy and start fresh rather than crash.
      await _index.copy('${_index.path}.broken');
    }
    notifyListeners();
  }

  Future<void> _saving = Future.value();

  /// Saves are queued: two overlapping writes would race on the temp file.
  Future<void> save() => _saving = _saving.catchError((Object _) {}).then((_) => _write());

  Future<void> _write() async {
    await dir.create(recursive: true);
    final tmp = File('${_index.path}.tmp');
    await tmp.writeAsString(jsonEncode(_docs.map((d) => d.toJson()).toList()), flush: true);
    await tmp.rename(_index.path);
  }

  Future<Doc> add({required String title, required String kind, required String text}) async {
    final id = '${DateTime.now().millisecondsSinceEpoch.toRadixString(36)}${math.Random().nextInt(1 << 20).toRadixString(36)}';
    final f = textFile(id);
    await f.parent.create(recursive: true);
    await f.writeAsString(text, flush: true);
    final doc = Doc(id: id, title: title, kind: kind, created: DateTime.now());
    _docs.add(doc);
    await save();
    notifyListeners();
    return doc;
  }

  Future<String> readText(Doc doc) => textFile(doc.id).readAsString();

  Future<void> updateText(Doc doc, String text) async {
    await textFile(doc.id).writeAsString(text, flush: true);
    doc.position = 0;
    await save();
    notifyListeners();
  }

  Future<void> remove(Doc doc) async {
    _docs.removeWhere((d) => d.id == doc.id);
    final f = textFile(doc.id);
    if (f.existsSync()) await f.delete();
    await save();
    notifyListeners();
  }

  Future<void> touch(Doc doc) async {
    doc.opened = DateTime.now();
    await save();
    notifyListeners();
  }

  Future<void> changed(Doc doc) async {
    await save();
    notifyListeners();
  }
}
