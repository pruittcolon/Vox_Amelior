import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/models/model_catalog.dart';

/// Where models live on disk and whether each one is fully installed.
///
/// A model counts as installed only when its `.installed` marker exists and
/// every file it lists is present at the recorded size, so a half-finished
/// download is never mistaken for a working model.
class ModelStore {
  ModelStore(this.root);

  final Directory root;

  Directory dir(ModelAsset asset) => Directory(p.join(root.path, asset.id));

  File file(ModelAsset asset, String name) => File(p.join(dir(asset).path, name));

  bool isInstalled(ModelAsset asset) {
    final marker = File(p.join(dir(asset).path, '.installed'));
    if (!marker.existsSync()) return false;
    try {
      final files = (jsonDecode(marker.readAsStringSync()) as Map<String, Object?>)['files']! as Map<String, Object?>;
      if (!files.keys.toSet().containsAll(asset.installedFileNames)) return false;
      for (final entry in files.entries) {
        final f = file(asset, entry.key);
        if (!f.existsSync() || f.lengthSync() != entry.value) return false;
      }
      return true;
    } on Object {
      return false;
    }
  }

  /// Records a completed installation. Call only after every file is in place.
  void markInstalled(ModelAsset asset) {
    final files = {
      for (final name in asset.installedFileNames) name: file(asset, name).lengthSync(),
    };
    File(p.join(dir(asset).path, '.installed')).writeAsStringSync(jsonEncode({'files': files}));
  }

  void remove(ModelAsset asset) {
    final d = dir(asset);
    if (d.existsSync()) d.deleteSync(recursive: true);
  }

  /// Bytes used on disk by [asset] (including partial downloads).
  int usedBytes(ModelAsset asset) {
    final d = dir(asset);
    if (!d.existsSync()) return 0;
    return d.listSync(recursive: true).whereType<File>().fold<int>(0, (a, f) => a + f.lengthSync());
  }

  /// Bytes of a partially downloaded file, for showing resume progress.
  int partialBytes(ModelAsset asset) {
    final d = dir(asset);
    if (!d.existsSync()) return 0;
    return d
        .listSync(recursive: true)
        .whereType<File>()
        .where((f) => f.path.endsWith('.part'))
        .fold<int>(0, (a, f) => a + f.lengthSync());
  }

  /// Deletes model folders that are no longer used (e.g. after an upgrade).
  void removeExcept(Set<String> keepIds) {
    if (!root.existsSync()) return;
    for (final d in root.listSync().whereType<Directory>()) {
      if (!keepIds.contains(p.basename(d.path))) d.deleteSync(recursive: true);
    }
  }
}
