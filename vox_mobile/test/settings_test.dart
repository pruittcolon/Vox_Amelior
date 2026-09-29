import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';

void main() {
  group('AppSettings', () {
    test('defaults are sensible', () {
      const s = AppSettings();
      expect(s.wakePhrases, contains('hey vox'));
      expect(s.retentionDays, 90);
      expect(s.gemmaAsset.id, ModelCatalog.gemma3nE4b.id);
    });

    test('JSON round trip keeps every field', () {
      final s = const AppSettings().copyWith(
        wakePhrases: ['computer'],
        matchThreshold: 0.7,
        retentionDays: 30,
        speakReplies: false,
        gemmaModelId: ModelCatalog.gemma3nE2b.id,
        gemmaUrlOverride: 'https://mirror.example/gemma.litertlm',
      );
      final back = AppSettings.fromJson(s.toJson());
      expect(back.wakePhrases, ['computer']);
      expect(back.matchThreshold, 0.7);
      expect(back.retentionDays, 30);
      expect(back.speakReplies, isFalse);
      expect(back.gemmaAsset.id, ModelCatalog.gemma3nE2b.id);
      expect(back.gemmaAsset.files.single.url, 'https://mirror.example/gemma.litertlm');
    });

    test('damaged or out-of-range values fall back safely', () {
      final s = AppSettings.fromJson({
        'wakePhrases': [1, '', '  '],
        'matchThreshold': 5,
        'matchMargin': -1,
        'retentionDays': -4,
        'speakReplies': 'yes',
        'gemmaModelId': 'does-not-exist',
      });
      expect(s.wakePhrases, const AppSettings().wakePhrases);
      expect(s.matchThreshold, 0.95);
      expect(s.matchMargin, 0);
      expect(s.retentionDays, 90);
      expect(s.speakReplies, isTrue);
      expect(s.gemmaAsset.id, ModelCatalog.gemma3nE4b.id);
    });

    test('a blank URL override is ignored, and can be cleared', () {
      expect(const AppSettings(gemmaUrlOverride: '  ').gemmaAsset.files.single.url, ModelCatalog.gemma3nE4b.files.single.url);
      final cleared = const AppSettings(gemmaUrlOverride: 'https://x.io/a').copyWith(clearGemmaUrl: true);
      expect(cleared.gemmaUrlOverride, isNull);
    });
  });

  group('SettingsRepository', () {
    test('saves and loads; corrupted storage yields defaults', () async {
      SharedPreferences.setMockInitialValues({});
      final repo = SettingsRepository();
      expect((await repo.load()).retentionDays, 90);
      await repo.save(const AppSettings(retentionDays: 7));
      expect((await repo.load()).retentionDays, 7);

      SharedPreferences.setMockInitialValues({'app_settings_v1': '{not json'});
      expect((await SettingsRepository().load()).retentionDays, 90);
    });
  });

  group('ServiceConfig', () {
    late Directory tmp;
    setUp(() => tmp = Directory.systemTemp.createTempSync('vox_cfg_'));
    tearDown(() => tmp.deleteSync(recursive: true));

    test('speech paths are only available when every model is installed', () {
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      expect(SpeechModelPaths.fromStore(store), isNull);

      for (final asset in [ModelCatalog.parakeet, ModelCatalog.voiceActivity, ModelCatalog.speakerVoiceprint]) {
        store.dir(asset).createSync(recursive: true);
        for (final name in asset.installedFileNames) {
          store.file(asset, name).writeAsBytesSync([1, 2, 3]);
        }
        store.markInstalled(asset);
      }
      final paths = SpeechModelPaths.fromStore(store)!;
      expect(p.basename(paths.encoder), 'encoder.int8.onnx');
      expect(File(paths.tokens).existsSync(), isTrue);
    });

    test('config survives a disk round trip and tolerates a missing or corrupt file', () {
      const cfg = ServiceConfig(
        dbPath: '/data/vox.db',
        paths: SpeechModelPaths(encoder: 'e', decoder: 'd', joiner: 'j', tokens: 't', vad: 'v', speaker: 's'),
        settings: AppSettings(retentionDays: 14),
      );
      final file = File(p.join(tmp.path, 'sub', 'service_config.json'));
      expect(ServiceConfig.readFrom(file), isNull);
      cfg.writeTo(file);
      final back = ServiceConfig.readFrom(file)!;
      expect(back.dbPath, '/data/vox.db');
      expect(back.paths.speaker, 's');
      expect(back.settings.retentionDays, 14);
      file.writeAsStringSync('garbage');
      expect(ServiceConfig.readFrom(file), isNull);
    });
  });
}
