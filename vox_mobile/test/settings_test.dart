import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/models/model_store.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';

void main() {
  group('AppSettings', () {
    test('defaults: Gemma 4 E4B with agent mode, location off', () {
      const s = AppSettings();
      expect(s.wakePhrases, isEmpty, reason: 'no wake word unless one is added');
      expect(AppSettings.fromJson({'wakePhrases': ['hey vox', 'ok vox', 'vox']}).wakePhrases, isEmpty);
      expect(AppSettings.fromJson({'wakePhrases': ['computer']}).wakePhrases, ['computer']);
      expect(s.llmAsset.id, ModelCatalog.gemma4E4b.id);
      expect(s.agentMode, isTrue);
      expect(s.locationMode, LocationMode.off);
    });

    test('JSON round trip keeps every field', () {
      final s = const AppSettings().copyWith(
        wakePhrases: ['computer'],
        matchThreshold: 0.7,
        retentionDays: 30,
        speakReplies: false,
        llmId: ModelCatalog.customLlmId,
        customLlmUrl: 'https://huggingface.co/x/y/resolve/main/model.litertlm',
        customLlmName: 'Mine',
        customLlmType: 'qwen3',
        customLlmTools: false,
        agentMode: false,
        instructions: 'Be brief.',
        locationMode: LocationMode.onlyAtPlaces,
        places: const [Place(id: 'h', name: 'Home', lat: 51.5, lon: -0.1, radiusM: 200)],
      );
      final back = AppSettings.fromJson(s.toJson());
      expect(back.wakePhrases, ['computer']);
      expect(back.retentionDays, 30);
      expect(back.speakReplies, isFalse);
      expect(back.agentMode, isFalse);
      expect(back.instructions, 'Be brief.');
      expect(back.locationMode, LocationMode.onlyAtPlaces);
      expect(back.places.single.name, 'Home');
      expect(back.places.single.radiusM, 200);
      final asset = back.llmAsset;
      expect(asset.id, ModelCatalog.customLlmId);
      expect(asset.title, 'Mine');
      expect(asset.llmType, 'qwen3');
      expect(asset.supportsTools, isFalse);
      expect(asset.files.single.fileName, 'model.litertlm');
    });

    test('custom model without a URL falls back to Gemma 4', () {
      expect(const AppSettings(llmId: ModelCatalog.customLlmId).llmAsset.id, ModelCatalog.gemma4E4b.id);
      expect(const AppSettings(llmId: 'gemma-4-e2b-it').llmAsset.id, ModelCatalog.gemma4E2b.id);
      expect(const AppSettings(llmId: 'nope').llmAsset.id, ModelCatalog.gemma4E4b.id);
    });

    test('damaged or out-of-range values fall back safely', () {
      final s = AppSettings.fromJson({
        'wakePhrases': [1, '', '  '],
        'matchThreshold': 5,
        'matchMargin': -1,
        'retentionDays': -4,
        'speakReplies': 'yes',
        'llmId': 42,
        'locationMode': 'sideways',
        'places': [
          {'lat': 'x'},
          {'lat': 1, 'lon': 2},
        ],
      });
      expect(s.wakePhrases, const AppSettings().wakePhrases);
      expect(s.matchThreshold, 0.95);
      expect(s.matchMargin, 0);
      expect(s.retentionDays, 90);
      expect(s.speakReplies, isTrue);
      expect(s.llmAsset.id, ModelCatalog.gemma4E4b.id);
      expect(s.locationMode, LocationMode.off);
      expect(s.places.length, 1);
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

    void fakeInstall(ModelStore store, ModelAsset asset) {
      store.dir(asset).createSync(recursive: true);
      for (final name in asset.installedFileNames) {
        store.file(asset, name).writeAsBytesSync([1, 2, 3]);
      }
      store.markInstalled(asset);
    }

    test('speech paths need every speech model; the assistant is optional', () {
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      expect(SpeechModelPaths.fromStore(store), isNull);
      for (final a in ModelCatalog.speech) {
        fakeInstall(store, a);
      }
      final paths = SpeechModelPaths.fromStore(store)!;
      expect(p.basename(paths.encoder), 'encoder.fp16.onnx', reason: 'fp16 is the default download');
      for (final n in ['encoder.fp16.onnx', 'decoder.fp16.onnx', 'joiner.fp16.onnx', 'tokens.txt']) {
        expect(File(p.join(p.dirname(paths.encoder), n)).existsSync(), isTrue, reason: n);
      }
      expect(LlmConfig.fromStore(store, ModelCatalog.gemma4E4b), isNull);
      fakeInstall(store, ModelCatalog.gemma4E4b);
      final llm = LlmConfig.fromStore(store, ModelCatalog.gemma4E4b)!;
      expect(llm.modelType, 'gemma4');
      expect(llm.supportsTools, isTrue);
    });

    test('upgrading from the 1.1B model removes its folder and asks for the speech model again', () {
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      for (final a in ModelCatalog.speech) {
        if (a.kind != ModelKind.speechToText) fakeInstall(store, a);
      }
      // What an earlier version left behind: the 1.1B model, fully installed.
      final old = Directory(p.join(store.root.path, 'parakeet-rnnt-1.1b-int8'))..createSync(recursive: true);
      File(p.join(old.path, 'encoder.int8.weights')).writeAsBytesSync(List.filled(1000, 7));
      File(p.join(old.path, 'encoder.int8.onnx')).writeAsBytesSync([1]);
      expect(SpeechModelPaths.fromStore(store), isNull);

      // Same call as AppServices._init.
      store.removeExcept({for (final m in ModelCatalog.all) m.id, ModelCatalog.customLlmId});

      expect(old.existsSync(), isFalse, reason: 'about 1.1 GB freed');
      expect(store.isInstalled(ModelCatalog.voiceActivity), isTrue, reason: 'other models are kept');
      expect(store.isInstalled(ModelCatalog.speakerVoiceprint), isTrue);
      expect(store.isInstalled(ModelCatalog.parakeetFp16), isFalse);
      expect(SpeechModelPaths.fromStore(store), isNull, reason: 'setup screen offers the 0.6B download');
      fakeInstall(store, ModelCatalog.parakeetFp16);
      expect(SpeechModelPaths.fromStore(store), isNotNull);
    });

    test('missing model files are found before any native code loads them', () {
      final store = ModelStore(Directory(p.join(tmp.path, 'models')));
      for (final a in ModelCatalog.speech) {
        fakeInstall(store, a);
      }
      final paths = SpeechModelPaths.fromStore(store)!;
      expect(paths.missingFiles(), isEmpty);
      File(paths.encoder).deleteSync();
      expect(paths.missingFiles(), [paths.encoder]);
    });

    test('config survives a disk round trip and tolerates a missing or corrupt file', () {
      const cfg = ServiceConfig(
        dbPath: '/data/vox.db',
        paths: SpeechModelPaths(encoder: 'e', decoder: 'd', joiner: 'j', tokens: 't', vad: 'v', speaker: 's'),
        settings: AppSettings(retentionDays: 14),
        queueDir: '/data/q',
        llm: LlmConfig(path: '/m/g.litertlm', modelType: 'gemma4', supportsTools: true, title: 'Gemma 4 E4B'),
      );
      final file = File(p.join(tmp.path, 'sub', 'service_config.json'));
      expect(ServiceConfig.readFrom(file), isNull);
      cfg.writeTo(file);
      final back = ServiceConfig.readFrom(file)!;
      expect(back.dbPath, '/data/vox.db');
      expect(back.queueDir, '/data/q');
      expect(back.llm!.path, '/m/g.litertlm');
      expect(back.settings.retentionDays, 14);
      file.writeAsStringSync('garbage');
      expect(ServiceConfig.readFrom(file), isNull);
    });
  });
}
