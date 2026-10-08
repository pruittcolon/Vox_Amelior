import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/settings/service_config.dart';

void main() {
  late Directory dir;
  late AppDatabase db;
  late AppServices s;

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    dir = Directory.systemTemp.createTempSync('vox_asr_choice_');
    db = AppDatabase.inMemory();
    s = AppServices.forTesting(dir: dir, db: db);
  });
  tearDown(() {
    db.close();
    dir.deleteSync(recursive: true);
  });

  void install(ModelAsset a) {
    s.models.dir(a).createSync(recursive: true);
    for (final n in a.installedFileNames) {
      s.models.file(a, n).writeAsBytesSync([1, 2, 3]);
    }
    s.models.markInstalled(a);
  }

  String encoderInConfig() => p.basename(ServiceConfig.readFrom(s.serviceConfigFile)!.paths.encoder);

  void installSpeechWith(List<ModelAsset> recognizers) {
    for (final a in [...ModelCatalog.speechSupport, ...recognizers]) {
      install(a);
    }
  }

  test('the service is never left a config pointing at models that are gone', () {
    for (final a in ModelCatalog.speech) {
      install(a);
    }
    expect(s.writeServiceConfig(), isTrue);
    expect(s.serviceConfigFile.existsSync(), isTrue);
    s.models.remove(ModelCatalog.parakeetFp16); // e.g. an upgrade replaced the speech model
    expect(s.writeServiceConfig(), isFalse);
    expect(s.serviceConfigFile.existsSync(), isFalse);
  });

  test('fp16 alone is enough to continue past setup, and int8 is never fetched unasked', () async {
    installSpeechWith([ModelCatalog.parakeetFp16]);
    expect(s.downloads.speechReady, isTrue);
    expect(s.speechReady, isTrue);
    expect(s.writeServiceConfig(), isTrue);
    expect(encoderInConfig(), 'encoder.fp16.onnx');
    await s.updateSettings(s.settings.value.copyWith(micGain: 2));
    expect(s.downloads.stateOf(ModelCatalog.parakeet).isBusy, isFalse);
    expect(s.models.dir(ModelCatalog.parakeet).existsSync(), isFalse);
  });

  test('int8 alone (fp16 deleted) also counts as ready', () {
    installSpeechWith([ModelCatalog.parakeet]);
    expect(s.downloads.speechReady, isTrue);
    expect(s.speechReady, isTrue, reason: 'falls back from the fp16 default');
    expect(s.writeServiceConfig(), isTrue);
    expect(encoderInConfig(), 'encoder.int8.onnx');
  });

  test('choosing int8 once it is installed points the service at it; fp16 goes back', () async {
    installSpeechWith(ModelCatalog.recognizers);
    await s.selectSpeechModel('int8');
    expect(s.settings.value.speechModel, 'int8');
    expect(encoderInConfig(), 'encoder.int8.onnx');
    await s.selectSpeechModel('fp16');
    expect(encoderInConfig(), 'encoder.fp16.onnx');
    expect(s.models.isInstalled(ModelCatalog.parakeet), isTrue, reason: 'kept, so switching back is instant');
  });

  test('deleting the model in use switches to the other one first', () async {
    installSpeechWith(ModelCatalog.recognizers);
    await s.selectSpeechModel('fp16');
    await s.removeSpeechModel(ModelCatalog.parakeetFp16);
    expect(s.settings.value.speechModel, 'int8');
    expect(s.models.dir(ModelCatalog.parakeetFp16).existsSync(), isFalse);
    expect(encoderInConfig(), 'encoder.int8.onnx');

    await s.selectSpeechModel('int8');
    install(ModelCatalog.parakeetFp16);
    await s.removeSpeechModel(ModelCatalog.parakeet);
    expect(s.settings.value.speechModel, 'fp16');
    expect(encoderInConfig(), 'encoder.fp16.onnx');
  });

  test('deleting fp16 from the Models screen also stops using it, and nothing re-downloads it', () async {
    installSpeechWith(ModelCatalog.recognizers);
    await s.selectSpeechModel('fp16');
    s.downloads.remove(ModelCatalog.parakeetFp16);
    await Future<void>.delayed(Duration.zero);
    expect(s.settings.value.speechModel, 'int8');
    expect(encoderInConfig(), 'encoder.int8.onnx');
    // An unrelated change afterwards must not start a 1.1 GB download.
    await s.updateSettings(s.settings.value.copyWith(micGain: 2));
    expect(s.downloads.stateOf(ModelCatalog.parakeetFp16).isBusy, isFalse);
  });

  test('deleting the only speech model leaves the choice alone and clears the service config', () async {
    installSpeechWith([ModelCatalog.parakeetFp16]);
    expect(s.writeServiceConfig(), isTrue);
    await s.removeSpeechModel(ModelCatalog.parakeetFp16);
    await Future<void>.delayed(Duration.zero);
    expect(s.settings.value.speechModel, 'fp16');
    expect(s.speechReady, isFalse);
    expect(s.serviceConfigFile.existsSync(), isFalse);
  });

  test('going back to int8 throws away a half-finished fp16 download', () async {
    installSpeechWith([ModelCatalog.parakeet]);
    final fp16 = ModelCatalog.parakeetFp16;
    s.models.dir(fp16).createSync(recursive: true);
    File(p.join(s.models.dir(fp16).path, 'parakeet-fp16.tar.bz2.part')).writeAsBytesSync(List.filled(1000, 1));
    await s.updateSettings(const AppSettings(speechModel: 'fp16'));
    await s.selectSpeechModel('int8');
    expect(s.models.dir(fp16).existsSync(), isFalse);
    expect(encoderInConfig(), 'encoder.int8.onnx');
  });
}
