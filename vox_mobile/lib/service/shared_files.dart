import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:vox_amelior_mobile/assistant/context_probe.dart';
import 'package:vox_amelior_mobile/core/log.dart';

// Files the app and the listening service both use. Both isolates derive
// the paths from the app's support directory the same way.

/// Where saved voice clips live (both isolates derive it the same way).
String voiceClipsDir(String supportDir) => p.join(supportDir, 'voice_clips');

String probeMarkerPath(String supportDir) => p.join(supportDir, 'context_probe.json');

String probeResultPath(String supportDir) => p.join(supportDir, 'context_probe_result.json');

void writeProbeResult(String supportDir, ProbeResult result) {
  try {
    File(probeResultPath(supportDir)).writeAsStringSync(jsonEncode(result.toJson()), flush: true);
  } on Object catch (e) {
    Log.w('probe', 'could not save the test result', e);
  }
}

/// A test result the app has not picked up yet (then deleted).
ProbeResult? takeProbeResult(String supportDir) {
  final f = File(probeResultPath(supportDir));
  if (!f.existsSync()) return null;
  try {
    return ProbeResult.fromJson(jsonDecode(f.readAsStringSync()));
  } on Object {
    return null;
  } finally {
    f.deleteSync();
  }
}
