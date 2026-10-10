import 'package:integration_test/integration_test_driver_extended.dart';

/// Runs the on-device tour (integration_test/app_tour_test.dart) and saves
/// its frame timings to build/tour/tour_perf.json. Screenshots are taken by
/// tool/run_tour.sh with adb, so the app renders exactly as on a phone.
Future<void> main() => integrationDriver(
      writeResponseOnFailure: true,
      responseDataCallback: (data) => writeResponseData(data, testOutputFilename: 'tour_perf', destinationDirectory: 'build/tour'),
    );
