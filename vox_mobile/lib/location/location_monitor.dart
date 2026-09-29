import 'dart:async';

import 'package:geolocator/geolocator.dart';
import 'package:vox_amelior_mobile/core/log.dart';

class LocationFix {
  const LocationFix(this.lat, this.lon, this.accuracyM, this.at);
  final double lat;
  final double lon;
  final double accuracyM;
  final DateTime at;
}

enum LocationAccess { granted, denied, serviceOff }

/// Reads the phone's position without draining the battery: network/Wi-Fi
/// accuracy, and the cached fix when it is recent enough.
class LocationMonitor {
  const LocationMonitor({this.maxCachedAge = const Duration(minutes: 3)});

  final Duration maxCachedAge;

  static Future<LocationAccess> access() async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) return LocationAccess.serviceOff;
      final p = await Geolocator.checkPermission();
      return (p == LocationPermission.always || p == LocationPermission.whileInUse)
          ? LocationAccess.granted
          : LocationAccess.denied;
    } on Object {
      return LocationAccess.denied;
    }
  }

  /// Asks the user for location permission (needs a visible screen).
  static Future<bool> requestPermission() async {
    try {
      var p = await Geolocator.checkPermission();
      if (p == LocationPermission.denied) p = await Geolocator.requestPermission();
      return p == LocationPermission.always || p == LocationPermission.whileInUse;
    } on Object catch (e) {
      Log.w('location', 'permission request failed', e);
      return false;
    }
  }

  Future<LocationFix?> current() async {
    try {
      final last = await Geolocator.getLastKnownPosition();
      if (last != null && DateTime.now().difference(last.timestamp) <= maxCachedAge) {
        return LocationFix(last.latitude, last.longitude, last.accuracy, last.timestamp);
      }
      final p = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.medium, timeLimit: Duration(seconds: 30)),
      );
      return LocationFix(p.latitude, p.longitude, p.accuracy, p.timestamp);
    } on Object catch (e) {
      Log.w('location', 'could not get position', e.runtimeType);
      return null;
    }
  }
}
