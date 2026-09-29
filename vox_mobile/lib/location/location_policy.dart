import 'dart:math' as math;

/// A saved place, such as home.
class Place {
  const Place({required this.id, required this.name, required this.lat, required this.lon, this.radiusM = 150});

  final String id;
  final String name;
  final double lat;
  final double lon;

  /// How close counts as "at this place".
  final double radiusM;

  Place copyWith({String? name, double? radiusM}) =>
      Place(id: id, name: name ?? this.name, lat: lat, lon: lon, radiusM: radiusM ?? this.radiusM);

  Map<String, Object?> toJson() => {'id': id, 'name': name, 'lat': lat, 'lon': lon, 'radius': radiusM};

  static Place? fromJson(Object? j) {
    if (j is! Map) return null;
    final lat = j['lat'];
    final lon = j['lon'];
    if (lat is! num || lon is! num) return null;
    return Place(
      id: '${j['id'] ?? '$lat,$lon'}',
      name: '${j['name'] ?? 'Place'}',
      lat: lat.toDouble(),
      lon: lon.toDouble(),
      radiusM: (j['radius'] as num?)?.toDouble().clamp(30, 5000) ?? 150,
    );
  }
}

enum LocationMode {
  /// Location is not used.
  off,

  /// Listen only while at one of the saved places (e.g. home).
  onlyAtPlaces,

  /// Pause while at one of the saved places (e.g. work, doctor).
  pauseAtPlaces,
}

class LocationDecision {
  const LocationDecision({required this.listen, required this.reason, this.place});

  final bool listen;
  final String reason;

  /// The place the phone is at, if any.
  final Place? place;
}

/// Decides whether to listen based on where the phone is.
///
/// Leaving a place requires moving [exitMarginM] (plus the fix's accuracy)
/// beyond its radius, so GPS jitter at the edge does not toggle listening.
class LocationPolicy {
  const LocationPolicy({this.exitMarginM = 60});

  final double exitMarginM;

  LocationDecision decide({
    required LocationMode mode,
    required List<Place> places,
    required double lat,
    required double lon,
    double accuracyM = 0,
    String? currentPlaceId,
  }) {
    if (mode == LocationMode.off || places.isEmpty) {
      return const LocationDecision(listen: true, reason: 'Location rules are off');
    }
    Place? at;
    for (final p in places) {
      final d = distanceMeters(lat, lon, p.lat, p.lon);
      final stayingHere = p.id == currentPlaceId;
      final limit = stayingHere ? p.radiusM + exitMarginM + accuracyM.clamp(0, 200) : p.radiusM;
      if (d <= limit) {
        at = p;
        if (stayingHere) break;
      }
    }
    return switch (mode) {
      LocationMode.onlyAtPlaces => at != null
          ? LocationDecision(listen: true, reason: 'At ${at.name}', place: at)
          : const LocationDecision(listen: false, reason: 'Away from your places'),
      LocationMode.pauseAtPlaces => at != null
          ? LocationDecision(listen: false, reason: 'At ${at.name}', place: at)
          : const LocationDecision(listen: true, reason: 'Not at a paused place'),
      LocationMode.off => const LocationDecision(listen: true, reason: 'Location rules are off'),
    };
  }
}

/// Great-circle distance in metres (haversine).
double distanceMeters(double lat1, double lon1, double lat2, double lon2) {
  const r = 6371000.0;
  double rad(double d) => d * math.pi / 180;
  final dLat = rad(lat2 - lat1);
  final dLon = rad(lon2 - lon1);
  final a = math.pow(math.sin(dLat / 2), 2) + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.pow(math.sin(dLon / 2), 2);
  return 2 * r * math.asin(math.sqrt(a.toDouble()));
}
