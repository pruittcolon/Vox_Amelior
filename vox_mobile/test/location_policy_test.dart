import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';

void main() {
  const home = Place(id: 'home', name: 'Home', lat: 51.5007, lon: -0.1246, radiusM: 150);
  const work = Place(id: 'work', name: 'Work', lat: 51.5155, lon: -0.0922, radiusM: 200);
  const policy = LocationPolicy();

  // ~111 m per 0.001° latitude.
  double north(double metres) => home.lat + metres / 111320;

  test('distance is accurate to a few metres', () {
    expect(distanceMeters(home.lat, home.lon, north(100), home.lon), closeTo(100, 1));
    expect(distanceMeters(home.lat, home.lon, work.lat, work.lon), closeTo(2781, 5));
  });

  test('off, or no places, always listens', () {
    expect(policy.decide(mode: LocationMode.off, places: const [home], lat: 0, lon: 0).listen, isTrue);
    expect(policy.decide(mode: LocationMode.onlyAtPlaces, places: const [], lat: 0, lon: 0).listen, isTrue);
  });

  test('only at places: listen at home, pause elsewhere', () {
    final atHome = policy.decide(mode: LocationMode.onlyAtPlaces, places: const [home, work], lat: north(50), lon: home.lon);
    expect(atHome.listen, isTrue);
    expect(atHome.place?.name, 'Home');
    expect(atHome.reason, 'At Home');
    final away = policy.decide(mode: LocationMode.onlyAtPlaces, places: const [home, work], lat: north(1000), lon: home.lon);
    expect(away.listen, isFalse);
    expect(away.place, isNull);
  });

  test('pause at places: pause at work, listen elsewhere', () {
    expect(policy.decide(mode: LocationMode.pauseAtPlaces, places: const [work], lat: work.lat, lon: work.lon).listen, isFalse);
    expect(policy.decide(mode: LocationMode.pauseAtPlaces, places: const [work], lat: home.lat, lon: home.lon).listen, isTrue);
  });

  test('hysteresis: GPS jitter at the edge does not flip the decision', () {
    // 180 m out: outside the 150 m radius when arriving...
    final arriving = policy.decide(mode: LocationMode.onlyAtPlaces, places: const [home], lat: north(180), lon: home.lon);
    expect(arriving.listen, isFalse);
    // ...but still "home" when we were already there (radius + 60 m margin).
    final staying = policy.decide(
      mode: LocationMode.onlyAtPlaces,
      places: const [home],
      lat: north(180),
      lon: home.lon,
      currentPlaceId: 'home',
    );
    expect(staying.listen, isTrue);
    // Clearly gone.
    final left = policy.decide(mode: LocationMode.onlyAtPlaces, places: const [home], lat: north(400), lon: home.lon, currentPlaceId: 'home');
    expect(left.listen, isFalse);
  });

  test('places survive JSON and reject junk', () {
    final back = Place.fromJson(home.toJson())!;
    expect(back.name, 'Home');
    expect(back.lat, home.lat);
    expect(Place.fromJson({'lat': 'x'}), isNull);
    expect(Place.fromJson({'lat': 1, 'lon': 2, 'radius': 999999})!.radiusM, 5000);
  });
}
