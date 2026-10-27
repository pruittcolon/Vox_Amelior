import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../ble/proto.dart';
import '../bridge/bridge_client.dart';
import '../log.dart';
import 'glasses_text.dart';

/// Current weather for the look-up screen.
class Weather {
  final int tempC;
  final int icon; // G1 weather icon
  final String place;
  final DateTime at;
  const Weather(this.tempC, this.icon, this.place, this.at);
}

/// G1 weather icons (as the community G1 apps number them).
class G1Icon {
  static const none = 0x00, night = 0x01, clouds = 0x02, drizzle = 0x03, heavyDrizzle = 0x04, rain = 0x05,
      heavyRain = 0x06, thunder = 0x07, thunderstorm = 0x08, snow = 0x09, mist = 0x0A, fog = 0x0B,
      freezingRain = 0x0F, sunny = 0x10;

  /// WMO weather code (Open-Meteo) to a G1 icon.
  static int fromWmo(int code, {bool day = true}) {
    if (code == 0 || code == 1) return day ? sunny : night;
    if (code == 2 || code == 3) return clouds;
    if (code == 45 || code == 48) return fog;
    if (code == 51 || code == 53) return drizzle;
    if (code == 55) return heavyDrizzle;
    if (code == 56 || code == 57 || code == 66 || code == 67) return freezingRain;
    if (code == 61 || code == 63 || code == 80 || code == 81) return rain;
    if (code == 65 || code == 82) return heavyRain;
    if ((code >= 71 && code <= 77) || code == 85 || code == 86) return snow;
    if (code == 95) return thunder;
    if (code == 96 || code == 99) return thunderstorm;
    return none;
  }
}

/// Weather from Open-Meteo (free, no account or API key).
class WeatherService {
  WeatherService({this.location});

  /// The phone's last known position (lat, lon), if allowed.
  final Future<(double, double)?> Function()? location;

  final HttpClient _http = HttpClient()..connectionTimeout = const Duration(seconds: 6);

  Future<Map<String, dynamic>> _get(String url) async {
    final req = await _http.getUrl(Uri.parse(url)).timeout(const Duration(seconds: 8));
    final res = await req.close().timeout(const Duration(seconds: 8));
    final body = await res.transform(utf8.decoder).join().timeout(const Duration(seconds: 8));
    if (res.statusCode != 200) throw HttpException('weather ${res.statusCode}');
    final data = jsonDecode(body);
    return data is Map<String, dynamic> ? data : <String, dynamic>{};
  }

  /// [city] '' = the phone's location.
  Future<Weather?> now(String city) async {
    double lat, lon;
    var place = city;
    if (city.trim().isEmpty) {
      final here = await location?.call();
      if (here == null) return null;
      (lat, lon) = here;
      place = 'Here';
    } else {
      final g = await _get('https://geocoding-api.open-meteo.com/v1/search?count=1&name=${Uri.encodeQueryComponent(city.trim())}');
      final results = g['results'] as List? ?? const [];
      if (results.isEmpty) return null;
      final r = results.first as Map;
      lat = (r['latitude'] as num).toDouble();
      lon = (r['longitude'] as num).toDouble();
      place = '${r['name'] ?? city}';
    }
    final w = await _get('https://api.open-meteo.com/v1/forecast?latitude=$lat&longitude=$lon'
        '&current=temperature_2m,weather_code,is_day');
    final c = w['current'] as Map? ?? const {};
    final temp = (c['temperature_2m'] as num?)?.round() ?? 0;
    final code = (c['weather_code'] as num?)?.toInt() ?? -1;
    return Weather(temp, G1Icon.fromWmo(code, day: c['is_day'] != 0), place, DateTime.now());
  }

  void close() => _http.close(force: true);
}

/// The look-up screen (the G1 dashboard): clock and weather, layout, and up
/// to four notes you choose - Claude's finished jobs, the last answer, phone
/// notifications, or your own text. Kept fresh on connect, every few
/// minutes, whenever a note's source changes, and when you look up.
class Dashboard extends ChangeNotifier {
  Dashboard(this.proto, {this.weatherService});

  final Proto proto;
  final WeatherService? weatherService;

  /// Something else owns the glasses for now (screen sharing, Flicker).
  bool Function() paused = () => false;

  // Settings (set by the app)
  bool enabled = true;
  int mode = 1; // 0 full, 1 dual, 2 minimal
  int pane = 0; // 0 notes, 1 stocks, 2 news, 3 calendar, 4 map
  bool h24 = false;
  bool fahrenheit = true;
  bool weatherOn = true;
  String city = '';
  List<String> slots = const ['work', 'reply', 'custom', 'off'];
  String custom = '';

  // Note sources
  WorkItem? lastWork;
  String lastReply = '';
  final List<String> phone = [];

  // State
  Weather? weather;
  DateTime? lastSync;
  String? error;
  bool syncing = false;

  final Map<int, String> _sentNotes = {};
  String _sentLayout = '';
  Timer? _debounce;
  Timer? _timer;
  bool _again = false;

  bool get _connected => proto.ble.isConnected;

  /// Glasses connected: send everything once, then keep the clock fresh.
  void onConnected() {
    _sentNotes.clear();
    _sentLayout = '';
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(minutes: 10), (_) => sync());
    // Notes live in the glasses' memory: rewrite all four once per connection.
    _debounce?.cancel();
    _debounce = Timer(const Duration(seconds: 2), () => sync(force: true));
  }

  void onDisconnected() {
    _timer?.cancel();
    _timer = null;
  }

  /// You looked up: refresh if it has been a while.
  void onHeadUp() {
    final last = lastSync;
    if (last == null || DateTime.now().difference(last) > const Duration(minutes: 2)) sync();
  }

  /// Something a note shows changed: send it in a moment.
  void markDirty({Duration delay = const Duration(seconds: 2)}) {
    _debounce?.cancel();
    _debounce = Timer(delay, sync);
  }

  void addPhoneNote(String line) {
    phone.insert(0, line);
    if (phone.length > 4) phone.removeLast();
    if (slots.contains('phone')) markDirty();
  }

  /// The four notes as (title, text); empty ones are deleted on the glasses.
  List<(String, String)> buildNotes() {
    String clock(DateTime t) {
      final h = h24 ? t.hour : (t.hour % 12 == 0 ? 12 : t.hour % 12);
      final m = t.minute.toString().padLeft(2, '0');
      return h24 ? '${h.toString().padLeft(2, '0')}:$m' : '$h:$m${t.hour < 12 ? 'am' : 'pm'}';
    }

    (String, String) note(String source) {
      switch (source) {
        case 'work':
          final w = lastWork;
          if (w == null) return ('Claude', 'No finished jobs yet.');
          return ('Claude done ${clock(w.time)}', '${w.summary}\n${w.project}');
        case 'reply':
          if (lastReply.trim().isEmpty) return ('', '');
          return ('Last answer', lastReply);
        case 'phone':
          if (phone.isEmpty) return ('', '');
          return ('Phone', phone.join('\n'));
        case 'custom':
          final lines = custom.split('\n');
          final title = lines.first.trim();
          final text = lines.skip(1).join('\n').trim();
          if (title.isEmpty && text.isEmpty) return ('', '');
          return (title.isEmpty ? 'Note' : title, text);
        default:
          return ('', '');
      }
    }

    final out = <(String, String)>[];
    for (final s in slots.take(4)) {
      final (t, x) = note(s);
      out.add((GlassesText.clean(t), GlassesText.clean(x)));
    }
    return out;
  }

  /// Send what changed (everything with [force]).
  Future<void> sync({bool force = false}) async {
    if (!enabled || !_connected || paused()) return;
    if (syncing) {
      _again = true;
      return;
    }
    syncing = true;
    notifyListeners();
    try {
      do {
        _again = false;
        await _syncOnce(force);
      } while (_again && _connected);
      error = null;
    } catch (e) {
      error = '$e';
    } finally {
      syncing = false;
      notifyListeners();
    }
  }

  Future<void> _syncOnce(bool force) async {
    if (weatherOn && weatherService != null) {
      final w = weather;
      if (w == null || DateTime.now().difference(w.at) > const Duration(minutes: 30) || force) {
        try {
          weather = await weatherService!.now(city) ?? weather;
        } catch (e) {
          EventLog.add('dash', 'weather', {'error': '$e'});
        }
      }
    }
    final w = weatherOn ? weather : null;
    await proto.sendTimeWeather(DateTime.now(),
        icon: w?.icon ?? 0, tempC: w?.tempC ?? 0, fahrenheit: fahrenheit, h24: h24);
    final layout = '$mode/$pane';
    if (force || layout != _sentLayout) {
      if (await proto.sendDashboardMode(mode, pane)) _sentLayout = layout;
    }
    final notes = buildNotes();
    for (var i = 0; i < 4; i++) {
      final (title, text) = i < notes.length ? notes[i] : ('', '');
      final key = '$title\n$text';
      if (!force && _sentNotes[i + 1] == key) continue;
      if (title.isEmpty && text.isEmpty && !_sentNotes.containsKey(i + 1) && !force) continue;
      if (await proto.sendNote(i + 1, title, text)) _sentNotes[i + 1] = key;
    }
    lastSync = DateTime.now();
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _timer?.cancel();
    weatherService?.close();
    super.dispose();
  }
}
