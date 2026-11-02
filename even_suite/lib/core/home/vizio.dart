import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// One SmartCast call: (method, url, headers, JSON body) -> decoded answer.
typedef VizioCall = Future<Map<String, dynamic>> Function(
    String method, Uri url, Map<String, String> headers, Object? body);

/// A Vizio SmartCast TV on your Wi-Fi (HTTPS on port 7345, its own
/// certificate), the way the `tv` remote in Termux drives it: remote keys,
/// inputs, and pairing (the TV shows a PIN once; the token it gives back is
/// kept on the phone).
class VizioTv {
  VizioTv({this.call, this.discoverAddresses});

  /// For tests: replaces the HTTPS calls.
  final VizioCall? call;

  /// For tests: replaces the Wi-Fi search (SSDP) - addresses that answered.
  final Future<List<String>> Function()? discoverAddresses;

  /// Remote keys: (codeset, code).
  static const keyCodes = {
    'UP': (3, 8), 'DOWN': (3, 0), 'LEFT': (3, 1), 'RIGHT': (3, 7), 'OK': (3, 2),
    'BACK': (4, 0), 'HOME': (4, 15), 'MENU': (4, 8), 'INFO': (4, 6), 'EXIT': (9, 0),
    'VOL_UP': (5, 1), 'VOL_DOWN': (5, 0), 'MUTE': (5, 4),
    'PLAY': (2, 3), 'PAUSE': (2, 2), 'FWD': (2, 0), 'REW': (2, 1),
    'INPUT': (7, 1), 'CH_UP': (8, 1), 'CH_DOWN': (8, 0),
    'POWER': (11, 2), 'POWER_OFF': (11, 0), 'POWER_ON': (11, 1),
  };

  static const port = 7345;

  String _pairDevice = '';
  String _pairToken = '';

  Uri _url(String ip, String path) => Uri(scheme: 'https', host: ip, port: port, path: path);

  Future<Map<String, dynamic>> _send(String ip, String method, String path,
      {Object? body, String auth = '', Duration timeout = const Duration(seconds: 6)}) {
    final headers = {'Content-Type': 'application/json', if (auth.isNotEmpty) 'AUTH': auth};
    return (call ?? _https)(method, _url(ip, path), headers, body).timeout(timeout);
  }

  static String result(Map<String, dynamic> d) => '${(d['STATUS'] as Map?)?['RESULT'] ?? ''}';

  static void _ok(Map<String, dynamic> d) {
    final r = result(d);
    if (r != 'SUCCESS') throw VizioException(r == 'REQUIRES_PAIRING' ? 'Pair the TV first' : 'TV said $r');
  }

  /// Press [names] in order ([keyCodes]).
  Future<void> keys(String ip, String auth, List<String> names) async {
    for (final n in names) {
      final (cs, code) = keyCodes[n] ?? (throw VizioException('No key $n'));
      _ok(await _send(ip, 'PUT', '/key_command/', auth: auth, body: {
        'KEYLIST': [
          {'CODESET': cs, 'CODE': code, 'ACTION': 'KEYPRESS'}
        ]
      }));
    }
  }

  /// Switch to an input: hdmi1, hdmi2, hdmi3, comp, tuner.
  Future<void> setInput(String ip, String auth, String cname) async {
    const menu = '/menu_native/dynamic/tv_settings/devices/current_input';
    final d = await _send(ip, 'GET', menu, auth: auth);
    _ok(d);
    final item = (d['ITEMS'] as List? ?? const []).cast<Map>().firstWhere((i) => i['CNAME'] == 'current_input',
        orElse: () => throw VizioException('The TV did not list its input'));
    _ok(await _send(ip, 'PUT', menu,
        auth: auth, body: {'REQUEST': 'MODIFY', 'HASHVAL': item['HASHVAL'], 'VALUE': cname}));
  }

  /// On, off, or null when it cannot be told (not paired, not reachable).
  Future<bool?> power(String ip, String auth) async {
    try {
      final d = await _send(ip, 'GET', '/state/device/power_mode', auth: auth, timeout: const Duration(seconds: 4));
      if (result(d) != 'SUCCESS') return null;
      return ((d['ITEMS'] as List).first as Map)['VALUE'] == 1;
    } catch (_) {
      return null;
    }
  }

  /// Its name ("Wu-Tang-TV"), or null when [ip] is not a Vizio TV.
  Future<String?> name(String ip) async {
    try {
      final d = await _send(ip, 'GET', '/state/device/deviceinfo', timeout: const Duration(seconds: 3));
      final v = ((d['ITEMS'] as List).first as Map)['VALUE'] as Map;
      return '${v['CAST_NAME'] ?? v['MODEL_NAME'] ?? 'Vizio TV'}';
    } catch (_) {
      return null;
    }
  }

  /// Find a Vizio TV on the Wi-Fi: (address, name), or null.
  Future<(String, String)?> discover() async {
    final ips = await (discoverAddresses ?? _ssdp)();
    for (final ip in ips) {
      final n = await name(ip);
      if (n != null) return (ip, n);
    }
    return null;
  }

  /// Pairing, step 1: the TV shows a 4-digit PIN.
  Future<void> startPairing(String ip, String deviceId) async {
    final d = await _send(ip, 'PUT', '/pairing/start', body: {'DEVICE_ID': deviceId, 'DEVICE_NAME': 'Even Suite'});
    _ok(d);
    _pairDevice = deviceId;
    _pairToken = '${((d['ITEM'] as Map?) ?? const {})['PAIRING_REQ_TOKEN']}';
  }

  /// Pairing, step 2: the PIN from the TV. Returns the token to keep.
  Future<String> finishPairing(String ip, String pin) async {
    final d = await _send(ip, 'PUT', '/pairing/pair', body: {
      'DEVICE_ID': _pairDevice,
      'CHALLENGE_TYPE': 1,
      'RESPONSE_VALUE': pin.trim(),
      'PAIRING_REQ_TOKEN': int.tryParse(_pairToken) ?? _pairToken,
    });
    _ok(d);
    final token = '${((d['ITEM'] as Map?) ?? const {})['AUTH_TOKEN'] ?? ''}';
    if (token.isEmpty) throw VizioException('The TV gave no token');
    return token;
  }

  /// Close the PIN on the TV (it blocks the TV's own apps until closed).
  Future<void> cancelPairing(String ip) async {
    try {
      await _send(ip, 'PUT', '/pairing/cancel', body: {
        'DEVICE_ID': _pairDevice,
        'CHALLENGE_TYPE': 1,
        'RESPONSE_VALUE': '1111',
        'PAIRING_REQ_TOKEN': int.tryParse(_pairToken) ?? _pairToken,
      });
    } catch (_) {}
  }

  static Future<Map<String, dynamic>> _https(String method, Uri url, Map<String, String> headers, Object? body) async {
    // The TV's own certificate, on the TV's address only.
    final client = HttpClient()
      ..connectionTimeout = const Duration(seconds: 5)
      ..badCertificateCallback = (cert, host, p) => host == url.host && p == url.port;
    try {
      final req = await client.openUrl(method, url);
      headers.forEach(req.headers.set);
      if (body != null) req.write(jsonEncode(body));
      final res = await req.close();
      final text = await res.transform(utf8.decoder).join();
      final j = jsonDecode(text.isEmpty ? '{}' : text);
      return j is Map<String, dynamic> ? j : <String, dynamic>{};
    } finally {
      client.close();
    }
  }

  /// SSDP: every device on the Wi-Fi that answers an M-SEARCH.
  static Future<List<String>> _ssdp() async {
    final socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
    final found = <String>{};
    final done = Completer<void>();
    final sub = socket.listen((e) {
      if (e != RawSocketEvent.read) return;
      final d = socket.receive();
      if (d != null) found.add(d.address.address);
    });
    const msg = 'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\n'
        'MX: 2\r\nST: ssdp:all\r\n\r\n';
    for (var i = 0; i < 2; i++) {
      socket.send(utf8.encode(msg), InternetAddress('239.255.255.250'), 1900);
    }
    Timer(const Duration(seconds: 3), done.complete);
    await done.future;
    await sub.cancel();
    socket.close();
    return found.toList();
  }
}

class VizioException implements Exception {
  VizioException(this.message);
  final String message;
  @override
  String toString() => message;
}
