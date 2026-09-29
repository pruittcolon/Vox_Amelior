import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Keeps the Hugging Face access token in Android's encrypted storage.
class TokenStore {
  TokenStore([FlutterSecureStorage? storage]) : _storage = storage ?? const FlutterSecureStorage();

  static const _hfKey = 'huggingface_token';
  final FlutterSecureStorage _storage;

  Future<String?> huggingFace() async {
    final v = await _storage.read(key: _hfKey);
    return (v == null || v.trim().isEmpty) ? null : v.trim();
  }

  Future<void> saveHuggingFace(String token) => _storage.write(key: _hfKey, value: token.trim());

  Future<void> clearHuggingFace() => _storage.delete(key: _hfKey);
}
