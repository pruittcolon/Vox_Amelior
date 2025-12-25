import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:dio/io.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
// Phase 6 Security: Use encrypted storage instead of SharedPreferences
// SharedPreferences stores data in plaintext, flutter_secure_storage uses KeyStore/Keychain
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:demo_ai_even/services/app_logger.dart';

/// Authentication Service for WhisperServer backend
class AuthService {
  static final AuthService _instance = AuthService._();
  static AuthService get instance => _instance;

  AuthService._();

  late Dio _dio;
  String _serverBase = '';
  String? _sessionToken;
  String? _csrfToken;
  Map<String, dynamic>? _currentUser;
  
  final StreamController<bool> _authStateController = StreamController<bool>.broadcast();
  Stream<bool> get authStateStream => _authStateController.stream;
  
  bool get isAuthenticated => _sessionToken != null && _currentUser != null;
  Map<String, dynamic>? get currentUser => _currentUser;
  
  /// Get the session token for WebSocket authentication
  String? getSessionToken() => _sessionToken;
  
  /// Get auth headers for API requests
  Map<String, String> getAuthHeaders() {
    final headers = <String, String>{};
    final cookies = <String>[];
    
    if (_sessionToken != null) {
      cookies.add('ws_session=$_sessionToken');
    }
    if (_csrfToken != null) {
      cookies.add('ws_csrf=$_csrfToken');
      headers['X-CSRF-Token'] = _csrfToken!;  // CSRF header required for POST requests
    }
    if (cookies.isNotEmpty) {
      headers['Cookie'] = cookies.join('; ');
    }
    return headers;
  }

  void initialize() {
    // Server base URL - configure in .env file (WHISPER_SERVER_BASE)
    // Empty string will cause clear error if not configured
    _serverBase = dotenv.env['WHISPER_SERVER_BASE']?.trim() ?? '';
    
    _dio = Dio(
      BaseOptions(
        baseUrl: _serverBase,
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 30),
        sendTimeout: const Duration(seconds: 30),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      ),
    );

    // Bypass self-signed certificate errors for local development
    // ISO 27002 5.14: TLS bypass ONLY in debug mode - release builds use strict verification
    if (_serverBase.startsWith('https')) {
      (_dio.httpClientAdapter as IOHttpClientAdapter).createHttpClient = () {
        final client = HttpClient();
        if (kDebugMode) {
          client.badCertificateCallback = (X509Certificate cert, String host, int port) => true;
        }
        return client;
      };
    }
    
    AppLogger.instance.log('AuthService', 'Initialized with base $_serverBase (HTTPS/Secure)');
  }

  /// Login with username and password
  Future<LoginResult> login(String username, String password) async {
    AppLogger.instance.log('AuthService', 'Attempting login for user: $username');
    
    try {
      final response = await _dio.post(
        '/api/auth/login',
        data: {
          'username': username,
          'password': password,
        },
      );

      if (response.statusCode == 200) {
        final data = response.data;
        
        if (data['success'] == true) {
          _sessionToken = data['session_token'];
          _csrfToken = data['csrf_token'];
          _currentUser = data['user'];
          
          // Persist session token
          await _saveSession();
          
          _authStateController.add(true);
          
          AppLogger.instance.log('AuthService', 'Login successful: ${_currentUser?['username']}');
          
          return LoginResult(
            success: true,
            message: data['message'] ?? 'Login successful',
            user: _currentUser,
          );
        } else {
          return LoginResult(
            success: false,
            message: data['message'] ?? 'Login failed',
          );
        }
      } else {
        return LoginResult(
          success: false,
          message: 'Server error: ${response.statusCode}',
        );
      }
    } on DioException catch (e) {
      AppLogger.instance.log('AuthService', 'Login error: $e', isError: true);
      
      String errorMessage = 'Network error';
      if (e.response?.statusCode == 401) {
        errorMessage = 'Invalid username or password';
      } else if (e.response?.statusCode == 429) {
        errorMessage = 'Too many attempts. Please try again later.';
      } else if (e.type == DioExceptionType.connectionTimeout) {
        errorMessage = 'Connection timeout';
      } else if (e.type == DioExceptionType.connectionError) {
        errorMessage = 'Cannot connect to server';
      }
      
      return LoginResult(
        success: false,
        message: errorMessage,
      );
    } catch (e) {
      AppLogger.instance.log('AuthService', 'Unexpected login error: $e', isError: true);
      return LoginResult(
        success: false,
        message: 'Unexpected error: $e',
      );
    }
  }

  /// Register a new account (self-signup) and establish a session
  Future<LoginResult> register(String username, String password, {String? email}) async {
    AppLogger.instance.log('AuthService', 'Attempting registration for user: $username');

    try {
      final payload = <String, dynamic>{
        'username': username,
        'password': password,
      };
      if (email != null && email.trim().isNotEmpty) {
        payload['email'] = email.trim();
      }

      final response = await _dio.post(
        '/api/auth/register',
        data: payload,
      );

      if (response.statusCode == 200) {
        final data = response.data;
        if (data['success'] == true) {
          _sessionToken = data['session_token'];
          _csrfToken = data['csrf_token'];
          _currentUser = data['user'];

          await _saveSession();
          _authStateController.add(true);

          AppLogger.instance.log('AuthService', 'Registration successful: ${_currentUser?['user_id'] ?? username}');
          return LoginResult(
            success: true,
            message: data['message'] ?? 'Account created',
            user: _currentUser,
          );
        }
        return LoginResult(
          success: false,
          message: data['message'] ?? 'Registration failed',
        );
      }

      return LoginResult(
        success: false,
        message: 'Server error: ${response.statusCode}',
      );
    } on DioException catch (e) {
      AppLogger.instance.log('AuthService', 'Registration error: $e', isError: true);
      String errorMessage = 'Network error';
      final status = e.response?.statusCode;
      if (status == 409) {
        errorMessage = 'Username already exists';
      } else if (status == 400) {
        errorMessage = e.response?.data?['detail']?.toString() ?? 'Invalid registration data';
      } else if (status == 429) {
        errorMessage = 'Too many attempts. Please try again later.';
      }
      return LoginResult(success: false, message: errorMessage);
    } catch (e) {
      AppLogger.instance.log('AuthService', 'Unexpected registration error: $e', isError: true);
      return LoginResult(success: false, message: 'Unexpected error: $e');
    }
  }

  /// Check if current session is valid
  Future<bool> checkSession() async {
    if (_sessionToken == null) {
      // Try to load from storage
      await _loadSession();
      if (_sessionToken == null) {
        return false;
      }
    }

    AppLogger.instance.log('AuthService', 'Checking session validity');
    
    try {
      final response = await _dio.get(
        '/api/auth/check',
        options: Options(
          headers: {
            'Cookie': 'ws_session=$_sessionToken',
          },
        ),
      );

      if (response.statusCode == 200) {
        final data = response.data;
        
        if (data['valid'] == true) {
          _currentUser = data['user'];
          _authStateController.add(true);
          AppLogger.instance.log('AuthService', 'Session valid');
          return true;
        } else {
          await logout();
          return false;
        }
      } else {
        await logout();
        return false;
      }
    } catch (e) {
      AppLogger.instance.log('AuthService', 'Session check failed: $e', isError: true);
      await logout();
      return false;
    }
  }

  /// Logout and clear session
  Future<void> logout() async {
    AppLogger.instance.log('AuthService', 'Logging out');
    
    if (_sessionToken != null) {
      try {
        await _dio.post(
          '/api/auth/logout',
          options: Options(
            headers: {
              'Cookie': 'ws_session=$_sessionToken',
            },
          ),
        );
      } catch (e) {
        AppLogger.instance.log('AuthService', 'Logout API call failed: $e', isError: true);
      }
    }
    
    _sessionToken = null;
    _currentUser = null;
    await _clearSession();
    _authStateController.add(false);
  }

  // Phase 6 Security: Use encrypted storage for sensitive session data
  // flutter_secure_storage uses Android KeyStore and iOS Keychain
  final _secureStorage = const FlutterSecureStorage(
    aOptions: AndroidOptions(
      encryptedSharedPreferences: true,
    ),
    iOptions: IOSOptions(
      accessibility: KeychainAccessibility.first_unlock_this_device,
    ),
  );

  /// Save session to secure encrypted storage
  Future<void> _saveSession() async {
    if (_sessionToken != null && _currentUser != null) {
      await _secureStorage.write(key: 'session_token', value: _sessionToken!);
      await _secureStorage.write(key: 'user_data', value: jsonEncode(_currentUser!));
      if (_csrfToken != null) {
        await _secureStorage.write(key: 'csrf_token', value: _csrfToken!);
      }
      AppLogger.instance.log('AuthService', 'Session saved to secure storage');
    }
  }

  /// Load session from secure encrypted storage
  Future<void> _loadSession() async {
    _sessionToken = await _secureStorage.read(key: 'session_token');
    _csrfToken = await _secureStorage.read(key: 'csrf_token');
    final userData = await _secureStorage.read(key: 'user_data');
    
    if (userData != null) {
      try {
        _currentUser = jsonDecode(userData);
        AppLogger.instance.log('AuthService', 'Session loaded from secure storage (csrf=${_csrfToken != null})');
      } catch (e) {
        AppLogger.instance.log('AuthService', 'Failed to parse user data: $e', isError: true);
        await _clearSession();
      }
    }
  }

  /// Clear session from secure encrypted storage
  Future<void> _clearSession() async {
    await _secureStorage.delete(key: 'session_token');
    await _secureStorage.delete(key: 'user_data');
    await _secureStorage.delete(key: 'csrf_token');
    AppLogger.instance.log('AuthService', 'Session cleared from secure storage');
  }

  void dispose() {
    _authStateController.close();
  }
}

class LoginResult {
  final bool success;
  final String message;
  final Map<String, dynamic>? user;

  LoginResult({
    required this.success,
    required this.message,
    this.user,
  });
}

