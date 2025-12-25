import 'dart:io';

import 'package:demo_ai_even/ble_manager.dart';
import 'package:demo_ai_even/controllers/evenai_model_controller.dart';
import 'package:demo_ai_even/views/home_page.dart';
import 'package:demo_ai_even/views/login_page.dart';
import 'package:demo_ai_even/services/auth_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

/// Custom HttpOverrides to accept self-signed SSL certificates.
/// SECURITY: Only enabled in debug mode (kDebugMode) per ISO 27002 5.14.
/// Release builds use strict TLS verification.
class DevCertHttpOverrides extends HttpOverrides {
  @override
  HttpClient createHttpClient(SecurityContext? context) {
    return super.createHttpClient(context)
      ..badCertificateCallback = (X509Certificate cert, String host, int port) {
        // Accept self-signed certificates for local development
        // Log for debugging
        print("🔐 [SSL] Accepting certificate for $host:$port");
        return true;
      };
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // ISO 27002 5.14: TLS bypass ONLY in debug mode
  // Release builds use strict certificate verification
  if (kDebugMode) {
    HttpOverrides.global = DevCertHttpOverrides();
    print("🔐 [DEBUG] SSL bypass enabled - debug mode only");
  } else {
    print("🔒 [RELEASE] Using strict TLS verification");
  }
  
  // Load environment variables
  try {
    await dotenv.load(fileName: ".env");
    print("✅ [DEBUG] .env loaded successfully");
  } catch (e) {
    print("❌ [ERROR] Failed to load .env: $e");
  }
  
  // Initialize services
  BleManager.get();
  Get.put(EvenaiModelController());
  AuthService.instance.initialize();
  
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vox Augmented',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
      ),
      // Start with LoginPage instead of HomePage
      home: LoginPage(), 
    );
  }
}
