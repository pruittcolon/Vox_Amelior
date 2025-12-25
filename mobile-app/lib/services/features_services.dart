import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/ble_manager.dart';
import 'package:demo_ai_even/controllers/bmp_update_manager.dart';
import 'package:demo_ai_even/services/proto.dart';
import 'package:demo_ai_even/utils/utils.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

class FeaturesServices {
  final bmpUpdateManager = BmpUpdateManager();
  
  Future<void> sendBmp(String imageUrl) async {
    Uint8List bmpData = await Utils.loadBmpImage(imageUrl);
    int initialSeq = 0;
    bool isSuccess = await Proto.sendHeartBeat();
    if (kDebugMode) {
      debugPrint(
          "${DateTime.now()} testBMP -------startSendBeatHeart----isSuccess---$isSuccess------");
    }
    BleManager.get().startSendBeatHeart();

    final results = await Future.wait([
      bmpUpdateManager.updateBmp("L", bmpData, seq: initialSeq),
      bmpUpdateManager.updateBmp("R", bmpData, seq: initialSeq)
    ]);

    bool successL = results[0];
    bool successR = results[1];

    if (successL) {
      _log("${DateTime.now()} left ble success");
    } else {
      _log("${DateTime.now()} left ble fail");
    }

    if (successR) {
      _log("${DateTime.now()} right ble success");
    } else {
      _log("${DateTime.now()} right ble fail");
    }
  }

  Future<void> exitBmp() async {
    bool isSuccess = await Proto.exit();
    _log("exitBmp----isSuccess---$isSuccess--");
  }
}
