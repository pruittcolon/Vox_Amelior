import 'dart:async';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:demo_ai_even/services/text_display_service.dart';
import 'package:demo_ai_even/services/proto.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

class TextService {
  static TextService? _instance;
  static TextService get get => _instance ??= TextService._();
  static bool isRunning = false;
  static int maxRetry = 5;
  static int _currentLine = 0;
  static Timer? _timer;
  static List<String> list = [];
  static List<String> sendReplys = [];

  TextService._();

  Future startSendText(String text) async {
    isRunning = true;

    _currentLine = 0;
    list = TextDisplayService.measureStringList(text);

    if (list.length < 4) {
      String startScreenWords =
          list.sublist(0, min(3, list.length)).map((str) => '$str\n').join();
      String headString = '\n\n';
      startScreenWords = headString + startScreenWords;

      await doSendText(startScreenWords, 0x01, 0x70, 0);
      // If custom interval is set (interview mode), wait before returning
      if (customIntervalMs != null) {
        _log('🕐 TextService: Single page - waiting ${customIntervalMs}ms');
        await Future.delayed(Duration(milliseconds: customIntervalMs!));
      }
      return;
    }

    if (list.length == 4) {
      String startScreenWords =
          list.sublist(0, 4).map((str) => '$str\n').join();
      String headString = '\n';
      startScreenWords = headString + startScreenWords;
      await doSendText(startScreenWords, 0x01, 0x70, 0);
      // If custom interval is set (interview mode), wait before returning
      if (customIntervalMs != null) {
        _log('🕐 TextService: Single page - waiting ${customIntervalMs}ms');
        await Future.delayed(Duration(milliseconds: customIntervalMs!));
      }
      return;
    }

    if (list.length == 5) {
      String startScreenWords =
          list.sublist(0, 5).map((str) => '$str\n').join();
      await doSendText(startScreenWords, 0x01, 0x70, 0);
      // If custom interval is set (interview mode), wait before returning
      if (customIntervalMs != null) {
        _log('🕐 TextService: Single page - waiting ${customIntervalMs}ms');
        await Future.delayed(Duration(milliseconds: customIntervalMs!));
      }
      return;
    }

    String startScreenWords = list.sublist(0, 5).map((str) => '$str\n').join();
    bool isSuccess = await doSendText(startScreenWords, 0x01, 0x70, 0);
    if (isSuccess) {
      _currentLine = 0;
      await updateReplyToOSByTimer();
    } else {
      clear();
    }
  }

  int retryCount = 0;
  Future<bool> doSendText(String text, int type, int status, int pos) async {
    if (kDebugMode) {
      debugPrint(
          '${DateTime.now()} doSendText--currentPage---${getCurrentPage()}-----text----$text-----type---$type---status---$status----pos---$pos-');
    }
    if (!isRunning) {
      return false;
    }

    bool isSuccess = await Proto.sendEvenAIData(text,
        newScreen: status | type,
        pos: pos,
        current_page_num: getCurrentPage(),
        max_page_num: getTotalPages());
    if (!isSuccess) {
      if (retryCount < maxRetry) {
        retryCount++;
        await doSendText(text, type, status, pos);
      } else {
        retryCount = 0;
        return false;
      }
    }
    retryCount = 0;
    return true;
  }

  /// Custom interval for interview mode (ms) - set before calling startSendText
  static int? customIntervalMs;

  Future updateReplyToOSByTimer() async {
    if (!isRunning) return;
    // Use custom interval if set, otherwise default 5 seconds
    int intervalMs = customIntervalMs ?? 5000;
    _log('🕐 TextService: Using timer interval: ${intervalMs}ms (customIntervalMs: $customIntervalMs)');

    _timer?.cancel();
    _timer = Timer.periodic(Duration(milliseconds: intervalMs), (timer) async {
      // CRITICAL: Check if we should still be running at start of each tick
      // This prevents pages from displaying after terminate is called
      if (!isRunning) {
        timer.cancel();
        _timer = null;
        return;
      }

      _currentLine = min(_currentLine + 5, list.length - 1);
      sendReplys = list.sublist(_currentLine);

      if (_currentLine > list.length - 1) {
        _timer?.cancel();
        _timer = null;

        clear();
      } else {
        // Check again before sending in case terminate was called
        if (!isRunning) {
          timer.cancel();
          _timer = null;
          return;
        }

        if (sendReplys.length < 4) {
          var mergedStr = sendReplys
              .sublist(0, sendReplys.length)
              .map((str) => '$str\n')
              .join();

          if (_currentLine >= list.length - 5) {
            await doSendText(mergedStr, 0x01, 0x70, 0);
            _timer?.cancel();
            _timer = null;
          } else {
            await doSendText(mergedStr, 0x01, 0x70, 0);
          }
        } else {
          var mergedStr = sendReplys
              .sublist(0, min(5, sendReplys.length))
              .map((str) => '$str\n')
              .join();

          if (_currentLine >= list.length - 5) {
            await doSendText(mergedStr, 0x01, 0x70, 0);
            _timer?.cancel();
            _timer = null;
          } else {
            await doSendText(mergedStr, 0x01, 0x70, 0);
          }
        }
      }
    });
  }

  int getTotalPages() {
    if (list.isEmpty) {
      return 0;
    }
    if (list.length < 6) {
      return 1;
    }
    int pages = 0;
    int div = list.length ~/ 5;
    int rest = list.length % 5;
    pages = div;
    if (rest != 0) {
      pages++;
    }
    return pages;
  }

  int getCurrentPage() {
    if (_currentLine == 0) {
      return 1;
    }
    int currentPage = 1;
    int div = _currentLine ~/ 5;
    int rest = _currentLine % 5;
    currentPage = 1 + div;
    if (rest != 0) {
      currentPage++;
    }
    return currentPage;
  }

  Future stopTextSendingByOS() async {
    _log("stopTextSendingByOS---------------");
    isRunning = false;
    clear();
  }

  void clear() {
    isRunning = false;
    _currentLine = 0;
    _timer?.cancel();
    _timer = null;
    list = [];
    sendReplys = [];
    retryCount = 0;
  }
}
