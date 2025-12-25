import 'dart:async';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:demo_ai_even/services/proto.dart';

// Debug logging helper
void _log(String message) {
  if (kDebugMode) {
    debugPrint(message);
  }
}

/// Service for managing text display and pagination on the smart glasses.
///
/// Handles text measurement, line wrapping, and pagination for the glasses display.
class TextDisplayService {
  static TextDisplayService? _instance;
  static TextDisplayService get instance => _instance ??= TextDisplayService._();

  /// Maximum width of the glasses display in logical pixels
  static const double maxDisplayWidth = 488;

  /// Font size used for display text
  static const double fontSize = 21;

  /// Maximum retry attempts for sending data
  static const int maxRetry = 10;

  /// Current list of measured lines
  List<String> _lines = [];

  /// Current line index for pagination
  int _currentLine = 0;

  /// Lines per page
  static const int linesPerPage = 5;

  /// Whether manual navigation is active
  bool _isManual = false;

  /// Timer for auto-pagination
  Timer? _timer;

  /// Retry counter for sending
  int _retryCount = 0;

  TextDisplayService._();

  /// Get the total number of pages
  int get totalPages {
    if (_lines.isEmpty) return 0;
    if (_lines.length < linesPerPage + 1) return 1;
    return (_lines.length / linesPerPage).ceil();
  }

  /// Get the current page number (1-indexed)
  int get currentPage {
    if (_currentLine == 0) return 1;
    return (_currentLine / linesPerPage).floor() + 1;
  }

  /// Check if there are more pages after the current one
  bool get hasNextPage => _currentLine + linesPerPage < _lines.length - 1;

  /// Check if there are pages before the current one
  bool get hasPreviousPage => _currentLine > 0;

  /// Set content to display, measuring and wrapping text
  void setContent(String text) {
    _currentLine = 0;
    _lines = measureStringList(text);
    _isManual = false;
    _log("📄 TextDisplayService: Set content with ${_lines.length} lines, ${totalPages} pages.");
  }

  /// Clear the current content
  void clear() {
    _lines.clear();
    _currentLine = 0;
    _isManual = false;
    _timer?.cancel();
    _timer = null;
    _retryCount = 0;
  }

  /// Send reply text to the glasses display
  Future<bool> sendReply(String text, int type, int status, int pos) async {
    _log('📺 TextDisplayService: Sending - Status: 0x${status.toRadixString(16)}');

    bool isSuccess = await Proto.sendEvenAIData(
      text,
      newScreen: _transferToNewScreen(type, status),
      pos: pos,
      current_page_num: currentPage,
      max_page_num: totalPages,
    );

    if (!isSuccess && _retryCount < maxRetry) {
      _retryCount++;
      _log("🔄 TextDisplayService: Retry $_retryCount/$maxRetry");
      await Future.delayed(const Duration(milliseconds: 200));
      return await sendReply(text, type, status, pos);
    }

    if (!isSuccess) {
      _log("❌ TextDisplayService: Send failed after all retries.");
    }

    _retryCount = 0;
    return isSuccess;
  }

  /// Send a network error reply
  Future<void> sendNetworkErrorReply(String text) async {
    _currentLine = 0;
    _lines = measureStringList(text);
    String replyWords = _lines.sublist(0, min(3, _lines.length)).map((str) => '$str\n').join();
    String headString = '\n\n';
    replyWords = headString + replyWords;
    await sendReply(replyWords, 0x01, 0x60, 0);
    clear();
  }

  /// Start sending reply with auto-pagination
  Future<void> startSendReply(String text) async {
    _currentLine = 0;
    _lines = measureStringList(text);

    if (_lines.length < 4) {
      String startScreenWords = _lines.sublist(0, min(3, _lines.length)).map((str) => '$str\n').join();
      String headString = '\n\n';
      startScreenWords = headString + startScreenWords;
      await sendReply(startScreenWords, 0x01, 0x30, 0);
      await Future.delayed(const Duration(seconds: 3));
      if (_isManual) return;
      await sendReply(startScreenWords, 0x01, 0x40, 0);
      return;
    }

    if (_lines.length == 4) {
      String startScreenWords = _lines.sublist(0, 4).map((str) => '$str\n').join();
      String headString = '\n';
      startScreenWords = headString + startScreenWords;
      await sendReply(startScreenWords, 0x01, 0x30, 0);
      await Future.delayed(const Duration(seconds: 3));
      if (_isManual) return;
      await sendReply(startScreenWords, 0x01, 0x40, 0);
      return;
    }

    if (_lines.length == 5) {
      String startScreenWords = _lines.sublist(0, 5).map((str) => '$str\n').join();
      await sendReply(startScreenWords, 0x01, 0x30, 0);
      await Future.delayed(const Duration(seconds: 3));
      if (_isManual) return;
      await sendReply(startScreenWords, 0x01, 0x40, 0);
      return;
    }

    String startScreenWords = _lines.sublist(0, 5).map((str) => '$str\n').join();
    bool isSuccess = await sendReply(startScreenWords, 0x01, 0x30, 0);
    if (isSuccess) {
      _currentLine = 0;
      await _updateReplyByTimer();
    } else {
      clear();
    }
  }

  /// Custom interval for interview mode (seconds) - set before calling startSendText
  static int? customIntervalSeconds;

  /// Auto-paginate with timer
  Future<void> _updateReplyByTimer() async {
    // Use custom interval if set, otherwise default 5 seconds
    final int interval = customIntervalSeconds ?? 5;
    _log('📺 TextDisplayService: Using timer interval: ${interval}s (customIntervalSeconds: $customIntervalSeconds)');
    _timer?.cancel();
    _timer = Timer.periodic(Duration(seconds: interval), (timer) async {
      if (_isManual) {
        _timer?.cancel();
        _timer = null;
        return;
      }

      _currentLine = min(_currentLine + linesPerPage, _lines.length - 1);
      List<String> sendReplys = _lines.sublist(_currentLine);

      if (_currentLine >= _lines.length - 1) {
        _timer?.cancel();
        _timer = null;
        return;
      }

      String mergedStr = sendReplys
          .sublist(0, min(linesPerPage, sendReplys.length))
          .map((str) => '$str\n')
          .join();

      int status = (_currentLine >= _lines.length - linesPerPage) ? 0x40 : 0x30;
      await sendReply(mergedStr, 0x01, status, 0);

      if (status == 0x40) {
        _timer?.cancel();
        _timer = null;
      }
    });
  }

  /// Navigate to next page manually
  void nextPage() {
    if (_lines.isEmpty) return;
    _isManual = true;
    _timer?.cancel();
    _timer = null;

    if (totalPages < 2) {
      _manualForJustOnePage();
      return;
    }

    if (_currentLine + linesPerPage > _lines.length - 1) {
      return;
    }

    _currentLine += linesPerPage;
    _updateReplyByManual();
  }

  /// Navigate to previous page manually
  void previousPage() {
    if (_lines.isEmpty) return;
    _isManual = true;
    _timer?.cancel();
    _timer = null;

    if (totalPages < 2) {
      _manualForJustOnePage();
      return;
    }

    if (_currentLine - linesPerPage < 0) {
      _currentLine = 0;
    } else {
      _currentLine -= linesPerPage;
    }

    _updateReplyByManual();
  }

  Future<void> _updateReplyByManual() async {
    if (_currentLine < 0 || _currentLine > _lines.length - 1) {
      return;
    }

    List<String> sendReplys = _lines.sublist(_currentLine);
    String mergedStr = sendReplys
        .sublist(0, min(linesPerPage, sendReplys.length))
        .map((str) => '$str\n')
        .join();
    await sendReply(mergedStr, 0x01, 0x50, 0);
  }

  Future<void> _manualForJustOnePage() async {
    if (_lines.length < 4) {
      String screenWords = _lines.sublist(0, min(3, _lines.length)).map((str) => '$str\n').join();
      String headString = '\n\n';
      screenWords = headString + screenWords;
      await sendReply(screenWords, 0x01, 0x50, 0);
      return;
    }

    if (_lines.length == 4) {
      String screenWords = _lines.sublist(0, 4).map((str) => '$str\n').join();
      String headString = '\n';
      screenWords = headString + screenWords;
      await sendReply(screenWords, 0x01, 0x50, 0);
      return;
    }

    if (_lines.length == 5) {
      String screenWords = _lines.sublist(0, 5).map((str) => '$str\n').join();
      await sendReply(screenWords, 0x01, 0x50, 0);
      return;
    }
  }

  /// Set manual mode (disables auto-pagination)
  void setManualMode(bool isManual) {
    _isManual = isManual;
    if (isManual) {
      _timer?.cancel();
      _timer = null;
    }
  }

  /// Measure and wrap text into display lines
  static List<String> measureStringList(String text, [double? maxW]) {
    final double maxWidth = maxW ?? maxDisplayWidth;
    
    List<String> paragraphs = text
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();

    List<String> result = [];
    TextStyle ts = const TextStyle(fontSize: fontSize);

    for (String paragraph in paragraphs) {
      final textSpan = TextSpan(text: paragraph, style: ts);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
        maxLines: null,
      );
      textPainter.layout(maxWidth: maxWidth);
      final lineCount = textPainter.computeLineMetrics().length;
      var start = 0;
      for (var i = 0; i < lineCount; i++) {
        final line = textPainter.getLineBoundary(TextPosition(offset: start));
        result.add(paragraph.substring(line.start, line.end).trim());
        start = line.end;
      }
    }

    return result;
  }

  int _transferToNewScreen(int type, int status) {
    return status | type;
  }
}
