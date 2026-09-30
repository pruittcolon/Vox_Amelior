import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/core/clock.dart';
import 'package:vox_amelior_mobile/core/log.dart';

class DispatchReport {
  const DispatchReport({this.delivered = 0, this.retrying = 0, this.failed = 0});

  final int delivered;
  final int retrying;
  final int failed;
}

/// Delivers queued webhooks with retry/backoff and optional HMAC signing.
class WebhookDispatcher {
  WebhookDispatcher(
    this._outbox,
    this._sender, {
    this.clock = systemClock,
    this.maxAttempts = 6,
    this.backoff = const [
      Duration(seconds: 30),
      Duration(minutes: 2),
      Duration(minutes: 10),
      Duration(hours: 1),
      Duration(hours: 6),
    ],
  });

  final OutboxRepository _outbox;
  final HttpSender _sender;
  final Clock clock;
  final int maxAttempts;
  final List<Duration> backoff;

  bool _busy = false;

  /// Sends everything that is due. Safe to call repeatedly; overlapping calls
  /// are ignored so one item is never sent twice concurrently.
  Future<DispatchReport> flush({int batch = 20}) async {
    if (_busy) return const DispatchReport();
    _busy = true;
    var delivered = 0;
    var retrying = 0;
    var failed = 0;
    try {
      for (final item in _outbox.due(limit: batch)) {
        final outcome = await _deliver(item);
        switch (outcome) {
          case _Outcome.delivered:
            delivered++;
          case _Outcome.retry:
            retrying++;
          case _Outcome.failed:
            failed++;
        }
      }
    } finally {
      _busy = false;
    }
    return DispatchReport(delivered: delivered, retrying: retrying, failed: failed);
  }

  Future<_Outcome> _deliver(OutboxItem item) async {
    final uri = Uri.tryParse(item.url);
    if (uri == null || !uri.hasAuthority || !(uri.scheme == 'https' || uri.scheme == 'http')) {
      _outbox.markAttemptFailed(item.id, 'Invalid URL');
      return _Outcome.failed;
    }
    if (uri.scheme == 'http' && !item.allowInsecure) {
      _outbox.markAttemptFailed(item.id, 'Plain HTTP is not allowed for this rule');
      return _Outcome.failed;
    }

    final headers = Map<String, String>.of(item.headers);
    final secret = item.secret;
    if (secret != null && secret.isNotEmpty) {
      final ts = (clock().millisecondsSinceEpoch ~/ 1000).toString();
      final mac = Hmac(sha256, utf8.encode(secret)).convert(utf8.encode('$ts.${item.body}'));
      headers['X-Vox-Timestamp'] = ts;
      headers['X-Vox-Signature'] = 'sha256=$mac';
    }

    try {
      final result = await _sender.send(HttpRequestSpec(
        method: item.method,
        url: uri,
        headers: headers,
        body: item.body,
      ));
      if (result.isSuccess) {
        _outbox.markDelivered(item.id);
        return _Outcome.delivered;
      }
      if (result.statusCode >= 300 && result.statusCode < 400) {
        return _retryOrFail(item, 'HTTP ${result.statusCode} redirect (use the final URL)', permanent: true);
      }
      final permanent = result.statusCode >= 400 &&
          result.statusCode < 500 &&
          result.statusCode != 408 &&
          result.statusCode != 429;
      return _retryOrFail(item, 'HTTP ${result.statusCode}', permanent: permanent);
    } on Object catch (e) {
      Log.w('webhook', 'delivery ${item.id} failed', e.runtimeType);
      return _retryOrFail(item, 'Network error: ${e.runtimeType}', permanent: false);
    }
  }

  _Outcome _retryOrFail(OutboxItem item, String error, {required bool permanent}) {
    final attemptsAfter = item.attempts + 1;
    if (permanent || attemptsAfter >= maxAttempts) {
      _outbox.markAttemptFailed(item.id, error);
      return _Outcome.failed;
    }
    final delay = backoff[(attemptsAfter - 1).clamp(0, backoff.length - 1)];
    _outbox.markAttemptFailed(item.id, error, retryAt: clock().add(delay));
    return _Outcome.retry;
  }
}

enum _Outcome { delivered, retry, failed }
