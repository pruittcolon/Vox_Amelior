import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';
import 'package:vox_amelior_mobile/automation/action_executor.dart';
import 'package:vox_amelior_mobile/automation/automation_repositories.dart';
import 'package:vox_amelior_mobile/automation/http_sender.dart';
import 'package:vox_amelior_mobile/automation/rule.dart';
import 'package:vox_amelior_mobile/automation/rule_engine.dart';
import 'package:vox_amelior_mobile/automation/segment_handler.dart';
import 'package:vox_amelior_mobile/automation/template.dart';
import 'package:vox_amelior_mobile/automation/webhook_dispatcher.dart';
import 'package:vox_amelior_mobile/core/database.dart';
import 'package:vox_amelior_mobile/data/models.dart';

class FakeSender implements HttpSender {
  final List<HttpRequestSpec> sent = [];
  final List<Object> script = []; // int status or Exception to throw

  @override
  Future<HttpResult> send(HttpRequestSpec spec) async {
    sent.add(spec);
    final next = script.isEmpty ? 200 : script.removeAt(0);
    if (next is Exception) throw next;
    return HttpResult(next as int);
  }
}

class FakeNotifier implements Notifier {
  final List<(String, String)> shown = [];
  @override
  Future<void> show(String title, String body) async => shown.add((title, body));
}

SegmentView seg(String text, {String? speaker, int id = 1}) => SegmentView(
      id: id,
      conversationId: 7,
      startedAt: DateTime(2026, 6, 1, 18, 5),
      duration: const Duration(seconds: 3),
      text: text,
      speakerId: speaker == null ? null : 's-$speaker',
      speakerName: speaker,
    );

AutomationRule rule(
  String id, {
  RuleTrigger? trigger,
  List<RuleAction>? actions,
  int cooldown = 30,
  bool enabled = true,
}) =>
    AutomationRule(
      id: id,
      name: 'Rule $id',
      enabled: enabled,
      trigger: trigger ?? const RuleTrigger(phrases: ['lights']),
      actions: actions ?? [const NotifyAction(title: 't', body: 'b')],
      cooldownSeconds: cooldown,
    );

void main() {
  group('template', () {
    test('substitutes values, filters and unknown names', () {
      final out = renderTemplate(
        '{"t":"{{text|json}}","q":"{{text|url}}","raw":"{{speaker}}","x":"{{nope}}"}',
        {'text': 'say "hi"\nnow', 'speaker': 'Alex'},
      );
      expect(out, r'{"t":"say \"hi\"\nnow","q":"say%20%22hi%22%0Anow","raw":"Alex","x":""}');
      expect(jsonDecode(out), isA<Map<String, Object?>>());
    });
  });

  group('rule validation', () {
    test('accepts a good rule', () {
      final r = rule('a', actions: [const WebhookAction(url: 'https://example.com/hook')]);
      expect(validateRule(r), isEmpty);
    });

    test('catches the common mistakes', () {
      expect(validateRule(rule('a', trigger: const RuleTrigger())).join(), contains('fire on everything'));
      expect(validateRule(rule('a', actions: [])).join(), contains('at least one action'));
      expect(validateRule(rule('a', trigger: const RuleTrigger(pattern: '('))).join(), contains('regular expression'));
      expect(validateRule(rule('a', actions: [const WebhookAction(url: 'ftp://x')])).join(), contains('https://'));
      expect(validateRule(rule('a', actions: [const WebhookAction(url: 'http://192.168.1.5/x')])).join(), contains('blocked'));
      expect(validateRule(rule('a', actions: [const WebhookAction(url: 'http://192.168.1.5/x', allowInsecureHttp: true)])), isEmpty);
      expect(validateRule(rule('a', actions: [const WebhookAction(url: 'https://x.io', headers: {'bad name': 'v'})])).join(), contains('header'));
    });

    test('a catch-all wake-command rule is allowed', () {
      expect(validateRule(rule('a', trigger: const RuleTrigger(scope: TriggerScope.wakeCommand))), isEmpty);
    });

    test('placeholders in the URL are accepted', () {
      final r = rule('a', actions: [const WebhookAction(url: 'https://x.io/say?q={{text|url}}')]);
      expect(validateRule(r), isEmpty);
    });

    test('rules survive a JSON round trip', () {
      final db = AppDatabase.inMemory();
      final repo = RuleRepository(db);
      final original = rule('r1', trigger: const RuleTrigger(scope: TriggerScope.wakeCommand, phrases: ['lights'], speakerName: 'Alex'), actions: [
        const WebhookAction(url: 'https://x.io', headers: {'A': 'b'}, secret: 's', bodyTemplate: '{}'),
        const NotifyAction(title: 'T', body: 'B'),
        const NoteAction(template: 'N'),
      ]);
      repo.save(original);
      final back = repo.all().single;
      expect(back.trigger.scope, TriggerScope.wakeCommand);
      expect(back.trigger.speakerName, 'Alex');
      expect(back.actions.length, 3);
      expect((back.actions.first as WebhookAction).secret, 's');
      db.close();
    });
  });

  group('rule engine', () {
    test('matches whole words only, case-insensitively, ignoring punctuation', () {
      final engine = RuleEngine();
      final r = [rule('a', trigger: const RuleTrigger(phrases: ['turn on the lights']))];
      expect(engine.evaluate(r, seg('Please, TURN on the lights!')).length, 1);
      expect(RuleEngine().evaluate(r, seg('turn on the lightsaber')).length, 0);
      expect(RuleEngine().evaluate(r, seg('nothing relevant')).length, 0);
    });

    test('regex triggers work and bad regexes never throw', () {
      final ok = [rule('a', trigger: const RuleTrigger(pattern: r'remind me to (.+)'))];
      final fires = RuleEngine().evaluate(ok, seg('Remind me to call mum'));
      expect(fires.single.context['match'], 'Remind me to call mum');
      final bad = [rule('b', trigger: const RuleTrigger(pattern: '('))];
      expect(RuleEngine().evaluate(bad, seg('anything')), isEmpty);
    });

    test('speaker filter only fires for that named person', () {
      final r = [rule('a', trigger: const RuleTrigger(phrases: ['lights'], speakerName: 'alex'))];
      expect(RuleEngine().evaluate(r, seg('lights', speaker: 'Alex')).length, 1);
      expect(RuleEngine().evaluate(r, seg('lights', speaker: 'Sam')).length, 0);
      expect(RuleEngine().evaluate(r, seg('lights')).length, 0);
    });

    test('wake-command rules only see the command after the wake phrase', () {
      final r = [rule('a', trigger: const RuleTrigger(scope: TriggerScope.wakeCommand, phrases: ['lights']))];
      expect(RuleEngine().evaluate(r, seg('lights on')), isEmpty);
      final fires = RuleEngine().evaluate(r, seg('hey vox lights on'), wakeCommand: 'lights on');
      expect(fires.single.context['command'], 'lights on');
      final catchAll = [rule('b', trigger: const RuleTrigger(scope: TriggerScope.wakeCommand))];
      expect(RuleEngine().evaluate(catchAll, seg('x'), wakeCommand: 'anything').length, 1);
    });

    test('cooldown suppresses repeats until it elapses; disabled rules never fire', () {
      var now = DateTime(2026, 1, 1, 12);
      final engine = RuleEngine(clock: () => now);
      final r = [rule('a', cooldown: 60), rule('b', enabled: false)];
      expect(engine.evaluate(r, seg('lights')).length, 1);
      now = now.add(const Duration(seconds: 30));
      expect(engine.evaluate(r, seg('lights')), isEmpty);
      now = now.add(const Duration(seconds: 31));
      expect(engine.evaluate(r, seg('lights')).length, 1);
    });

    test('context exposes speaker, time and ids', () {
      final f = RuleEngine().evaluate([rule('a')], seg('lights', speaker: 'Alex')).single;
      expect(f.context['speaker'], 'Alex');
      expect(f.context['time'], '18:05');
      expect(f.context['date'], '2026-06-01');
      expect(f.context['conversation_id'], '7');
    });
  });

  group('webhook dispatcher', () {
    late AppDatabase db;
    late OutboxRepository outbox;
    late FakeSender sender;
    late WebhookDispatcher dispatcher;
    var now = DateTime(2026, 1, 1, 12);

    setUp(() {
      now = DateTime(2026, 1, 1, 12);
      db = AppDatabase.inMemory();
      outbox = OutboxRepository(db, clock: () => now);
      sender = FakeSender();
      dispatcher = WebhookDispatcher(outbox, sender, clock: () => now);
    });
    tearDown(() => db.close());

    int enqueue({String url = 'https://x.io/h', String? secret, bool insecure = false}) => outbox.enqueue(
          url: url,
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: '{"a":1}',
          secret: secret,
          allowInsecure: insecure,
        );

    test('delivers and marks delivered', () async {
      enqueue();
      final report = await dispatcher.flush();
      expect(report.delivered, 1);
      expect(outbox.recent().single.status, OutboxStatus.delivered);
      expect(sender.sent.single.body, '{"a":1}');
    });

    test('signs the body with HMAC-SHA256 over timestamp.body', () async {
      enqueue(secret: 'topsecret');
      await dispatcher.flush();
      final h = sender.sent.single.headers;
      final expected = Hmac(sha256, utf8.encode('topsecret')).convert(utf8.encode('${h['X-Vox-Timestamp']}.{"a":1}'));
      expect(h['X-Vox-Signature'], 'sha256=$expected');
    });

    test('server errors retry with backoff, then succeed', () async {
      sender.script.addAll([503, 200]);
      enqueue();
      var r = await dispatcher.flush();
      expect(r.retrying, 1);
      expect((await dispatcher.flush()).delivered, 0); // not due yet
      now = now.add(const Duration(seconds: 31));
      r = await dispatcher.flush();
      expect(r.delivered, 1);
    });

    test('network exceptions are retried, and give up after max attempts', () async {
      sender.script.addAll(List.generate(10, (_) => const SocketException('down')));
      enqueue();
      for (var i = 0; i < 6; i++) {
        await dispatcher.flush();
        now = now.add(const Duration(hours: 7));
      }
      final item = outbox.recent().single;
      expect(item.status, OutboxStatus.failed);
      expect(item.attempts, 6);
      expect(item.lastError, contains('Network error'));
    });

    test('client errors fail immediately without retrying', () async {
      sender.script.add(404);
      enqueue();
      final r = await dispatcher.flush();
      expect(r.failed, 1);
      expect(outbox.recent().single.status, OutboxStatus.failed);
      outbox.retryNow(outbox.recent().single.id);
      expect(outbox.recent().single.status, OutboxStatus.pending);
    });

    test('plain http is refused unless the rule allows it; junk URLs fail safely', () async {
      enqueue(url: 'http://192.168.1.10/x');
      enqueue(url: 'http://192.168.1.10/x', insecure: true);
      enqueue(url: 'file:///etc/passwd');
      final r = await dispatcher.flush();
      expect(r.delivered, 1);
      expect(r.failed, 2);
      expect(sender.sent.length, 1);
    });

    test('overlapping flushes never send an item twice', () async {
      enqueue();
      final results = await Future.wait([dispatcher.flush(), dispatcher.flush()]);
      expect(results.fold<int>(0, (a, r) => a + r.delivered), 1);
      expect(sender.sent.length, 1);
    });
  });

  group('DioHttpSender over a real local server', () {
    test('sends method, headers and body, and returns the status', () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));
      final received = Completer<(String, String?, String)>();
      unawaited(server.forEach((req) async {
        final body = await utf8.decodeStream(req);
        received.complete((req.method, req.headers.value('x-test'), body));
        req.response.statusCode = 202;
        await req.response.close();
      }));

      final result = await DioHttpSender().send(HttpRequestSpec(
        method: 'POST',
        url: Uri.parse('http://127.0.0.1:${server.port}/hook'),
        headers: {'X-Test': 'yes', 'Content-Type': 'application/json'},
        body: '{"hello":"world"}',
      ));
      expect(result.statusCode, 202);
      expect(result.isSuccess, isTrue);
      expect(await received.future, ('POST', 'yes', '{"hello":"world"}'));
    });

    test('connection failures throw so the dispatcher can retry', () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      final port = server.port;
      await server.close();
      expect(
        DioHttpSender().send(HttpRequestSpec(method: 'GET', url: Uri.parse('http://127.0.0.1:$port'), headers: const {}, body: '')),
        throwsA(anything),
      );
    });
  });

  group('segment handler', () {
    late AppDatabase db;
    late FakeSender sender;
    late FakeNotifier notifier;
    late SegmentHandler handler;
    late RuleRepository rules;
    late NoteRepository notes;
    late AssistantRequestRepository requests;

    setUp(() {
      db = AppDatabase.inMemory();
      sender = FakeSender();
      notifier = FakeNotifier();
      rules = RuleRepository(db);
      notes = NoteRepository(db);
      requests = AssistantRequestRepository(db);
      final outbox = OutboxRepository(db);
      handler = SegmentHandler(
        rules: rules,
        engine: RuleEngine(),
        executor: ActionExecutor(
          outbox: outbox,
          dispatcher: WebhookDispatcher(outbox, sender),
          notes: notes,
          notifier: notifier,
        ),
        requests: requests,
        wakeParser: WakeCommandParser(['hey vox', 'vox']),
      );
    });
    tearDown(() => db.close());

    test('a matching rule notifies, saves a note and sends a webhook with rendered values', () async {
      rules.save(rule('r1', trigger: const RuleTrigger(phrases: ['groceries']), actions: [
        const NotifyAction(title: 'Heard {{speaker}}', body: '{{text}}'),
        const NoteAction(template: '[{{date}}] {{text}}'),
        const WebhookAction(url: 'https://x.io/h?q={{match|url}}', bodyTemplate: '{"who":"{{speaker|json}}"}'),
      ]));
      final outcome = await handler.handle(seg('we need groceries', speaker: 'Alex'));
      expect(outcome.firedRules, 1);
      expect(notifier.shown.single, ('Heard Alex', 'we need groceries'));
      expect(notes.all().single.text, '[2026-06-01] we need groceries');
      expect(sender.sent.single.url.toString(), 'https://x.io/h?q=groceries');
      expect(sender.sent.single.body, '{"who":"Alex"}');
      expect(rules.all().single.lastFiredAt, isNotNull);
    });

    test('with no template the webhook gets a sensible default JSON payload', () async {
      rules.save(rule('r1', actions: [const WebhookAction(url: 'https://x.io/h')]));
      await handler.handle(seg('turn the lights on', speaker: 'Sam'));
      final payload = jsonDecode(sender.sent.single.body) as Map<String, Object?>;
      expect(payload['source'], 'vox-amelior');
      expect(payload['speaker'], 'Sam');
      expect(payload['text'], 'turn the lights on');
    });

    test('a failing action does not prevent the others', () async {
      rules.save(rule('r1', actions: [
        const WebhookAction(url: 'not a url'),
        const NoteAction(template: 'still saved {{text}}'),
      ]));
      await handler.handle(seg('lights'));
      expect(notes.all().single.text, 'still saved lights');
    });

    test('wake commands are queued for the assistant', () async {
      final outcome = await handler.handle(seg('Hey Vox, what did Sam say about the plumber?'));
      expect(outcome.assistantRequestId, isNotNull);
      expect(requests.takePending().single.text, 'what did Sam say about the plumber?');
    });

    test('ordinary speech triggers neither rules nor assistant', () async {
      final outcome = await handler.handle(seg('nice weather today'));
      expect(outcome.firedRules, 0);
      expect(outcome.assistantRequestId, isNull);
      expect(requests.takePending(), isEmpty);
    });

    test('stale assistant requests expire instead of answering hours later', () {
      var now = DateTime(2026, 1, 1, 12);
      final repo = AssistantRequestRepository(db, clock: () => now);
      final id = repo.add('old question');
      now = now.add(const Duration(hours: 2));
      expect(repo.takePending(), isEmpty);
      expect(repo.recent().single.status, RequestStatus.failed);
      expect(repo.recent().single.id, id);
    });
  });
}
