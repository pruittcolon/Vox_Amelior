import 'dart:convert';

/// What a rule listens to.
enum TriggerScope {
  /// Any transcribed speech.
  anySpeech,

  /// Only commands said after the wake phrase ("Hey Vox, ...").
  wakeCommand,
}

class RuleTrigger {
  const RuleTrigger({
    this.scope = TriggerScope.anySpeech,
    this.phrases = const [],
    this.pattern,
    this.speakerName,
  });

  final TriggerScope scope;

  /// Fires if any phrase appears as whole words (case-insensitive).
  final List<String> phrases;

  /// Alternative to [phrases]: a regular expression.
  final String? pattern;

  /// Only fire for this person's voice (by name). Null means anyone.
  final String? speakerName;

  bool get hasCondition => phrases.isNotEmpty || (pattern != null && pattern!.isNotEmpty);

  Map<String, Object?> toJson() => {
        'scope': scope.name,
        'phrases': phrases,
        'pattern': pattern,
        'speaker': speakerName,
      };

  factory RuleTrigger.fromJson(Map<String, Object?> j) => RuleTrigger(
        scope: TriggerScope.values.asNameMap()[j['scope']] ?? TriggerScope.anySpeech,
        phrases: (j['phrases'] as List<Object?>?)?.cast<String>() ?? const [],
        pattern: j['pattern'] as String?,
        speakerName: j['speaker'] as String?,
      );
}

sealed class RuleAction {
  const RuleAction();

  Map<String, Object?> toJson();

  factory RuleAction.fromJson(Map<String, Object?> j) {
    switch (j['type']) {
      case 'webhook':
        return WebhookAction(
          url: j['url'] as String? ?? '',
          method: j['method'] as String? ?? 'POST',
          headers: ((j['headers'] as Map<String, Object?>?) ?? const {}).map((k, v) => MapEntry(k, '$v')),
          bodyTemplate: j['body'] as String?,
          secret: j['secret'] as String?,
          allowInsecureHttp: j['allowInsecureHttp'] as bool? ?? false,
        );
      case 'notify':
        return NotifyAction(title: j['title'] as String? ?? '', body: j['body'] as String? ?? '');
      case 'note':
        return NoteAction(template: j['template'] as String? ?? '');
      default:
        throw FormatException('Unknown action type: ${j['type']}');
    }
  }
}

/// Sends an HTTP request (Home Assistant, n8n, IFTTT, your own server...).
class WebhookAction extends RuleAction {
  const WebhookAction({
    required this.url,
    this.method = 'POST',
    this.headers = const {},
    this.bodyTemplate,
    this.secret,
    this.allowInsecureHttp = false,
  });

  /// Supports placeholders, e.g. `https://example.com/hook?q={{text|url}}`.
  final String url;
  final String method;
  final Map<String, String> headers;

  /// Request body with placeholders. Blank sends a default JSON payload.
  final String? bodyTemplate;

  /// If set, requests carry `X-Vox-Signature: sha256=<HMAC>` so the receiver
  /// can verify they came from this app.
  final String? secret;

  /// Plain `http://` is refused unless this is on (for LAN devices).
  final bool allowInsecureHttp;

  @override
  Map<String, Object?> toJson() => {
        'type': 'webhook',
        'url': url,
        'method': method,
        'headers': headers,
        'body': bodyTemplate,
        'secret': secret,
        'allowInsecureHttp': allowInsecureHttp,
      };
}

/// Shows a phone notification.
class NotifyAction extends RuleAction {
  const NotifyAction({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Map<String, Object?> toJson() => {'type': 'notify', 'title': title, 'body': body};
}

/// Saves a note (reminder, shopping item, ...) locally.
class NoteAction extends RuleAction {
  const NoteAction({required this.template});

  final String template;

  @override
  Map<String, Object?> toJson() => {'type': 'note', 'template': template};
}

class AutomationRule {
  const AutomationRule({
    required this.id,
    required this.name,
    this.enabled = true,
    required this.trigger,
    required this.actions,
    this.cooldownSeconds = 30,
    this.lastFiredAt,
  });

  final String id;
  final String name;
  final bool enabled;
  final RuleTrigger trigger;
  final List<RuleAction> actions;

  /// Minimum time between firings, so repeated phrases don't spam webhooks.
  final int cooldownSeconds;
  final DateTime? lastFiredAt;

  AutomationRule copyWith({
    String? name,
    bool? enabled,
    RuleTrigger? trigger,
    List<RuleAction>? actions,
    int? cooldownSeconds,
    DateTime? lastFiredAt,
  }) =>
      AutomationRule(
        id: id,
        name: name ?? this.name,
        enabled: enabled ?? this.enabled,
        trigger: trigger ?? this.trigger,
        actions: actions ?? this.actions,
        cooldownSeconds: cooldownSeconds ?? this.cooldownSeconds,
        lastFiredAt: lastFiredAt ?? this.lastFiredAt,
      );

  String triggerJson() => jsonEncode(trigger.toJson());
  String actionsJson() => jsonEncode(actions.map((a) => a.toJson()).toList());
}

/// Human-readable problems with a rule; empty means it is valid.
List<String> validateRule(AutomationRule rule) {
  final errors = <String>[];
  if (rule.name.trim().isEmpty) errors.add('Give the rule a name.');
  if (rule.actions.isEmpty) errors.add('Add at least one action.');
  if (rule.cooldownSeconds < 0) errors.add('Cooldown cannot be negative.');

  final t = rule.trigger;
  if (t.scope == TriggerScope.anySpeech && !t.hasCondition) {
    errors.add('Add a phrase or pattern, otherwise this rule would fire on everything you say.');
  }
  if (t.pattern != null && t.pattern!.isNotEmpty) {
    try {
      RegExp(t.pattern!);
    } on FormatException {
      errors.add('The pattern is not a valid regular expression.');
    }
  }

  for (final a in rule.actions) {
    switch (a) {
      case WebhookAction():
        // Placeholders are resolved at fire time, so validate the static part.
        final probe = Uri.tryParse(a.url.replaceAll(RegExp(r'\{\{[^}]*\}\}'), 'x'));
        if (probe == null || !probe.hasScheme || probe.host.isEmpty || !(probe.scheme == 'http' || probe.scheme == 'https')) {
          errors.add('Webhook URL must start with https:// (or http:// for a device on your network).');
        } else if (probe.scheme == 'http' && !a.allowInsecureHttp) {
          errors.add('Plain http:// is blocked. Use https://, or allow insecure HTTP for a local device.');
        }
        if (!const {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}.contains(a.method.toUpperCase())) {
          errors.add('Unsupported HTTP method ${a.method}.');
        }
        for (final k in a.headers.keys) {
          if (!RegExp(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$").hasMatch(k)) {
            errors.add('Invalid header name "$k".');
          }
        }
      case NotifyAction():
        if (a.title.trim().isEmpty && a.body.trim().isEmpty) errors.add('A notification needs a title or text.');
      case NoteAction():
        if (a.template.trim().isEmpty) errors.add('A note action needs some text.');
    }
  }
  return errors;
}
