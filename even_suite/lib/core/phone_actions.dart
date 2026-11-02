import 'dart:convert';

import 'package:flutter/foundation.dart';

import 'system.dart';

/// Something to do on the phone: set a reminder, a timer, an alarm, the
/// flashlight, music, volume, open an app, call or text someone, directions,
/// a web search.
class PhoneAction {
  const PhoneAction(this.kind, [this.args = const {}]);

  final String kind;
  final Map<String, Object?> args;

  Map<String, Object?> toMap() => {'do': kind, ...args};

  @override
  String toString() => 'PhoneAction($kind, $args)';

  static final _tag = RegExp(r'\[\[phone:(\{.*?\})\]\]', dotAll: true);

  /// The actions Claude asked for in its reply (one `[[phone:{...}]]` each).
  static List<PhoneAction> extract(String text) {
    final out = <PhoneAction>[];
    for (final m in _tag.allMatches(text)) {
      try {
        final map = jsonDecode(m.group(1)!);
        if (map is Map && map['do'] is String) {
          out.add(PhoneAction('${map['do']}', {
            for (final e in map.entries)
              if (e.key != 'do') '${e.key}': e.value,
          }));
        }
      } catch (_) {}
    }
    return out;
  }

  /// [text] without the action tags (also a half-written one at the end,
  /// while the reply is still streaming in).
  static String strip(String text) {
    var t = text.replaceAll(_tag, '').replaceAll('[[full]]', '');
    final open = t.lastIndexOf('[[');
    if (open >= 0 && !t.substring(open).contains(']]')) t = t.substring(0, open);
    return t.replaceAll(RegExp(r'\n{3,}'), '\n\n').trim();
  }

  // ------------------------------------------------------- quick commands

  static int? _num(String s) =>
      int.tryParse(s) ??
      const {
        'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9, 'ten': 10, 'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40, 'forty-five': 45, 'sixty': 60,
      }[s];

  /// Said plainly, these happen at once, without asking Claude: the
  /// flashlight, music, volume, a timer, "remind me in 10 minutes to ...".
  static PhoneAction? parseLocal(String said, DateTime now) {
    final t = said.toLowerCase().replaceAll(RegExp(r"[^a-z0-9' -]"), ' ').replaceAll(RegExp(r'\s+'), ' ').trim();
    if (RegExp(r'^(turn )?(the )?(flash ?light|torch)( on)?$|^turn on (the )?(flash ?light|torch)$').hasMatch(t)) {
      return const PhoneAction('torch', {'on': true});
    }
    if (RegExp(r'^(turn )?(the )?(flash ?light|torch) off$|^turn off (the )?(flash ?light|torch)$').hasMatch(t)) {
      return const PhoneAction('torch', {'on': false});
    }
    if (RegExp(r'^(pause|stop) (the )?(music|song|playback)$|^pause$').hasMatch(t)) {
      return const PhoneAction('media', {'key': 'pause'});
    }
    if (RegExp(r'^(play|resume) (the )?(music|song)$|^resume$').hasMatch(t)) {
      return const PhoneAction('media', {'key': 'play'});
    }
    if (RegExp(r'^(next|skip)( (song|track))?$|^skip (this )?(song|track)$').hasMatch(t)) {
      return const PhoneAction('media', {'key': 'next'});
    }
    if (RegExp(r'^(previous|last|go back a) (song|track)$').hasMatch(t)) {
      return const PhoneAction('media', {'key': 'previous'});
    }
    if (RegExp(r'^(turn (the )?)?volume up$|^louder$|^turn it up$').hasMatch(t)) {
      return const PhoneAction('volume', {'dir': 'up'});
    }
    if (RegExp(r'^(turn (the )?)?volume down$|^quieter$|^turn it down$').hasMatch(t)) {
      return const PhoneAction('volume', {'dir': 'down'});
    }
    // "set a timer for 5 minutes", "timer 90 seconds", "5 minute timer"
    final timer = RegExp(r'^(?:set (?:a |the )?)?(?:timer (?:for )?([a-z0-9-]+) (second|minute|hour)s?|([a-z0-9-]+) (second|minute|hour)s? timer)$')
        .firstMatch(t);
    if (timer != null) {
      final n = _num(timer.group(1) ?? timer.group(3) ?? '');
      final unit = timer.group(2) ?? timer.group(4) ?? 'minute';
      if (n != null && n > 0) {
        final secs = n * (unit == 'hour' ? 3600 : unit == 'second' ? 1 : 60);
        return PhoneAction('remind', {
          'at': now.add(Duration(seconds: secs)).millisecondsSinceEpoch,
          'text': 'Timer done ($n $unit${n == 1 ? '' : 's'})',
        });
      }
    }
    // "remind me in 10 minutes to call mom"
    final rem = RegExp(r'^remind me in ([a-z0-9-]+) (second|minute|hour)s? (?:to |that |about )?(.+)$').firstMatch(t);
    if (rem != null) {
      final n = _num(rem.group(1)!);
      if (n != null && n > 0) {
        final unit = rem.group(2)!;
        final secs = n * (unit == 'hour' ? 3600 : unit == 'second' ? 1 : 60);
        return PhoneAction('remind', {
          'at': now.add(Duration(seconds: secs)).millisecondsSinceEpoch,
          'text': _cap(rem.group(3)!),
        });
      }
    }
    return null;
  }

  static String _cap(String s) => s.isEmpty ? s : '${s[0].toUpperCase()}${s.substring(1)}';

  /// For Claude's instructions: what it may ask the phone to do. (No clock
  /// time in here: Agent Hub restarts Claude when its instructions change.)
  static const primer = 'You can act on the user\'s Android phone. When they ask for one of these, add a line at the very end '
        'of your reply, exactly like [[phone:{"do":"remind","at":"2026-10-28T15:00","text":"Call the dentist"}]] '
        '(JSON, one line per action, never shown to them). Actions:\n'
        '- remind: "at" (local time, YYYY-MM-DDTHH:MM) or "in" (seconds from now), "text" - reminders and timers\n'
        '- alarm: "hour" (0-23), "minute", "label" - a Clock app alarm (to wake up)\n'
        '- torch: "on" true/false · media: "key" play/pause/next/previous · volume: "dir" up/down/mute\n'
        '- open: "app" (its name) · navigate: "place" · search: "query"\n'
        '- call: "to" (contact name or number) · text: "to", "body" (opens the message; they tap send)\n'
        'For "in 10 minutes" use "in"; for a clock time use "at" (a time already past today means tomorrow). '
        'Your visible reply then just confirms in a few words ("Reminder set for 3 PM."). Use actions only when asked.';
}

/// Runs [PhoneAction]s through Android and says what happened.
class PhoneActions {
  PhoneActions(this.system);
  final SystemBridge system;

  /// Does it; returns a short line for the glasses ("Reminder set for 3:00 PM").
  Future<String> run(PhoneAction a, {DateTime? now}) async {
    final at = DateTime.now();
    final args = Map<String, Object?>.of(a.toMap());
    if (a.kind == 'remind') {
      final when = _when(a.args, now ?? at);
      if (when == null) return 'When should I remind you?';
      args['at'] = when.millisecondsSinceEpoch;
      args['id'] ??= 'r${when.millisecondsSinceEpoch}';
      args.remove('in');
    }
    final r = await system.phone(args);
    if (r['ok'] != true) return 'Could not: ${r['error'] ?? 'not allowed'}';
    if (a.kind == 'remind') {
      final when = DateTime.fromMillisecondsSinceEpoch(args['at'] as int);
      final text = '${a.args['text'] ?? ''}';
      return text.startsWith('Timer done') ? 'Timer set · ${_clock(when)}' : 'Reminder set · ${_clock(when)}';
    }
    final say = '${r['say'] ?? ''}';
    if (r['note'] == 'background') return '$say (allow "Display over other apps" so it opens in your pocket)';
    return say.isEmpty ? 'Done' : say;
  }

  static DateTime? _when(Map<String, Object?> args, DateTime now) {
    final inn = args['in'];
    if (inn is num) return now.add(Duration(seconds: inn.round()));
    final at = args['at'];
    DateTime? t;
    if (at is int) t = DateTime.fromMillisecondsSinceEpoch(at);
    if (at is String) t = DateTime.tryParse(at);
    if (t == null) return null;
    // Claude may not know the time of day: a clock time already past is the next one.
    var next = t;
    while (next.isBefore(now.subtract(const Duration(minutes: 1)))) {
      next = next.add(const Duration(days: 1));
    }
    return next;
  }

  @visibleForTesting
  static DateTime? whenFor(Map<String, Object?> args, DateTime now) => _when(args, now);

  static String _clock(DateTime t) {
    final h = t.hour % 12 == 0 ? 12 : t.hour % 12;
    return '$h:${t.minute.toString().padLeft(2, '0')} ${t.hour < 12 ? 'AM' : 'PM'}';
  }
}
