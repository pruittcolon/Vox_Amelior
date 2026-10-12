/// Spoken commands: what you can say after "Claude", and the words that end
/// a message ("send", "cancel").
enum VoiceCommand {
  send,
  cancel,
  scrollUp,
  scrollDown,
  top,
  bottom,
  repeat,
  hide,
  stop,
  stopListening,
  stopReply,
  toTerminal,
  toChat,
  enter,
  escape,
  choice1,
  choice2,
  choice3,
}

/// The end of a message: [body] is what to send, [command] is send/cancel.
class Ending {
  final String body;
  final VoiceCommand? command;
  const Ending(this.body, this.command);

  @override
  String toString() => 'Ending("$body", $command)';
}

class VoiceCommands {
  VoiceCommands._();

  static const Map<VoiceCommand, List<String>> phrases = {
    VoiceCommand.send: [
      'send', 'send it', 'send that', 'send this', 'send now', 'send it now',
      'send message', 'send the message', 'send please', 'submit',
    ],
    VoiceCommand.cancel: [
      'cancel', 'cancel that', 'cancel it', 'never mind', 'nevermind',
      'scratch that', 'forget it', 'delete that', 'clear that',
    ],
    VoiceCommand.scrollUp: [
      'up', 'scroll up', 'page up', 'go up', 'move up', 'back', 'go back',
      'previous', 'previous page', 'page back',
    ],
    VoiceCommand.scrollDown: [
      'down', 'scroll down', 'page down', 'go down', 'move down', 'next',
      'next page', 'more', 'show more', 'continue',
    ],
    VoiceCommand.top: [
      'top', 'go to top', 'go to the top', 'scroll to top', 'scroll to the top',
      'first page', 'beginning', 'from the top',
    ],
    VoiceCommand.bottom: [
      'bottom', 'go to bottom', 'go to the bottom', 'scroll to bottom',
      'scroll to the bottom', 'end', 'latest', 'newest', 'final page',
    ],
    VoiceCommand.repeat: [
      'repeat', 'again', 'show again', 'show that again', 'show it again',
      'say that again', 'show reply', 'show the reply', 'show the answer',
    ],
    VoiceCommand.hide: [
      'hide', 'hide that', 'clear', 'clear screen', 'clear the screen',
      'clear glasses', 'clear the glasses', 'close', 'dismiss',
    ],
    VoiceCommand.stop: ['stop'],
    VoiceCommand.stopListening: [
      'stop listening', 'go to sleep', 'sleep', 'goodbye', 'good bye', 'bye',
      "that's all", 'thats all', 'turn off', 'stop listening now',
    ],
    VoiceCommand.stopReply: [
      'stop claude', 'stop answering', 'interrupt', 'cancel reply',
      'stop the reply', 'stop that',
    ],
    VoiceCommand.toTerminal: [
      'terminal', 'switch to terminal', 'terminal mode', 'go to terminal',
      'use terminal', 'use the terminal',
    ],
    VoiceCommand.toChat: [
      'chat', 'chat mode', 'switch to chat', 'go to chat', 'use chat',
    ],
    VoiceCommand.enter: ['enter', 'press enter', 'hit enter', 'return'],
    VoiceCommand.escape: ['escape', 'press escape', 'hit escape'],
    VoiceCommand.choice1: [
      'one', 'option one', 'press one', 'number one', 'yes', '1', 'option 1', 'press 1',
    ],
    VoiceCommand.choice2: [
      'two', 'option two', 'press two', 'number two', '2', 'option 2', 'press 2',
    ],
    VoiceCommand.choice3: [
      'three', 'option three', 'press three', 'number three', '3', 'option 3', 'press 3',
    ],
  };

  static final Map<String, VoiceCommand> _lookup = {
    for (final e in phrases.entries)
      for (final p in e.value) p: e.key,
  };

  static const _politeLead = ['please ', 'can you ', 'could you ', 'okay ', 'ok ', 'now '];
  static const _politeTail = [' please', ' now', ' thanks', ' thank you'];

  static String normalize(String text) {
    var t = text
        .toLowerCase()
        .replaceAll(RegExp(r"[^a-z0-9' ]"), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    var changed = true;
    while (changed) {
      changed = false;
      for (final p in _politeLead) {
        if (t.startsWith(p) && t.length > p.length) {
          t = t.substring(p.length);
          changed = true;
        }
      }
      for (final p in _politeTail) {
        if (t.endsWith(p) && t.length > p.length) {
          t = t.substring(0, t.length - p.length);
          changed = true;
        }
      }
    }
    return t;
  }

  /// A whole utterance that is exactly a command ("scroll down", "send").
  static VoiceCommand? parse(String text) {
    final t = normalize(text);
    if (t.isEmpty) return null;
    return _lookup[t];
  }

  static final _word = RegExp(r"[A-Za-z0-9']+");
  static const _connectors = {'and', 'then', 'okay', 'ok', 'now', 'please', 'so', 'just'};

  /// Words that make a trailing "send" part of the sentence ("what should I
  /// send", "how to send it") rather than the command.
  static const _sentenceWords = {
    'to', 'i', 'we', 'you', 'they', 'he', 'she', 'should', 'would',
    'could', 'can', 'will', 'did', 'do', 'not', "don't", 'dont', "didn't",
    'didnt', 'cannot', "can't", 'cant', 'must', 'might', 'may', "i'll", 'ill',
    "won't", 'wont', 'never', 'always',
  };

  /// Checks whether a message ends with "send" (or "cancel"). Looks at the
  /// last one to three words; [Ending.body] is the message without them.
  static Ending splitEnding(String text) {
    final words = _word.allMatches(text).toList();
    for (var n = 3; n >= 1; n--) {
      if (words.length < n) continue;
      final tail = words.sublist(words.length - n);
      final phrase = tail.map((m) => m[0]!.toLowerCase()).join(' ');
      final cmd = _lookup[phrase];
      if (cmd != VoiceCommand.send && cmd != VoiceCommand.cancel) continue;
      final before = words.length > n ? words[words.length - n - 1][0]!.toLowerCase() : null;
      if (cmd == VoiceCommand.send && before != null && _sentenceWords.contains(before)) {
        continue;
      }
      // A bare "cancel" only counts on its own ("please cancel my order" is a message).
      if (phrase == 'cancel' && words.length > 1) continue;
      // Drop "and"/"then"/"please" left in front of the ending.
      var cut = tail.first.start;
      var i = words.length - n - 1;
      while (i >= 0 && _connectors.contains(words[i][0]!.toLowerCase())) {
        cut = words[i].start;
        i--;
      }
      final body = text
          .substring(0, cut)
          .replaceFirst(RegExp(r'[\s,;:\-]+$'), '')
          .trim();
      return Ending(body, cmd);
    }
    return Ending(text.trim(), null);
  }
}
