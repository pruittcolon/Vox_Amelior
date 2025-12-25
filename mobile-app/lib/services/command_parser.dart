/// Parses natural language voice commands into structured data.
///
/// Supports:
/// - Timers: "set timer for 5 minutes", "timer 30 seconds", "timer for two hours"
/// - Alarms: "set alarm for 7 AM", "wake me up at 6:30"
/// - Calls: "call mom", "call 555-1234"
/// - Messages: "text john hello", "message mom"
/// - Search: "search for weather", "google restaurants"
/// - Navigation: "navigate to downtown", "directions to home"
/// - Reminders: "remind me to take medication"
class CommandParser {
  /// Word numbers to digit mapping
  static const Map<String, int> _wordNumbers = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    // Common combinations
    'twenty one': 21, 'twenty two': 22, 'twenty three': 23,
    'twenty four': 24, 'twenty five': 25, 'twenty six': 26,
    'twenty seven': 27, 'twenty eight': 28, 'twenty nine': 29,
    'thirty one': 31, 'thirty two': 32, 'thirty three': 33,
    'thirty four': 34, 'thirty five': 35, 'forty five': 45,
    'a': 1, 'an': 1, // "a minute", "an hour"
  };

  /// Parse a word number to an integer, or return null if not recognized.
  static int? _parseWordNumber(String word) {
    final lower = word.toLowerCase().trim();
    // Check for digit first
    final digit = int.tryParse(lower);
    if (digit != null) return digit;
    // Check word numbers
    return _wordNumbers[lower];
  }

  /// Parse a transcript and return a structured command, or null if no match.
  ///
  /// The parser tries each pattern in order and returns the first match.
  /// If no pattern matches, returns null (caller should fall back to Google Assistant).
  static ParsedCommand? parse(String transcript) {
    // Strip trailing punctuation that speech-to-text might add
    var lower = transcript.toLowerCase().trim();
    lower = lower.replaceAll(RegExp(r'[.?!,]+$'), '').trim();

    // Skip very short transcripts
    if (lower.length < 3) return null;

    // === TIMER PATTERNS ===
    // "set timer for 5 minutes", "timer 10 seconds", "set a timer for 1 hour"
    // Also supports: "timer for two hours", "set timer for twenty minutes"
    
    // Pattern for digits
    final timerDigitMatch = RegExp(
      r'(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (timerDigitMatch != null) {
      final value = int.parse(timerDigitMatch.group(1)!);
      final unit = timerDigitMatch.group(2)!.toLowerCase();
      final seconds = _toSeconds(value, unit);
      return ParsedCommand.timer(seconds: seconds, originalText: transcript);
    }
    
    // Pattern for word numbers: "timer for two hours", "set timer for twenty minutes"
    final timerWordMatch = RegExp(
      r'(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?([a-z]+(?:\s+[a-z]+)?)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (timerWordMatch != null) {
      final wordNum = timerWordMatch.group(1)!;
      final value = _parseWordNumber(wordNum);
      if (value != null) {
        final unit = timerWordMatch.group(2)!.toLowerCase();
        final seconds = _toSeconds(value, unit);
        return ParsedCommand.timer(seconds: seconds, originalText: transcript);
      }
    }

    // === ALARM PATTERNS ===
    // "set alarm for 7 AM", "set alarm for 7:30", "wake me up at 6:30 PM"
    final alarmPatterns = [
      RegExp(
          r'(?:set\s+)?(?:an?\s+)?alarm\s+(?:for\s+)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?',
          caseSensitive: false),
      RegExp(
          r'wake\s+(?:me\s+)?up\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?',
          caseSensitive: false),
    ];

    for (final pattern in alarmPatterns) {
      final alarmMatch = pattern.firstMatch(lower);
      if (alarmMatch != null) {
        var hour = int.parse(alarmMatch.group(1)!);
        final minute = alarmMatch.group(2) != null
            ? int.parse(alarmMatch.group(2)!)
            : 0;
        final period =
            alarmMatch.group(3)?.replaceAll('.', '').toLowerCase();

        // Convert to 24-hour format
        if (period == 'pm' && hour < 12) hour += 12;
        if (period == 'am' && hour == 12) hour = 0;

        return ParsedCommand.alarm(
            hour: hour, minute: minute, originalText: transcript);
      }
    }

    // === CALL PATTERNS ===
    // "call mom", "call 555-1234", "phone john"
    final callMatch = RegExp(
      r'(?:call|phone|dial)\s+(.+)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (callMatch != null) {
      final target = callMatch.group(1)!.trim();
      if (target.isNotEmpty) {
        return ParsedCommand.call(target: target, originalText: transcript);
      }
    }

    // === MESSAGE PATTERNS ===
    // "text john hello there", "message mom", "send message to dad"
    final messageMatch = RegExp(
      r'(?:text|message|sms)\s+(\w+)\s*(.*)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (messageMatch != null) {
      final recipient = messageMatch.group(1)!;
      final content = messageMatch.group(2)?.trim();
      return ParsedCommand.message(
          recipient: recipient,
          content: content?.isNotEmpty == true ? content : null,
          originalText: transcript);
    }

    // === SEARCH PATTERNS ===
    // "search for weather", "google restaurants nearby", "look up recipes"
    final searchMatch = RegExp(
      r'(?:search\s+(?:for\s+)?|google\s+|look\s+up\s+|find\s+)(.+)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (searchMatch != null) {
      final query = searchMatch.group(1)!.trim();
      if (query.isNotEmpty) {
        return ParsedCommand.search(query: query, originalText: transcript);
      }
    }

    // === NAVIGATION PATTERNS ===
    // "navigate to downtown", "directions to home", "take me to the store"
    final navMatch = RegExp(
      r'(?:navigate\s+to|directions\s+to|take\s+me\s+to|go\s+to|drive\s+to)\s+(.+)',
      caseSensitive: false,
    ).firstMatch(lower);

    if (navMatch != null) {
      final destination = navMatch.group(1)!.trim();
      if (destination.isNotEmpty) {
        return ParsedCommand.navigation(
            destination: destination, originalText: transcript);
      }
    }

    // === REMINDER PATTERNS ===
    // "remind me to take medication", "reminder to call mom at 3"
    final reminderMatch = RegExp(
      r'remind(?:er)?\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+(?:at|in)\s+(.+))?$',
      caseSensitive: false,
    ).firstMatch(lower);

    if (reminderMatch != null) {
      final task = reminderMatch.group(1)!.trim();
      final when = reminderMatch.group(2)?.trim();
      if (task.isNotEmpty) {
        return ParsedCommand.reminder(
            task: task,
            when: when?.isNotEmpty == true ? when : null,
            originalText: transcript);
      }
    }

    // No pattern matched - return null to fall back to Google Assistant
    return null;
  }

  /// Convert time value to seconds based on unit.
  static int _toSeconds(int value, String unit) {
    if (unit.startsWith('sec')) return value;
    if (unit.startsWith('min')) return value * 60;
    if (unit.startsWith('hour') || unit.startsWith('hr')) return value * 3600;
    return value;
  }
}

/// Represents a parsed voice command with its type and parameters.
class ParsedCommand {
  /// The type of command (timer, alarm, call, etc.)
  final CommandType type;

  /// Command-specific parameters
  final Map<String, dynamic> params;

  /// The original transcript text
  final String originalText;

  ParsedCommand._(this.type, this.params, this.originalText);

  /// Create a timer command.
  factory ParsedCommand.timer(
          {required int seconds, required String originalText}) =>
      ParsedCommand._(CommandType.timer, {'seconds': seconds}, originalText);

  /// Create an alarm command.
  factory ParsedCommand.alarm(
          {required int hour,
          required int minute,
          required String originalText}) =>
      ParsedCommand._(
          CommandType.alarm, {'hour': hour, 'minute': minute}, originalText);

  /// Create a call command.
  factory ParsedCommand.call(
          {required String target, required String originalText}) =>
      ParsedCommand._(CommandType.call, {'target': target}, originalText);

  /// Create a message command.
  factory ParsedCommand.message(
          {required String recipient,
          String? content,
          required String originalText}) =>
      ParsedCommand._(CommandType.message,
          {'recipient': recipient, 'content': content}, originalText);

  /// Create a search command.
  factory ParsedCommand.search(
          {required String query, required String originalText}) =>
      ParsedCommand._(CommandType.search, {'query': query}, originalText);

  /// Create a navigation command.
  factory ParsedCommand.navigation(
          {required String destination, required String originalText}) =>
      ParsedCommand._(
          CommandType.navigation, {'destination': destination}, originalText);

  /// Create a reminder command.
  factory ParsedCommand.reminder(
          {required String task, String? when, required String originalText}) =>
      ParsedCommand._(
          CommandType.reminder, {'task': task, 'when': when}, originalText);

  @override
  String toString() => 'ParsedCommand($type, $params)';
}

/// Types of voice commands that can be parsed.
enum CommandType {
  /// Countdown timer (e.g., "set timer for 5 minutes")
  timer,

  /// Alarm (e.g., "set alarm for 7 AM")
  alarm,

  /// Phone call (e.g., "call mom")
  call,

  /// Text message (e.g., "text john hello")
  message,

  /// Web search (e.g., "search for weather")
  search,

  /// Navigation (e.g., "navigate to downtown")
  navigation,

  /// Reminder (e.g., "remind me to take medication")
  reminder,
}
