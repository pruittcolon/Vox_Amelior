import 'package:flutter_test/flutter_test.dart';
import 'package:demo_ai_even/services/command_parser.dart';

void main() {
  group('CommandParser', () {
    group('Timer Commands', () {
      test('parses "set timer for 5 minutes"', () {
        final result = CommandParser.parse('set timer for 5 minutes');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 300); // 5 * 60
      });

      test('parses "timer 30 seconds"', () {
        final result = CommandParser.parse('timer 30 seconds');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 30);
      });

      test('parses "set a timer for 1 hour"', () {
        final result = CommandParser.parse('set a timer for 1 hour');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 3600); // 1 * 3600
      });

      test('parses "timer 2 hrs"', () {
        final result = CommandParser.parse('timer 2 hrs');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 7200); // 2 * 3600
      });

      test('parses "set timer for 10 min"', () {
        final result = CommandParser.parse('set timer for 10 min');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 600); // 10 * 60
      });
      
      // Word number tests
      test('parses "set timer for two hours"', () {
        final result = CommandParser.parse('set timer for two hours');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 7200); // 2 * 3600
      });
      
      test('parses "timer for twenty minutes"', () {
        final result = CommandParser.parse('timer for twenty minutes');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 1200); // 20 * 60
      });
      
      test('parses "set timer for a minute"', () {
        final result = CommandParser.parse('set timer for a minute');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 60); // 1 * 60
      });
      
      test('parses "timer five seconds"', () {
        final result = CommandParser.parse('timer five seconds');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 5);
      });
    });

    group('Alarm Commands', () {
      test('parses "set alarm for 7 AM"', () {
        final result = CommandParser.parse('set alarm for 7 AM');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 7);
        expect(result.params['minute'], 0);
      });

      test('parses "set alarm for 7:30 PM"', () {
        final result = CommandParser.parse('set alarm for 7:30 PM');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 19); // 7 PM = 19
        expect(result.params['minute'], 30);
      });

      test('parses "alarm for 6:15"', () {
        final result = CommandParser.parse('alarm for 6:15');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 6);
        expect(result.params['minute'], 15);
      });

      test('parses "wake me up at 8"', () {
        final result = CommandParser.parse('wake me up at 8');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 8);
        expect(result.params['minute'], 0);
      });

      test('parses "set an alarm for 12 pm"', () {
        final result = CommandParser.parse('set an alarm for 12 pm');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 12); // 12 PM stays 12
        expect(result.params['minute'], 0);
      });
    });

    group('Call Commands', () {
      test('parses "call mom"', () {
        final result = CommandParser.parse('call mom');
        expect(result, isNotNull);
        expect(result!.type, CommandType.call);
        expect(result.params['target'], 'mom');
      });

      test('parses "call 555-1234"', () {
        final result = CommandParser.parse('call 555-1234');
        expect(result, isNotNull);
        expect(result!.type, CommandType.call);
        expect(result.params['target'], '555-1234');
      });

      test('parses "phone john"', () {
        final result = CommandParser.parse('phone john');
        expect(result, isNotNull);
        expect(result!.type, CommandType.call);
        expect(result.params['target'], 'john');
      });

      test('parses "dial the office"', () {
        final result = CommandParser.parse('dial the office');
        expect(result, isNotNull);
        expect(result!.type, CommandType.call);
        expect(result.params['target'], 'the office');
      });
    });

    group('Message Commands', () {
      test('parses "text john hello"', () {
        final result = CommandParser.parse('text john hello');
        expect(result, isNotNull);
        expect(result!.type, CommandType.message);
        expect(result.params['recipient'], 'john');
        expect(result.params['content'], 'hello');
      });

      test('parses "message mom"', () {
        final result = CommandParser.parse('message mom');
        expect(result, isNotNull);
        expect(result!.type, CommandType.message);
        expect(result.params['recipient'], 'mom');
      });

      test('parses "sms dad on my way"', () {
        final result = CommandParser.parse('sms dad on my way');
        expect(result, isNotNull);
        expect(result!.type, CommandType.message);
        expect(result.params['recipient'], 'dad');
        expect(result.params['content'], 'on my way');
      });
    });

    group('Search Commands', () {
      test('parses "search for weather"', () {
        final result = CommandParser.parse('search for weather');
        expect(result, isNotNull);
        expect(result!.type, CommandType.search);
        expect(result.params['query'], 'weather');
      });

      test('parses "google restaurants nearby"', () {
        final result = CommandParser.parse('google restaurants nearby');
        expect(result, isNotNull);
        expect(result!.type, CommandType.search);
        expect(result.params['query'], 'restaurants nearby');
      });

      test('parses "look up recipes"', () {
        final result = CommandParser.parse('look up recipes');
        expect(result, isNotNull);
        expect(result!.type, CommandType.search);
        expect(result.params['query'], 'recipes');
      });

      test('parses "find coffee shops"', () {
        final result = CommandParser.parse('find coffee shops');
        expect(result, isNotNull);
        expect(result!.type, CommandType.search);
        expect(result.params['query'], 'coffee shops');
      });
    });

    group('Navigation Commands', () {
      test('parses "navigate to downtown"', () {
        final result = CommandParser.parse('navigate to downtown');
        expect(result, isNotNull);
        expect(result!.type, CommandType.navigation);
        expect(result.params['destination'], 'downtown');
      });

      test('parses "directions to home"', () {
        final result = CommandParser.parse('directions to home');
        expect(result, isNotNull);
        expect(result!.type, CommandType.navigation);
        expect(result.params['destination'], 'home');
      });

      test('parses "take me to the store"', () {
        final result = CommandParser.parse('take me to the store');
        expect(result, isNotNull);
        expect(result!.type, CommandType.navigation);
        expect(result.params['destination'], 'the store');
      });

      test('parses "go to 123 main street"', () {
        final result = CommandParser.parse('go to 123 main street');
        expect(result, isNotNull);
        expect(result!.type, CommandType.navigation);
        expect(result.params['destination'], '123 main street');
      });
    });

    group('Reminder Commands', () {
      test('parses "remind me to take medication"', () {
        final result = CommandParser.parse('remind me to take medication');
        expect(result, isNotNull);
        expect(result!.type, CommandType.reminder);
        expect(result.params['task'], 'take medication');
      });

      test('parses "reminder to buy groceries at 3"', () {
        final result = CommandParser.parse('reminder to buy groceries at 3');
        expect(result, isNotNull);
        expect(result!.type, CommandType.reminder);
        expect(result.params['task'], 'buy groceries');
        expect(result.params['when'], '3');
      });
    });

    group('No Match (Fallback)', () {
      test('returns null for "what is the weather"', () {
        final result = CommandParser.parse('what is the weather');
        expect(result, isNull);
      });

      test('returns null for "tell me a joke"', () {
        final result = CommandParser.parse('tell me a joke');
        expect(result, isNull);
      });

      test('returns null for "hello"', () {
        final result = CommandParser.parse('hello');
        expect(result, isNull);
      });

      test('returns null for empty string', () {
        final result = CommandParser.parse('');
        expect(result, isNull);
      });

      test('returns null for very short input', () {
        final result = CommandParser.parse('hi');
        expect(result, isNull);
      });
    });
    
    group('Punctuation Handling', () {
      test('parses "set timer for two hours." with trailing period', () {
        final result = CommandParser.parse('set timer for two hours.');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 7200);
      });
      
      test('parses "set timer for 5 minutes?" with trailing question mark', () {
        final result = CommandParser.parse('set timer for 5 minutes?');
        expect(result, isNotNull);
        expect(result!.type, CommandType.timer);
        expect(result.params['seconds'], 300);
      });
      
      test('parses "call mom." with trailing period', () {
        final result = CommandParser.parse('call mom.');
        expect(result, isNotNull);
        expect(result!.type, CommandType.call);
        expect(result.params['target'], 'mom');
      });
      
      test('parses "set alarm for 7 AM!" with trailing exclamation', () {
        final result = CommandParser.parse('set alarm for 7 AM!');
        expect(result, isNotNull);
        expect(result!.type, CommandType.alarm);
        expect(result.params['hour'], 7);
      });
    });
  });
}
