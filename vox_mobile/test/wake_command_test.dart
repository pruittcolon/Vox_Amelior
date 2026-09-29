import 'package:flutter_test/flutter_test.dart';
import 'package:vox_amelior_mobile/assistant/wake_command.dart';

void main() {
  final parser = WakeCommandParser(['hey vox', 'vox']);

  test('extracts the command after the wake phrase', () {
    expect(parser.extract('Hey Vox, what did Sam say about the plumber?'), 'what did Sam say about the plumber?');
    expect(parser.extract('vox turn off the lights'), 'turn off the lights');
    expect(parser.extract('OK Vox: remind me later'), 'remind me later');
  });

  test('returns an empty command for a bare wake phrase', () {
    expect(parser.extract('Hey Vox'), '');
    expect(parser.extract('hey, vox!'), '');
  });

  test('ignores speech that merely contains or resembles the phrase', () {
    expect(parser.extract('I saw a voxel engine today'), isNull);
    expect(parser.extract('tell hey vox to wait'), isNull);
    expect(parser.extract('what is the weather'), isNull);
  });

  test('no phrases means the assistant never triggers', () {
    expect(WakeCommandParser([]).extract('hey vox hi'), isNull);
    expect(WakeCommandParser(['  ']).extract('hey vox hi'), isNull);
  });

  test('special characters in phrases are matched literally', () {
    final p = WakeCommandParser(['hey (vox)']);
    expect(p.extract('hey (vox) hello'), 'hello');
  });
}
