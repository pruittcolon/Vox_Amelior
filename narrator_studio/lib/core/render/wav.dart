/// 16-bit mono WAV reading and (streaming) writing.
library;

import 'dart:io';
import 'dart:typed_data';

import 'audio.dart';

Uint8List wavHeader(int sampleRate, int dataBytes) {
  final b = ByteData(44);
  void ascii(int at, String s) {
    for (var i = 0; i < 4; i++) {
      b.setUint8(at + i, s.codeUnitAt(i));
    }
  }

  ascii(0, 'RIFF');
  b.setUint32(4, 36 + dataBytes, Endian.little);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  b.setUint32(16, 16, Endian.little);
  b.setUint16(20, 1, Endian.little); // PCM
  b.setUint16(22, 1, Endian.little); // mono
  b.setUint32(24, sampleRate, Endian.little);
  b.setUint32(28, sampleRate * 2, Endian.little);
  b.setUint16(32, 2, Endian.little);
  b.setUint16(34, 16, Endian.little);
  ascii(36, 'data');
  b.setUint32(40, dataBytes, Endian.little);
  return b.buffer.asUint8List();
}

Uint8List encodeWav(Float32List samples, int sampleRate) {
  final data = toInt16Bytes(samples);
  return Uint8List.fromList([...wavHeader(sampleRate, data.length), ...data]);
}

/// Reads PCM16 / float32 WAV (any channel count; mixed down to mono).
Pcm decodeWav(Uint8List bytes) {
  final b = ByteData.sublistView(bytes);
  String id(int at) => String.fromCharCodes(bytes.sublist(at, at + 4));
  if (bytes.length < 12 || id(0) != 'RIFF' || id(8) != 'WAVE') {
    throw const FormatException('Not a WAV file');
  }
  var at = 12;
  var format = 1, channels = 1, rate = 24000, bits = 16;
  while (at + 8 <= bytes.length) {
    final chunk = id(at);
    final size = b.getUint32(at + 4, Endian.little);
    final body = at + 8;
    if (chunk == 'fmt ') {
      format = b.getUint16(body, Endian.little);
      channels = b.getUint16(body + 2, Endian.little);
      rate = b.getUint32(body + 4, Endian.little);
      bits = b.getUint16(body + 14, Endian.little);
      if (format == 0xFFFE && size >= 26) format = b.getUint16(body + 24, Endian.little);
    } else if (chunk == 'data') {
      final end = (body + size).clamp(body, bytes.length);
      final frameBytes = channels * bits ~/ 8;
      final frames = (end - body) ~/ frameBytes;
      final out = Float32List(frames);
      for (var f = 0; f < frames; f++) {
        var sum = 0.0;
        for (var c = 0; c < channels; c++) {
          final o = body + f * frameBytes + c * bits ~/ 8;
          sum += switch ((format, bits)) {
            (1, 16) => b.getInt16(o, Endian.little) / 32768.0,
            (1, 8) => (b.getUint8(o) - 128) / 128.0,
            (1, 24) => ((b.getUint8(o) | b.getUint8(o + 1) << 8 | b.getInt8(o + 2) << 16)) / 8388608.0,
            (1, 32) => b.getInt32(o, Endian.little) / 2147483648.0,
            (3, 32) => b.getFloat32(o, Endian.little),
            _ => throw FormatException('Unsupported WAV format $format/$bits-bit'),
          };
        }
        out[f] = sum / channels;
      }
      return Pcm(out, rate);
    }
    at = body + size + (size & 1);
  }
  throw const FormatException('WAV file has no audio data');
}

/// Appends audio to a WAV file as it is rendered; [close] fixes the header.
class WavWriter {
  WavWriter._(this.file, this._raf, this.sampleRate);

  static Future<WavWriter> create(String path, int sampleRate) async {
    final file = File(path);
    await file.parent.create(recursive: true);
    final raf = await file.open(mode: FileMode.write);
    await raf.writeFrom(wavHeader(sampleRate, 0));
    return WavWriter._(file, raf, sampleRate);
  }

  final File file;
  final RandomAccessFile _raf;
  final int sampleRate;
  int _bytes = 0;

  int get frames => _bytes ~/ 2;
  double get seconds => frames / sampleRate;

  Future<void> add(Float32List samples) async {
    final data = toInt16Bytes(samples);
    await _raf.writeFrom(data);
    _bytes += data.length;
  }

  Future<void> close() async {
    await _raf.setPosition(0);
    await _raf.writeFrom(wavHeader(sampleRate, _bytes));
    await _raf.close();
  }
}
