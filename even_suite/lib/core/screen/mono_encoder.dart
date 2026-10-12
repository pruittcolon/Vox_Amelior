import 'dart:typed_data';

/// Turns a 576x136 grayscale frame into the 1-bit BMP the G1 shows
/// (same layout as the demo's images: 62-byte header, bottom-up rows,
/// palette 0 = white = lit, 1 = black = off).
class MonoEncoder {
  MonoEncoder._();

  static const int width = 576;
  static const int height = 136;
  static const int stride = width ~/ 8;
  static const int dataSize = stride * height;

  /// Like the demo's images: pixel data plus 2 padding bytes, counted in the
  /// header. The glasses' CRC appears to cover those 2 bytes too.
  static const int bmpSize = 62 + dataSize + 2;

  /// [gray]: width*height luminance bytes. Pixels outside the content box
  /// (x0,y0)-(x1,y1) are letterbox and stay off.
  ///
  /// [style] 'text': one sharp threshold (Otsu) - best for apps and text.
  /// 'photo': Floyd-Steinberg dithering - best for pictures.
  /// [invert] 'auto' darkens light screens (so the glasses do not glow
  /// white), 'on' always inverts, 'off' never.
  static MonoFrame encode(
    Uint8List gray, {
    int x0 = 0,
    int y0 = 0,
    int x1 = width,
    int y1 = height,
    String style = 'text',
    String invert = 'auto',
  }) {
    assert(gray.length == width * height);
    x0 = x0.clamp(0, width);
    x1 = x1.clamp(x0, width);
    y0 = y0.clamp(0, height);
    y1 = y1.clamp(y0, height);

    final hist = List<int>.filled(256, 0);
    var sum = 0;
    for (var y = y0; y < y1; y++) {
      final row = y * width;
      for (var x = x0; x < x1; x++) {
        final g = gray[row + x];
        hist[g]++;
        sum += g;
      }
    }
    final count = (x1 - x0) * (y1 - y0);
    final mean = count == 0 ? 0 : sum ~/ count;
    final flip = invert == 'on' || (invert == 'auto' && mean > 128);

    final lit = Uint8List(width * height);
    if (count > 0) {
      if (style == 'photo') {
        _dither(gray, lit, x0, y0, x1, y1, flip);
      } else {
        final t = otsu(hist, count);
        for (var y = y0; y < y1; y++) {
          final row = y * width;
          for (var x = x0; x < x1; x++) {
            final g = gray[row + x];
            lit[row + x] = (flip ? g <= t : g > t) ? 1 : 0;
          }
        }
      }
    }
    return MonoFrame(bmp(lit), lit, flipped: flip);
  }

  /// Otsu's threshold: the gray level that best splits the histogram in two.
  static int otsu(List<int> hist, int total) {
    var sumAll = 0;
    for (var i = 0; i < 256; i++) {
      sumAll += i * hist[i];
    }
    var sumB = 0, wB = 0, best = 0.0, threshold = 127;
    for (var t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB == 0) continue;
      final wF = total - wB;
      if (wF == 0) break;
      sumB += t * hist[t];
      final mB = sumB / wB;
      final mF = (sumAll - sumB) / wF;
      final between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) {
        best = between;
        threshold = t;
      }
    }
    return threshold;
  }

  static void _dither(Uint8List gray, Uint8List lit, int x0, int y0, int x1, int y1, bool flip) {
    final w = x1 - x0;
    var cur = Int16List(w + 2);
    var next = Int16List(w + 2);
    for (var y = y0; y < y1; y++) {
      final row = y * width;
      for (var i = 0; i < w; i++) {
        final g = gray[row + x0 + i];
        cur[i + 1] += flip ? 255 - g : g;
      }
      for (var i = 0; i < w; i++) {
        final v = cur[i + 1];
        final on = v >= 128;
        lit[row + x0 + i] = on ? 1 : 0;
        final err = v - (on ? 255 : 0);
        cur[i + 2] += err * 7 ~/ 16;
        next[i] += err * 3 ~/ 16;
        next[i + 1] += err * 5 ~/ 16;
        next[i + 2] += err ~/ 16;
      }
      final t = cur;
      cur = next;
      next = t..fillRange(0, t.length, 0);
    }
  }

  /// 1-bit BMP from lit flags (1 = lit).
  static Uint8List bmp(Uint8List lit) {
    final out = Uint8List(bmpSize);
    final bd = ByteData.view(out.buffer);
    out[0] = 0x42; // B
    out[1] = 0x4D; // M
    bd.setUint32(2, bmpSize, Endian.little);
    bd.setUint32(10, 62, Endian.little);
    bd.setUint32(14, 40, Endian.little);
    bd.setInt32(18, width, Endian.little);
    bd.setInt32(22, height, Endian.little);
    bd.setUint16(26, 1, Endian.little);
    bd.setUint16(28, 1, Endian.little);
    bd.setUint32(34, dataSize + 2, Endian.little);
    bd.setInt32(38, 2834, Endian.little); // 72 dpi, as in the demo files
    bd.setInt32(42, 2834, Endian.little);
    out.setRange(54, 62, const [0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00]);
    var o = 62;
    for (var y = height - 1; y >= 0; y--) {
      final row = y * width;
      for (var xb = 0; xb < stride; xb++) {
        var b = 0;
        final base = row + xb * 8;
        for (var bit = 0; bit < 8; bit++) {
          b = (b << 1) | (lit[base + bit] == 1 ? 0 : 1);
        }
        out[o++] = b;
      }
    }
    return out;
  }
}

class MonoFrame {
  final Uint8List bmp;
  final Uint8List lit;
  final bool flipped;
  const MonoFrame(this.bmp, this.lit, {this.flipped = false});

  /// Share of lit pixels (a bright frame is tiring to look at).
  double get litShare {
    var n = 0;
    for (final v in lit) {
      n += v;
    }
    return lit.isEmpty ? 0 : n / lit.length;
  }
}
