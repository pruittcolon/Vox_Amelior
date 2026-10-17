/// Vector artwork drawn by the app: episode covers, voice portraits and a
/// live equalizer. Everything is painted, so it is sharp at any size and
/// screen density, and every episode gets its own cover from its title.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'theme.dart';

class CoverPalette {
  const CoverPalette(this.a, this.b, this.glow);
  final Color a;
  final Color b;
  final Color glow;
}

const coverPalettes = [
  CoverPalette(Color(0xFF2A2477), Color(0xFFD9577E), Color(0xFFFFC28A)), // dusk
  CoverPalette(Color(0xFF0B3C5D), Color(0xFF1FB5A8), Color(0xFFB8F2E6)), // lagoon
  CoverPalette(Color(0xFF7A1F2B), Color(0xFFF29E4C), Color(0xFFFFE3A3)), // ember
  CoverPalette(Color(0xFF12372A), Color(0xFF6FBF8E), Color(0xFFE3F6C5)), // forest
  CoverPalette(Color(0xFF3A1C71), Color(0xFF2EC4D6), Color(0xFFCFF7FF)), // aurora
  CoverPalette(Color(0xFF5C3A21), Color(0xFFE0A96D), Color(0xFFFFF0D6)), // amber
  CoverPalette(Color(0xFF111A33), Color(0xFF5B6CD6), Color(0xFFC9D2FF)), // midnight
  CoverPalette(Color(0xFF5E1742), Color(0xFFF08FA8), Color(0xFFFFE0E8)), // rose
];

int stableHash(String s) {
  var h = 0x811C9DC5;
  for (final c in s.codeUnits) {
    h = ((h ^ c) * 0x01000193) & 0x7FFFFFFF;
  }
  return h;
}

CoverPalette paletteFor(String key) => coverPalettes[stableHash(key) % coverPalettes.length];

/// An episode cover: a gradient with a motif for its style.
class CoverArt extends StatelessWidget {
  const CoverArt({super.key, required this.seed, this.style, this.size = 56, this.radius});

  /// Usually the title: the same title always gets the same cover.
  final String seed;

  /// podcast, interview, lecture, audiobook, story, news (or null).
  final String? style;
  final double size;
  final double? radius;

  @override
  Widget build(BuildContext context) {
    final r = radius ?? size * 0.22;
    return SizedBox.square(
      dimension: size,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(r),
        child: CustomPaint(painter: _CoverPainter(paletteFor(seed), style, stableHash(seed))),
      ),
    );
  }
}

class _CoverPainter extends CustomPainter {
  _CoverPainter(this.p, this.style, this.seed);
  final CoverPalette p;
  final String? style;
  final int seed;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width, h = size.height;
    final rect = Offset.zero & size;
    final rnd = math.Random(seed);
    canvas.drawRect(
      rect,
      Paint()
        ..shader = LinearGradient(
          begin: Alignment(-1 + rnd.nextDouble() * 0.6, -1),
          end: Alignment(1, 1 - rnd.nextDouble() * 0.6),
          colors: [p.a, Color.lerp(p.a, p.b, 0.55)!, p.b],
        ).createShader(rect),
    );
    // A soft light source.
    final light = Offset(w * (0.25 + rnd.nextDouble() * 0.5), h * (0.15 + rnd.nextDouble() * 0.3));
    canvas.drawCircle(
      light,
      w * 0.7,
      Paint()
        ..shader = RadialGradient(colors: [p.glow.withValues(alpha: 0.45), p.glow.withValues(alpha: 0)])
            .createShader(Rect.fromCircle(center: light, radius: w * 0.7)),
    );
    final line = Paint()
      ..color = Colors.white.withValues(alpha: 0.55)
      ..style = PaintingStyle.stroke
      ..strokeWidth = math.max(1, w * 0.022)
      ..strokeCap = StrokeCap.round;
    final fill = Paint()..color = Colors.white.withValues(alpha: 0.18);
    switch (style) {
      case 'podcast':
        // Sound waves from a microphone point.
        final c = Offset(w * 0.22, h * 0.78);
        for (var i = 1; i <= 5; i++) {
          canvas.drawArc(Rect.fromCircle(center: c, radius: w * 0.14 * i), -math.pi * 0.5, math.pi * 0.5, false,
              line..color = Colors.white.withValues(alpha: 0.62 - i * 0.09));
        }
        canvas.drawCircle(c, w * 0.06, Paint()..color = Colors.white.withValues(alpha: 0.85));
      case 'interview':
        canvas.drawCircle(Offset(w * 0.38, h * 0.58), w * 0.24, fill);
        canvas.drawCircle(Offset(w * 0.62, h * 0.48), w * 0.24, fill..color = Colors.white.withValues(alpha: 0.26));
        canvas.drawCircle(Offset(w * 0.38, h * 0.58), w * 0.24, line..color = Colors.white.withValues(alpha: 0.5));
        canvas.drawCircle(Offset(w * 0.62, h * 0.48), w * 0.24, line);
      case 'lecture':
        final grid = Paint()
          ..color = Colors.white.withValues(alpha: 0.12)
          ..strokeWidth = math.max(0.6, w * 0.008);
        for (var i = 1; i < 6; i++) {
          canvas.drawLine(Offset(w * i / 6, 0), Offset(w * i / 6, h), grid);
          canvas.drawLine(Offset(0, h * i / 6), Offset(w, h * i / 6), grid);
        }
        canvas.drawOval(Rect.fromCenter(center: Offset(w * 0.55, h * 0.55), width: w * 0.7, height: h * 0.32), line);
        canvas.drawCircle(Offset(w * 0.55, h * 0.55), w * 0.09, Paint()..color = Colors.white.withValues(alpha: 0.85));
      case 'story':
        for (var i = 0; i < 14; i++) {
          final o = Offset(rnd.nextDouble() * w, rnd.nextDouble() * h * 0.8);
          canvas.drawCircle(o, w * (0.006 + rnd.nextDouble() * 0.014), Paint()..color = Colors.white.withValues(alpha: 0.5 + rnd.nextDouble() * 0.4));
        }
        final moon = Offset(w * 0.68, h * 0.34);
        canvas.drawCircle(moon, w * 0.16, Paint()..color = Colors.white.withValues(alpha: 0.85));
        canvas.drawCircle(moon.translate(w * 0.07, -h * 0.04), w * 0.14, Paint()..color = Color.lerp(p.a, p.b, 0.4)!);
        final hill = Path()
          ..moveTo(0, h * 0.82)
          ..quadraticBezierTo(w * 0.35, h * 0.62, w * 0.7, h * 0.8)
          ..quadraticBezierTo(w * 0.86, h * 0.88, w, h * 0.78)
          ..lineTo(w, h)
          ..lineTo(0, h)
          ..close();
        canvas.drawPath(hill, Paint()..color = Colors.black.withValues(alpha: 0.22));
      case 'news':
        for (var i = 0; i < 4; i++) {
          final y = h * (0.3 + i * 0.14);
          final len = w * (i == 0 ? 0.62 : 0.42 + rnd.nextDouble() * 0.3);
          canvas.drawRRect(
            RRect.fromRectAndRadius(Rect.fromLTWH(w * 0.16, y, len, h * (i == 0 ? 0.075 : 0.045)), Radius.circular(w * 0.03)),
            Paint()..color = Colors.white.withValues(alpha: i == 0 ? 0.8 : 0.35),
          );
        }
        canvas.drawCircle(Offset(w * 0.8, h * 0.2), w * 0.05, Paint()..color = const Color(0xFFFF6B6B));
      case 'audiobook':
      default:
        // An open book.
        final c = Offset(w * 0.5, h * 0.62);
        final bw = w * 0.32, bh = h * 0.24;
        for (final side in [-1.0, 1.0]) {
          final page = Path()
            ..moveTo(c.dx, c.dy)
            ..quadraticBezierTo(c.dx + side * bw * 0.5, c.dy - bh * 0.25, c.dx + side * bw, c.dy - bh * 0.05)
            ..lineTo(c.dx + side * bw, c.dy - bh * 1.05)
            ..quadraticBezierTo(c.dx + side * bw * 0.5, c.dy - bh * 1.25, c.dx, c.dy - bh)
            ..close();
          canvas.drawPath(page, Paint()..color = Colors.white.withValues(alpha: 0.82));
          for (var i = 1; i <= 3; i++) {
            final y = c.dy - bh * (0.25 * i);
            canvas.drawLine(
              Offset(c.dx + side * bw * 0.18, y - bh * 0.12),
              Offset(c.dx + side * bw * 0.82, y - bh * 0.06),
              Paint()
                ..color = p.a.withValues(alpha: 0.35)
                ..strokeWidth = math.max(0.6, w * 0.01),
            );
          }
        }
    }
  }

  @override
  bool shouldRepaint(_CoverPainter old) => old.seed != seed || old.style != style || old.p != p;
}

const _characterIcons = <String, IconData>{
  'narrator': Icons.auto_stories_rounded,
  'storyteller': Icons.local_fire_department_rounded,
  'professor': Icons.school_rounded,
  'sage': Icons.spa_rounded,
  'host': Icons.mic_rounded,
  'cohost': Icons.headset_mic_rounded,
  'explorer': Icons.explore_rounded,
  'anchor': Icons.newspaper_rounded,
  'reporter': Icons.campaign_rounded,
  'night': Icons.nightlight_round,
  'student': Icons.backpack_rounded,
  'maestro': Icons.palette_rounded,
  'philosophe': Icons.psychology_rounded,
  'santa': Icons.celebration_rounded,
  'librarian': Icons.local_library_rounded,
  'announcer': Icons.speaker_rounded,
};

/// A round portrait for a character or voice: its own gradient, with the
/// character's symbol or the voice's initial.
class VoiceAvatar extends StatelessWidget {
  const VoiceAvatar({super.key, required this.name, this.characterId, this.size = 40});
  final String name;
  final String? characterId;
  final double size;

  @override
  Widget build(BuildContext context) {
    final p = paletteFor(characterId ?? name);
    final icon = _characterIcons[characterId];
    final initial = name.replaceFirst(RegExp(r'^(The|La|Il)\s+', caseSensitive: false), '').trim();
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [p.b, p.a]),
        boxShadow: [BoxShadow(color: p.a.withValues(alpha: 0.25), blurRadius: size * 0.2, offset: Offset(0, size * 0.06))],
      ),
      alignment: Alignment.center,
      child: icon != null
          ? Icon(icon, color: Colors.white, size: size * 0.5)
          : Text(
              initial.isEmpty ? '?' : initial.characters.first.toUpperCase(),
              style: TextStyle(fontFamily: displayFont, fontWeight: FontWeight.w600, color: Colors.white, fontSize: size * 0.44, height: 1),
            ),
    );
  }
}

/// Bars that dance while audio plays (and rest when it doesn't).
class Equalizer extends StatefulWidget {
  const Equalizer({super.key, required this.playing, this.color, this.size = 18, this.bars = 4});
  final bool playing;
  final Color? color;
  final double size;
  final int bars;

  @override
  State<Equalizer> createState() => _EqualizerState();
}

class _EqualizerState extends State<Equalizer> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 1400));

  @override
  void initState() {
    super.initState();
    if (widget.playing) _c.repeat();
  }

  @override
  void didUpdateWidget(Equalizer old) {
    super.didUpdateWidget(old);
    if (widget.playing && !_c.isAnimating) {
      _c.repeat();
    } else if (!widget.playing && _c.isAnimating) {
      _c.stop();
    }
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.color ?? Theme.of(context).colorScheme.primary;
    return AnimatedBuilder(
      animation: _c,
      builder: (context, _) => CustomPaint(
        size: Size(widget.size * 1.1, widget.size),
        painter: _EqPainter(widget.playing ? _c.value : -1, color, widget.bars),
      ),
    );
  }
}

class _EqPainter extends CustomPainter {
  _EqPainter(this.t, this.color, this.bars);
  final double t;
  final Color color;
  final int bars;

  @override
  void paint(Canvas canvas, Size size) {
    final gap = size.width / (bars * 2 - 1);
    final paint = Paint()..color = color;
    for (var i = 0; i < bars; i++) {
      final level = t < 0
          ? 0.25 + 0.1 * (i % 2)
          : 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * (t * (1 + i * 0.37) + i * 0.23))).abs();
      final bh = size.height * level;
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(i * gap * 2, size.height - bh, gap, bh), Radius.circular(gap / 2)),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(_EqPainter old) => old.t != t || old.color != color;
}

/// A section label in small caps with an optional action.
class SectionLabel extends StatelessWidget {
  const SectionLabel(this.text, {super.key, this.trailing});
  final String text;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 10),
      child: Row(
        children: [
          Expanded(
            child: Text(
              text.toUpperCase(),
              style: theme.textTheme.labelMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant, letterSpacing: 1.2),
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}

/// A rounded gradient tile holding an icon (feature cards, packs).
class IconTile extends StatelessWidget {
  const IconTile({super.key, required this.icon, required this.seed, this.size = 44});
  final IconData icon;
  final String seed;
  final double size;

  @override
  Widget build(BuildContext context) {
    final p = paletteFor(seed);
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(size * 0.3),
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [p.b, p.a]),
      ),
      child: Icon(icon, color: Colors.white, size: size * 0.52),
    );
  }
}
