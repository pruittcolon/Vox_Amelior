import 'package:flutter/material.dart';

import '../app/theme.dart';

/// Small rounded status label with a dot.
class StatusPill extends StatelessWidget {
  const StatusPill(this.label, {super.key, required this.color, this.pulse = false});

  final String label;
  final Color color;
  final bool pulse;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.14),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: color.withOpacity(0.45)),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        pulse ? _PulseDot(color: color) : _Dot(color: color),
        const SizedBox(width: 6),
        Flexible(
          child: Text(label,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: color, fontSize: 12.5, fontWeight: FontWeight.w700)),
        ),
      ]),
    );
  }
}

class _Dot extends StatelessWidget {
  const _Dot({required this.color});
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      );
}

class _PulseDot extends StatefulWidget {
  const _PulseDot({required this.color});
  final Color color;
  @override
  State<_PulseDot> createState() => _PulseDotState();
}

class _PulseDotState extends State<_PulseDot> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))..repeat(reverse: true);

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => FadeTransition(
        opacity: Tween(begin: 0.35, end: 1.0).animate(_c),
        child: _Dot(color: widget.color),
      );
}

/// A card with a title row and content.
class SectionCard extends StatelessWidget {
  const SectionCard({
    super.key,
    required this.child,
    this.title,
    this.icon,
    this.iconColor,
    this.trailing,
    this.padding = const EdgeInsets.all(18),
  });

  final Widget child;
  final String? title;
  final IconData? icon;
  final Color? iconColor;
  final Widget? trailing;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: padding,
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          if (title != null) ...[
            Row(children: [
              if (icon != null) ...[
                IconBadge(icon!, color: iconColor ?? AppColors.glasses),
                const SizedBox(width: 12),
              ],
              Expanded(child: Text(title!, style: Theme.of(context).textTheme.titleMedium)),
              if (trailing != null) trailing!,
            ]),
            const SizedBox(height: 14),
          ],
          child,
        ]),
      ),
    );
  }
}

class IconBadge extends StatelessWidget {
  const IconBadge(this.icon, {super.key, required this.color, this.size = 40});
  final IconData icon;
  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) => Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(size * 0.32),
        ),
        child: Icon(icon, color: color, size: size * 0.55),
      );
}

/// Big square-ish tile for quick actions.
class ActionTile extends StatelessWidget {
  const ActionTile({
    super.key,
    required this.icon,
    required this.label,
    required this.caption,
    required this.color,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final String caption;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            IconBadge(icon, color: color),
            const SizedBox(height: 12),
            Text(label, style: Theme.of(context).textTheme.titleMedium, maxLines: 1, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 2),
            Text(caption,
                style: Theme.of(context).textTheme.bodySmall, maxLines: 2, overflow: TextOverflow.ellipsis),
          ]),
        ),
      ),
    );
  }
}

/// Up / down buttons that page the glasses screen.
class PageButtons extends StatelessWidget {
  const PageButtons({
    super.key,
    required this.onUp,
    required this.onDown,
    this.label = '',
    this.compact = false,
  });

  final VoidCallback? onUp;
  final VoidCallback? onDown;
  final String label;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final size = compact ? 44.0 : 52.0;
    Widget btn(IconData icon, String tip, VoidCallback? onTap) => SizedBox(
          width: size,
          height: size,
          child: IconButton.filledTonal(
            tooltip: tip,
            onPressed: onTap,
            icon: Icon(icon, size: compact ? 24 : 28),
          ),
        );
    return Row(mainAxisSize: MainAxisSize.min, children: [
      btn(Icons.keyboard_arrow_up_rounded, 'Scroll up', onUp),
      if (label.isNotEmpty)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10),
          child: Text(label, style: const TextStyle(color: AppColors.muted, fontWeight: FontWeight.w600)),
        )
      else
        const SizedBox(width: 8),
      btn(Icons.keyboard_arrow_down_rounded, 'Scroll down', onDown),
    ]);
  }
}

/// Empty-state message with an icon.
class EmptyHint extends StatelessWidget {
  const EmptyHint({super.key, required this.icon, required this.title, required this.lines});
  final IconData icon;
  final String title;
  final List<String> lines;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(32),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          IconBadge(icon, color: AppColors.claude, size: 64),
          const SizedBox(height: 16),
          Text(title, style: Theme.of(context).textTheme.titleLarge, textAlign: TextAlign.center),
          const SizedBox(height: 10),
          for (final l in lines)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(l, textAlign: TextAlign.center, style: const TextStyle(color: AppColors.muted, height: 1.4)),
            ),
        ]),
      ),
    );
  }
}

void showToast(BuildContext context, String msg) {
  final m = ScaffoldMessenger.maybeOf(context);
  m?.hideCurrentSnackBar();
  m?.showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 3)));
}
