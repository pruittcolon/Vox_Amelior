import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

/// Gear button that opens settings and everything else (the More screen).
class SettingsButton extends StatelessWidget {
  const SettingsButton({super.key, required this.builder});

  /// Builds the settings screen (kept out of this file to avoid an import cycle).
  final WidgetBuilder builder;

  @override
  Widget build(BuildContext context) => IconButton(
        tooltip: 'Settings and more',
        icon: const Icon(Icons.settings_rounded),
        onPressed: () => Navigator.push(context, MaterialPageRoute<void>(builder: builder)),
      );
}

/// Small bold heading above a group of content.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.text, {super.key, this.trailing, this.padding = const EdgeInsets.fromLTRB(20, 24, 16, 8)});

  final String text;
  final Widget? trailing;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Padding(
      padding: padding,
      child: Row(
        children: [
          Expanded(
            child: Text(
              text.toUpperCase(),
              style: t.textTheme.labelMedium?.copyWith(
                letterSpacing: 1.1,
                fontWeight: FontWeight.w700,
                color: t.colorScheme.primary,
              ),
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}

/// Friendly placeholder for empty lists.
class EmptyState extends StatelessWidget {
  const EmptyState({super.key, required this.icon, required this.title, this.message, this.action});

  final IconData icon;
  final String title;
  final String? message;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(color: t.colorScheme.primaryContainer, shape: BoxShape.circle),
              child: Icon(icon, size: 36, color: t.colorScheme.onPrimaryContainer),
            ),
            const SizedBox(height: 16),
            Text(title, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600), textAlign: TextAlign.center),
            if (message != null) ...[
              const SizedBox(height: 8),
              Text(message!, style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant), textAlign: TextAlign.center),
            ],
            if (action != null) ...[const SizedBox(height: 20), action!],
          ],
        ),
      ),
    );
  }
}

class SpeakerAvatar extends StatelessWidget {
  const SpeakerAvatar({super.key, required this.label, this.known = true, this.radius = 18});

  final String label;
  final bool known;
  final double radius;

  @override
  Widget build(BuildContext context) {
    final c = speakerColor(label, known: known);
    return CircleAvatar(
      radius: radius,
      backgroundColor: c.withValues(alpha: 0.16),
      foregroundColor: c,
      child: Text(
        label.isEmpty ? '?' : label.characters.first.toUpperCase(),
        style: TextStyle(fontWeight: FontWeight.w700, fontSize: radius * 0.8),
      ),
    );
  }
}

/// Overlapping avatars for a conversation's participants.
class AvatarStack extends StatelessWidget {
  const AvatarStack({super.key, required this.labels, this.max = 4});

  final List<String> labels;
  final int max;

  @override
  Widget build(BuildContext context) {
    final shown = labels.take(max).toList();
    return SizedBox(
      height: 28,
      width: 28.0 + (shown.length - 1).clamp(0, max) * 18,
      child: Stack(
        children: [
          for (var i = 0; i < shown.length; i++)
            Positioned(
              left: i * 18.0,
              child: DecoratedBox(
                decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Theme.of(context).colorScheme.surface, width: 2)),
                child: SpeakerAvatar(label: shown[i], known: !shown[i].startsWith('Guest') && shown[i] != 'Unknown', radius: 12),
              ),
            ),
        ],
      ),
    );
  }
}

/// Small rounded status label.
class Pill extends StatelessWidget {
  const Pill(this.text, {super.key, this.icon, this.color});

  final String text;
  final IconData? icon;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.primary;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: c.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(99)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[Icon(icon, size: 14, color: c), const SizedBox(width: 4)],
          Flexible(
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: c, fontWeight: FontWeight.w600, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

/// Rounded card with optional tap and padding, used across screens.
class VoxCard extends StatelessWidget {
  const VoxCard({super.key, required this.child, this.onTap, this.padding = const EdgeInsets.all(16), this.color});

  final Widget child;
  final VoidCallback? onTap;
  final EdgeInsets padding;
  final Color? color;

  @override
  Widget build(BuildContext context) => Card(
        color: color,
        child: InkWell(onTap: onTap, child: Padding(padding: padding, child: child)),
      );
}
