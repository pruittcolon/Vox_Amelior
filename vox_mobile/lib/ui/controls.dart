import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// A soft card holding a stack of settings rows, with hairlines between them.
class SettingsGroup extends StatelessWidget {
  const SettingsGroup({super.key, this.title, this.footer, required this.children});

  final String? title;
  final String? footer;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (title != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(6, 12, 6, 8),
              child: Text(
                title!.toUpperCase(),
                style: t.textTheme.labelMedium?.copyWith(letterSpacing: 1.1, fontWeight: FontWeight.w700, color: t.colorScheme.primary),
              ),
            ),
          Card(
            child: Column(
              children: [
                for (var i = 0; i < children.length; i++) ...[
                  if (i > 0) Divider(height: 1, indent: 16, endIndent: 16, color: t.colorScheme.outlineVariant.withValues(alpha: 0.4)),
                  children[i],
                ],
              ],
            ),
          ),
          if (footer != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 8, 8, 0),
              child: Text(footer!, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
            ),
        ],
      ),
    );
  }
}

/// A round tinted icon, used at the start of settings rows.
class IconBadge extends StatelessWidget {
  const IconBadge(this.icon, {super.key, this.color, this.size = 40});

  final IconData icon;
  final Color? color;
  final double size;

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.primary;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(color: c.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(size * 0.32)),
      child: Icon(icon, color: c, size: size * 0.55),
    );
  }
}

/// A tappable row that opens another page: icon, title, live summary, chevron.
class NavRow extends StatelessWidget {
  const NavRow({super.key, required this.icon, required this.title, required this.summary, required this.onTap, this.color});

  final IconData icon;
  final String title;
  final String summary;
  final VoidCallback onTap;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            IconBadge(icon, color: color),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Text(summary, maxLines: 2, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: t.colorScheme.onSurfaceVariant),
          ],
        ),
      ),
    );
  }
}

/// A switch row with a plain-language explanation.
class SettingSwitch extends StatelessWidget {
  const SettingSwitch({super.key, required this.title, required this.subtitle, required this.value, required this.onChanged});

  final String title;
  final String subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) => SwitchListTile(
        contentPadding: const EdgeInsets.fromLTRB(16, 4, 12, 4),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        subtitle: Padding(padding: const EdgeInsets.only(top: 2), child: Text(subtitle)),
        value: value,
        onChanged: (v) {
          HapticFeedback.selectionClick();
          onChanged(v);
        },
      );
}

/// A slider that is quick to use and exact when you want it to be:
/// drag, nudge with − / +, or tap the value to type one. Changes are applied
/// when you let go, and a reset arrow appears when it differs from the default.
class SettingSlider extends StatefulWidget {
  const SettingSlider({
    super.key,
    required this.title,
    required this.value,
    required this.min,
    required this.max,
    required this.step,
    required this.defaultValue,
    required this.format,
    required this.onChanged,
    this.subtitle,
    this.lowLabel,
    this.highLabel,
    this.parse,
    this.unitHint = '',
    this.padding = const EdgeInsets.fromLTRB(16, 14, 8, 8),
  });

  final String title;
  final String? subtitle;
  final double value;
  final double min;
  final double max;
  final double step;
  final double defaultValue;

  /// Text shown for a value ("+15%", "0.6 s").
  final String Function(double value) format;
  final ValueChanged<double> onChanged;

  /// Meaning of the two ends ("Hears quieter", "Ignores noise").
  final String? lowLabel;
  final String? highLabel;

  /// Turns typed text into a value (null = not a number). Defaults to a plain number.
  final double? Function(String text)? parse;
  final String unitHint;
  final EdgeInsets padding;

  @override
  State<SettingSlider> createState() => _SettingSliderState();
}

class _SettingSliderState extends State<SettingSlider> {
  double? _drag;

  double _snap(double v) {
    final steps = ((v - widget.min) / widget.step).round();
    return (widget.min + steps * widget.step).clamp(widget.min, widget.max);
  }

  double get _shown => _snap(_drag ?? widget.value);
  bool get _atDefault => (_shown - widget.defaultValue).abs() < widget.step / 2;

  @override
  void didUpdateWidget(SettingSlider old) {
    super.didUpdateWidget(old);
    // The page has caught up with what was chosen (or refused it): show its value again.
    if (old.value != widget.value) _drag = null;
  }

  void _commit(double v) {
    final s = _snap(v);
    // Keep showing the choice until the page rebuilds, so quick taps on − / + add up.
    setState(() => _drag = s);
    if ((s - widget.value).abs() > 1e-9) widget.onChanged(s);
  }

  Future<void> _type() async {
    final text = await showDialog<String>(
      context: context,
      builder: (_) => _ExactValueDialog(
        title: widget.title,
        initial: widget.format(_shown).replaceAll(RegExp(r'[^0-9.\-]'), ''),
        hint: widget.unitHint,
      ),
    );
    if (text == null) return;
    final v = (widget.parse ?? (s) => double.tryParse(s.trim()))(text);
    if (v != null) _commit(v);
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final divisions = ((widget.max - widget.min) / widget.step).round();
    return Padding(
      padding: widget.padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: Text(widget.title, style: t.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700))),
              if (!_atDefault)
                IconButton(
                  tooltip: 'Back to ${widget.format(widget.defaultValue)}',
                  visualDensity: VisualDensity.compact,
                  icon: const Icon(Icons.restart_alt_rounded, size: 20),
                  onPressed: () {
                    HapticFeedback.selectionClick();
                    _commit(widget.defaultValue);
                  },
                ),
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: InkWell(
                  borderRadius: BorderRadius.circular(99),
                  onTap: _type,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(color: t.colorScheme.primaryContainer, borderRadius: BorderRadius.circular(99)),
                    child: Text(
                      widget.format(_shown),
                      style: TextStyle(fontWeight: FontWeight.w800, color: t.colorScheme.onPrimaryContainer, fontFeatures: const [FontFeature.tabularFigures()]),
                    ),
                  ),
                ),
              ),
            ],
          ),
          if (widget.subtitle != null)
            Padding(
              padding: const EdgeInsets.only(right: 8, top: 2),
              child: Text(widget.subtitle!, style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
            ),
          Row(
            children: [
              IconButton(
                tooltip: 'Less',
                visualDensity: VisualDensity.compact,
                icon: const Icon(Icons.remove_rounded),
                onPressed: _shown <= widget.min + 1e-9 ? null : () => _commit(_shown - widget.step),
              ),
              Expanded(
                child: Slider(
                  value: _shown.clamp(widget.min, widget.max),
                  min: widget.min,
                  max: widget.max,
                  divisions: divisions,
                  onChanged: (v) {
                    final s = _snap(v);
                    if (s != _drag) HapticFeedback.selectionClick();
                    setState(() => _drag = s);
                  },
                  onChangeEnd: _commit,
                ),
              ),
              IconButton(
                tooltip: 'More',
                visualDensity: VisualDensity.compact,
                icon: const Icon(Icons.add_rounded),
                onPressed: _shown >= widget.max - 1e-9 ? null : () => _commit(_shown + widget.step),
              ),
            ],
          ),
          if (widget.lowLabel != null || widget.highLabel != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(44, 0, 44, 4),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Flexible(child: Text(widget.lowLabel ?? '', style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant))),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(widget.highLabel ?? '',
                        textAlign: TextAlign.end, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// A row of big, clearly labelled choices (one is selected).
class ChoiceCards<T> extends StatelessWidget {
  const ChoiceCards({super.key, required this.options, required this.selected, required this.onSelected});

  final List<ChoiceOption<T>> options;
  final T selected;
  final ValueChanged<T> onSelected;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          for (final o in options)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Material(
                color: o.value == selected ? t.colorScheme.primaryContainer : t.colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(16),
                child: InkWell(
                  borderRadius: BorderRadius.circular(16),
                  onTap: () {
                    HapticFeedback.selectionClick();
                    onSelected(o.value);
                  },
                  child: Padding(
                    padding: const EdgeInsets.all(14),
                    child: Row(
                      children: [
                        Icon(o.value == selected ? Icons.check_circle_rounded : Icons.circle_outlined,
                            color: o.value == selected ? t.colorScheme.primary : t.colorScheme.outline),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(o.title, style: t.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
                              const SizedBox(height: 2),
                              Text(o.subtitle, style: t.textTheme.bodySmall),
                              if (o.status != null) ...[const SizedBox(height: 6), o.status!],
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class ChoiceOption<T> {
  const ChoiceOption({required this.value, required this.title, required this.subtitle, this.status});

  final T value;
  final String title;
  final String subtitle;
  final Widget? status;
}

/// Asks for an exact number. Owns its text field so it is disposed with the dialog.
class _ExactValueDialog extends StatefulWidget {
  const _ExactValueDialog({required this.title, required this.initial, required this.hint});

  final String title;
  final String initial;
  final String hint;

  @override
  State<_ExactValueDialog> createState() => _ExactValueDialogState();
}

class _ExactValueDialogState extends State<_ExactValueDialog> {
  late final TextEditingController _controller = TextEditingController(text: widget.initial);

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
        title: Text(widget.title),
        content: TextField(
          controller: _controller,
          autofocus: true,
          keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
          decoration: InputDecoration(hintText: 'Exact value', helperText: widget.hint.isEmpty ? null : widget.hint),
          onSubmitted: (v) => Navigator.pop(context, v),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, _controller.text), child: const Text('Set')),
        ],
      );
}
