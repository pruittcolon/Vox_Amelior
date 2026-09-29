import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/theme.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Colours, dark mode, text size and shapes. Changes apply instantly.
class AppearanceScreen extends StatelessWidget {
  const AppearanceScreen({super.key, required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Appearance'),
        actions: [
          TextButton(
            onPressed: () {
              const d = AppSettings();
              services.updateSettings(services.settings.value.copyWith(
                themeMode: d.themeMode,
                accent: d.accent,
                textScale: d.textScale,
                corners: d.corners,
              ));
            },
            child: const Text('Reset'),
          ),
        ],
      ),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) {
          final t = Theme.of(context);
          void set(AppSettings next) => services.updateSettings(next);
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              _Preview(),
              const SectionHeader('Theme', padding: EdgeInsets.fromLTRB(4, 24, 4, 10)),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'system', label: Text('Auto'), icon: Icon(Icons.brightness_auto_rounded)),
                  ButtonSegment(value: 'light', label: Text('Light'), icon: Icon(Icons.light_mode_rounded)),
                  ButtonSegment(value: 'dark', label: Text('Dark'), icon: Icon(Icons.dark_mode_rounded)),
                ],
                selected: {st.themeMode},
                onSelectionChanged: (v) => set(st.copyWith(themeMode: v.first)),
              ),
              const SectionHeader('Colour', padding: EdgeInsets.fromLTRB(4, 24, 4, 10)),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  for (final (name, color) in kAccentChoices)
                    Tooltip(
                      message: name,
                      child: InkWell(
                        customBorder: const CircleBorder(),
                        onTap: () => set(st.copyWith(accent: color.toARGB32())),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 180),
                          width: 48,
                          height: 48,
                          decoration: BoxDecoration(
                            color: color,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: st.accent == color.toARGB32() ? t.colorScheme.onSurface : Colors.transparent,
                              width: 3,
                            ),
                          ),
                          child: st.accent == color.toARGB32() ? const Icon(Icons.check_rounded, color: Colors.white) : null,
                        ),
                      ),
                    ),
                ],
              ),
              const SectionHeader('Text size', padding: EdgeInsets.fromLTRB(4, 24, 4, 4)),
              Row(
                children: [
                  const Text('A', style: TextStyle(fontSize: 14)),
                  Expanded(
                    child: Slider(
                      value: st.textScale.clamp(0.85, 1.4),
                      min: 0.85,
                      max: 1.4,
                      divisions: 11,
                      label: '${(st.textScale * 100).round()}%',
                      onChanged: (v) => set(st.copyWith(textScale: double.parse(v.toStringAsFixed(2)))),
                    ),
                  ),
                  const Text('A', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w600)),
                ],
              ),
              const SectionHeader('Corners', padding: EdgeInsets.fromLTRB(4, 24, 4, 10)),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'square', label: Text('Sharp')),
                  ButtonSegment(value: 'soft', label: Text('Soft')),
                  ButtonSegment(value: 'rounded', label: Text('Round')),
                ],
                selected: {st.corners},
                onSelectionChanged: (v) => set(st.copyWith(corners: v.first)),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// A small sample of the app in the chosen style.
class _Preview extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return VoxCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const SpeakerAvatar(label: 'Alex'),
              const SizedBox(width: 10),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: t.colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text('Did we book the plumber for Thursday?', style: t.textTheme.bodyLarge),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            children: [
              const Pill('Preview', icon: Icons.palette_rounded),
              Pill('3 found', icon: Icons.check_circle_rounded, color: t.colorScheme.tertiary),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(child: FilledButton(onPressed: () {}, child: const Text('Ask'))),
              const SizedBox(width: 10),
              Expanded(child: OutlinedButton(onPressed: () {}, child: const Text('Review'))),
            ],
          ),
        ],
      ),
    );
  }
}
