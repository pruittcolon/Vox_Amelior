import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';
import 'package:vox_amelior_mobile/ui/listening_settings_screen.dart';

/// Who is talking: naming people, unknown guests, and when a line is split.
class SpeakersSettingsScreen extends StatelessWidget {
  const SpeakersSettingsScreen({super.key, required this.services});

  final AppServices services;

  static String _pct(double v) => '${(v * 100).round()}%';

  static double? _parsePct(String t) {
    final n = double.tryParse(t.trim().replaceAll('%', ''));
    return n == null ? null : (n > 1 ? n / 100 : n);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Voices & speakers')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: services.settings,
        builder: (context, st, _) {
          void update(AppSettings next) => services.updateSettings(next);
          final hasDiarizer = services.models.isInstalled(ModelCatalog.diarizer);
          return ListView(
            padding: const EdgeInsets.only(bottom: 32),
            children: [
              SettingsGroup(
                title: 'Naming people',
                children: [
                  SettingSwitch(
                    title: 'Several voice patterns per person',
                    subtitle: 'Recognises someone close up, across the room or with a cold. Off uses one average voice.',
                    value: st.multiPatterns,
                    onChanged: (v) => update(st.copyWith(multiPatterns: v)),
                  ),
                  SettingSlider(
                    title: 'How sure before naming someone',
                    subtitle: 'Higher means fewer wrong names and more "Guest" labels.',
                    value: st.matchThreshold,
                    min: 0.3,
                    max: 0.9,
                    step: 0.05,
                    defaultValue: 0.55,
                    format: _pct,
                    parse: _parsePct,
                    unitHint: 'Percent, 30 to 90',
                    lowLabel: 'Names easily',
                    highLabel: 'Only if sure',
                    onChanged: (v) => update(st.copyWith(matchThreshold: v)),
                  ),
                  SettingSlider(
                    title: 'Lead over the runner-up',
                    subtitle: 'How much better the best match must beat the second best.',
                    value: st.matchMargin,
                    min: 0,
                    max: 0.3,
                    step: 0.01,
                    defaultValue: 0.04,
                    format: _pct,
                    parse: _parsePct,
                    unitHint: 'Percent, 0 to 30',
                    lowLabel: 'Any lead',
                    highLabel: 'Clear winner',
                    onChanged: (v) => update(st.copyWith(matchMargin: v)),
                  ),
                  SettingSlider(
                    title: 'Grouping unknown voices',
                    subtitle: 'How alike two strangers must sound to be the same "Guest".',
                    value: st.guestThreshold,
                    min: 0.3,
                    max: 0.9,
                    step: 0.05,
                    defaultValue: 0.6,
                    format: _pct,
                    parse: _parsePct,
                    unitHint: 'Percent, 30 to 90',
                    lowLabel: 'Fewer guests',
                    highLabel: 'More guests',
                    onChanged: (v) => update(st.copyWith(guestThreshold: v)),
                  ),
                ],
              ),
              SettingsGroup(
                title: 'Splitting lines',
                footer: hasDiarizer
                    ? 'A line is only split when every part is at least this long and has this many words, '
                        'and the parts are different named people. Otherwise it stays whole.'
                    : 'Needs the speaker-change model (More → Models).',
                children: [
                  SettingSwitch(
                    title: 'Split lines when the speaker changes',
                    subtitle: 'Separate lines for quick back-and-forth, and "talking at once" marks.',
                    value: st.splitSpeakers,
                    onChanged: (v) => update(st.copyWith(splitSpeakers: v)),
                  ),
                  if (st.splitSpeakers) ...[
                    SettingSlider(
                      title: 'Shortest part of a line',
                      value: st.splitMinSeconds,
                      min: 0.8,
                      max: 3.0,
                      step: 0.1,
                      defaultValue: 1.5,
                      format: formatSeconds,
                      unitHint: 'Seconds',
                      lowLabel: 'Splits eagerly',
                      highLabel: 'Splits carefully',
                      onChanged: (v) => update(st.copyWith(splitMinSeconds: v)),
                    ),
                    SettingSlider(
                      title: 'Fewest words in a part',
                      value: st.splitMinWords.toDouble(),
                      min: 1,
                      max: 5,
                      step: 1,
                      defaultValue: 2,
                      format: (v) => v.round() == 1 ? '1 word' : '${v.round()} words',
                      unitHint: 'Words, 1 to 5',
                      lowLabel: 'Splits eagerly',
                      highLabel: 'Splits carefully',
                      onChanged: (v) => update(st.copyWith(splitMinWords: v.round())),
                    ),
                  ],
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}
