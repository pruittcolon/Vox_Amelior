import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/mic_tune.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Seconds as shown: "0.6 s", "0.65 s", "1.5 s".
String formatSeconds(double v) {
  final two = v.toStringAsFixed(2);
  return '${two.endsWith('0') ? two.substring(0, two.length - 1) : two} s';
}

/// Everything about how Vox hears: the microphone, when speech starts and
/// ends, and which speech model reads it.
class ListeningSettingsScreen extends StatelessWidget {
  const ListeningSettingsScreen({super.key, required this.services});

  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Microphone & hearing')),
      body: ListenableBuilder(
        listenable: Listenable.merge([services.settings, services.downloads]),
        builder: (context, _) {
          final st = services.settings.value;
          void update(AppSettings next) => services.updateSettings(next);
          return ListView(
            padding: const EdgeInsets.only(bottom: 32),
            children: [
              Padding(padding: const EdgeInsets.fromLTRB(16, 8, 16, 8), child: MicTuneCard(services: services)),
              SettingsGroup(
                title: 'When speech counts',
                footer: 'These apply straight away while Vox is listening.',
                children: [
                  SettingSlider(
                    title: 'Sensitivity',
                    subtitle: 'How easily a sound is taken for speech.',
                    // Stored as a threshold (lower hears more); shown as sensitivity (higher hears more).
                    value: 1 - st.vadThreshold,
                    min: 0.2,
                    max: 0.8,
                    step: 0.05,
                    defaultValue: 0.5,
                    format: (v) => '${(v * 100).round()}%',
                    unitHint: 'Percent, 20 to 80',
                    parse: (t) {
                      final n = double.tryParse(t.trim().replaceAll('%', ''));
                      return n == null ? null : (n > 1 ? n / 100 : n);
                    },
                    lowLabel: 'Ignores noise',
                    highLabel: 'Hears quiet voices',
                    onChanged: (v) => update(st.copyWith(vadThreshold: (1 - v).clamp(0.2, 0.8))),
                  ),
                  SettingSlider(
                    title: 'Pause that ends a sentence',
                    subtitle: 'Short: quick, separate lines. Long: keeps a slow speaker together.',
                    value: st.pauseSeconds,
                    min: 0.3,
                    max: 1.5,
                    step: 0.05,
                    defaultValue: 0.6,
                    format: formatSeconds,
                    unitHint: 'Seconds',
                    lowLabel: '0.3 s',
                    highLabel: '1.5 s',
                    onChanged: (v) => update(st.copyWith(pauseSeconds: v)),
                  ),
                  SettingSlider(
                    title: 'Ignore short sounds',
                    subtitle: 'Coughs, clicks and bumps shorter than this are dropped.',
                    value: st.minSpeechSeconds,
                    min: 0.1,
                    max: 1.0,
                    step: 0.05,
                    defaultValue: 0.3,
                    format: formatSeconds,
                    unitHint: 'Seconds',
                    lowLabel: 'Keeps everything',
                    highLabel: 'Only real words',
                    onChanged: (v) => update(st.copyWith(minSpeechSeconds: v)),
                  ),
                ],
              ),
              SettingsGroup(
                title: 'Speech model',
                footer: 'All three are the same NVIDIA Parakeet model and write punctuation and capitals. '
                    'High precision (fp16) is downloaded first and used by default. The smaller int8 and the full '
                    'precision fp32 models are only downloaded if you pick them; any one is enough to listen. '
                    'fp32 is rarely more accurate than fp16 but needs about twice the space and memory.',
                children: [
                  ChoiceCards<String>(
                    selected: st.speechModel,
                    onSelected: (v) => services.selectSpeechModel(v),
                    options: [
                      ChoiceOption(
                        value: 'fp16',
                        title: 'High precision (fp16)',
                        subtitle: '${formatBytes(ModelCatalog.parakeetFp16.approxDownloadBytes)} download · most accurate · recommended',
                        status: _status(context, ModelCatalog.parakeetFp16),
                      ),
                      ChoiceOption(
                        value: 'int8',
                        title: 'Smaller (int8)',
                        subtitle: '${formatBytes(ModelCatalog.parakeet.approxDownloadBytes)} download · small and fast',
                        status: _status(context, ModelCatalog.parakeet),
                      ),
                      ChoiceOption(
                        value: 'fp32',
                        title: 'Full precision (fp32)',
                        subtitle: '${formatBytes(ModelCatalog.parakeetFp32.approxDownloadBytes)} download · largest and slowest · '
                            'needs that much free space',
                        status: _status(context, ModelCatalog.parakeetFp32),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _status(BuildContext context, ModelAsset asset) {
    final d = services.downloads.stateOf(asset);
    switch (d.status) {
      case DownloadStatus.installed:
        const installed = Pill('Installed', icon: Icons.check_rounded, color: Color(0xFF0E9F6E));
        return Wrap(
          spacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            installed,
            TextButton.icon(
              style: TextButton.styleFrom(visualDensity: VisualDensity.compact),
              icon: const Icon(Icons.delete_outline_rounded, size: 18),
              label: const Text('Delete'),
              onPressed: () async {
                final other = ModelCatalog.recognizers.where((m) => m.id != asset.id && services.models.isInstalled(m)).firstOrNull;
                final message = other != null
                    ? 'Frees about ${formatBytes(asset.approxDownloadBytes)}. Vox switches to ${other.title}.'
                    : 'Frees about ${formatBytes(asset.approxDownloadBytes)}. This is your only speech model, so '
                        'listening stops until you download one again.';
                if (await confirm(context, 'Delete ${asset.title}?', message)) {
                  await services.removeSpeechModel(asset);
                }
              },
            ),
          ],
        );
      case DownloadStatus.downloading:
      case DownloadStatus.queued:
      case DownloadStatus.unpacking:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              d.status == DownloadStatus.downloading ? 'Downloading ${describeDownload(d)}' : describeDownload(d),
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 4),
            LinearProgressIndicator(value: d.status == DownloadStatus.queued ? null : d.progress, minHeight: 5),
          ],
        );
      case DownloadStatus.failed:
        return Pill(d.error ?? 'Download failed. Tap to retry.', icon: Icons.error_outline_rounded, color: Theme.of(context).colorScheme.error);
      case DownloadStatus.notInstalled:
        return const Pill('Downloads when selected', icon: Icons.download_rounded);
    }
  }
}
