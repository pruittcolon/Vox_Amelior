import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/ui/automations_screen.dart';
import 'package:vox_amelior_mobile/ui/models_screen.dart';
import 'package:vox_amelior_mobile/ui/places_screen.dart';
import 'package:vox_amelior_mobile/ui/settings_screen.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

class MoreScreen extends StatelessWidget {
  const MoreScreen({super.key, required this.services});

  final AppServices services;

  void _go(BuildContext context, Widget screen) => Navigator.push(context, MaterialPageRoute<void>(builder: (_) => screen));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('More')),
      body: ListenableBuilder(
        listenable: Listenable.merge([services.settings, services.downloads]),
        builder: (context, _) {
          final st = services.settings.value;
          final assistant = st.llmAsset;
          final tiles = <(IconData, String, String, Widget)>[
            (
              Icons.download_rounded,
              'Models',
              services.downloads.isBusy
                  ? 'Downloading…'
                  : '${services.speechReady ? 'Speech ready' : 'Speech not downloaded'} · '
                      '${services.assistantReady ? assistant.title : '${assistant.title} not downloaded'}',
              ModelsScreen(services: services),
            ),
            (
              Icons.place_rounded,
              'Places',
              switch (st.locationMode) {
                LocationMode.off => 'Off — listen everywhere',
                LocationMode.onlyAtPlaces => 'Listen only at ${st.places.map((p) => p.name).join(', ')}',
                LocationMode.pauseAtPlaces => 'Pause at ${st.places.map((p) => p.name).join(', ')}',
              },
              PlacesScreen(services: services),
            ),
            (Icons.bolt_rounded, 'Automations', 'Webhooks, notifications and notes triggered by speech', AutomationsScreen(services: services)),
            (Icons.tune_rounded, 'Settings', 'Assistant, voices, privacy', SettingsScreen(services: services)),
          ];
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
            children: [
              for (final (icon, title, subtitle, screen) in tiles)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: VoxCard(
                    onTap: () => _go(context, screen),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: Theme.of(context).colorScheme.primaryContainer,
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Icon(icon, color: Theme.of(context).colorScheme.onPrimaryContainer),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                              Text(subtitle, maxLines: 2, overflow: TextOverflow.ellipsis),
                            ],
                          ),
                        ),
                        const Icon(Icons.chevron_right_rounded),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 8),
              const AboutListTile(
                icon: Icon(Icons.info_outline_rounded),
                applicationName: 'Vox Amelior',
                aboutBoxChildren: [
                  Text('Private, on-device assistant. Speech: NVIDIA Parakeet RNNT 1.1B, Silero VAD and TitaNet via '
                      'sherpa-onnx. Assistant: Google Gemma 4 via LiteRT-LM. Nothing leaves your phone.'),
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}
