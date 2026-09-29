import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/speaker_repository.dart';
import 'package:vox_amelior_mobile/location/location_monitor.dart';
import 'package:vox_amelior_mobile/location/location_policy.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Turn listening on or off automatically depending on where you are.
class PlacesScreen extends StatefulWidget {
  const PlacesScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<PlacesScreen> createState() => _PlacesScreenState();
}

class _PlacesScreenState extends State<PlacesScreen> {
  bool _locating = false;

  AppServices get s => widget.services;

  Future<void> _update(AppSettings next) async {
    final wasOff = s.settings.value.locationMode == LocationMode.off;
    await s.updateSettings(next);
    if (next.locationMode == LocationMode.off || !wasOff) return;
    if (!await LocationMonitor.requestPermission()) {
      if (mounted) showMessage(context, 'Allow location access so Vox can tell where it is.');
      return;
    }
    // Android only lets a running service use location if it was started
    // with it, so restart listening once with location enabled.
    if (s.listening.isListening) {
      final error = await s.listening.start(useLocation: true);
      if (error != null && mounted) showMessage(context, error);
    }
  }

  Future<void> _addHere() async {
    if (!await LocationMonitor.requestPermission()) {
      if (mounted) showMessage(context, 'Location permission is needed to save a place.');
      return;
    }
    setState(() => _locating = true);
    final fix = await const LocationMonitor(maxCachedAge: Duration(seconds: 30)).current();
    if (!mounted) return;
    setState(() => _locating = false);
    if (fix == null) {
      showMessage(context, 'Could not get your location. Is location turned on?');
      return;
    }
    final name = await askText(context, 'Name this place', initial: s.settings.value.places.isEmpty ? 'Home' : '', hint: 'Home, Work…');
    if (name == null || name.isEmpty) return;
    final place = Place(id: SpeakerRepository.newId(), name: name, lat: fix.lat, lon: fix.lon, radiusM: fix.accuracyM > 150 ? fix.accuracyM.clamp(150, 500) : 150);
    final st = s.settings.value;
    await _update(st.copyWith(
      places: [...st.places, place],
      locationMode: st.locationMode == LocationMode.off ? LocationMode.onlyAtPlaces : st.locationMode,
    ));
    if (mounted) showMessage(context, 'Saved "$name" (±${fix.accuracyM.round()} m)');
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Places')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _locating ? null : _addHere,
        icon: _locating
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.add_location_alt_rounded),
        label: const Text('Save current location'),
      ),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: s.settings,
        builder: (context, st, _) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 120),
          children: [
            Text(
              'Vox checks where the phone is every few minutes (using low-power location) and pauses or resumes listening. '
              'Keep Vox turned on — it resumes by itself when you arrive.',
              style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
            ),
            const SectionHeader('Rule', padding: EdgeInsets.fromLTRB(4, 20, 4, 8)),
            VoxCard(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: RadioGroup<LocationMode>(
                groupValue: st.locationMode,
                onChanged: (m) => _update(st.copyWith(locationMode: m)),
                child: const Column(
                  children: [
                    RadioListTile(value: LocationMode.off, title: Text('Off'), subtitle: Text('Listen everywhere')),
                    RadioListTile(
                      value: LocationMode.onlyAtPlaces,
                      title: Text('Only at my places'),
                      subtitle: Text('E.g. listen at home, pause everywhere else'),
                    ),
                    RadioListTile(
                      value: LocationMode.pauseAtPlaces,
                      title: Text('Pause at my places'),
                      subtitle: Text('E.g. never listen at work or the doctor'),
                    ),
                  ],
                ),
              ),
            ),
            const SectionHeader('My places', padding: EdgeInsets.fromLTRB(4, 24, 4, 8)),
            if (st.places.isEmpty)
              const EmptyState(
                icon: Icons.home_work_rounded,
                title: 'No places yet',
                message: 'Go to a place (like home) and tap "Save current location".',
              )
            else
              for (final p in st.places)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: VoxCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.place_rounded),
                            const SizedBox(width: 8),
                            Expanded(child: Text(p.name, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700))),
                            IconButton(
                              tooltip: 'Rename',
                              icon: const Icon(Icons.edit_rounded),
                              onPressed: () async {
                                final name = await askText(context, 'Rename place', initial: p.name);
                                if (name == null || name.isEmpty) return;
                                await _update(st.copyWith(places: [for (final x in st.places) x.id == p.id ? x.copyWith(name: name) : x]));
                              },
                            ),
                            IconButton(
                              tooltip: 'Delete',
                              icon: const Icon(Icons.delete_outline_rounded),
                              onPressed: () => _update(st.copyWith(places: st.places.where((x) => x.id != p.id).toList())),
                            ),
                          ],
                        ),
                        Text('Within ${p.radiusM.round()} m', style: t.textTheme.bodySmall),
                        Slider(
                          value: p.radiusM.clamp(50, 1000),
                          min: 50,
                          max: 1000,
                          divisions: 19,
                          label: '${p.radiusM.round()} m',
                          onChanged: (v) => _update(st.copyWith(places: [for (final x in st.places) x.id == p.id ? x.copyWith(radiusM: v) : x])),
                        ),
                      ],
                    ),
                  ),
                ),
          ],
        ),
      ),
    );
  }
}
