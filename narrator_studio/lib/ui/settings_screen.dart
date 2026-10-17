import 'dart:io';
import 'dart:isolate';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;

import '../core/models/catalog.dart';
import '../core/render/audio.dart';
import '../core/tts/accel.dart';
import '../core/tts/voices.dart';
import 'scope.dart';
import 'voice_set.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String? _speedReport;
  bool _measuring = false;
  String? _accelReport;
  bool _testingAccel = false;

  /// Runs tool-style timings of VibeVoice on each processor in a background
  /// isolate, and saves the report to Downloads.
  Future<void> _testAccel() async {
    final app = context.app;
    final native = context.scope.native;
    if (!app.models.isInstalled('vibevoice')) {
      context.toast('Download the Lifelike voices first (Voices tab).');
      return;
    }
    await context.scope.player.stop();
    setState(() {
      _testingAccel = true;
      _accelReport = null;
    });
    final dir = app.vibevoiceDir;
    try {
      final report = await Isolate.run(() => accelReport(dir));
      final f = File(p.join(app.cacheDir.path, 'narrator-accel.txt'));
      await f.parent.create(recursive: true);
      await f.writeAsString(report);
      var where = '';
      try {
        where = await native.saveToDownloads(f.path, 'narrator-accel.txt', 'text/plain');
      } catch (e) {
        where = 'not saved ($e)';
      }
      if (mounted) setState(() => _accelReport = 'Report: $where\n\n$report');
    } catch (e) {
      if (mounted) setState(() => _accelReport = 'Test failed: $e');
    } finally {
      if (mounted) setState(() => _testingAccel = false);
    }
  }

  /// Times a fixed paragraph with 2, 4, 6 and 8 threads and keeps the fastest.
  Future<void> _measure() async {
    final app = context.app;
    if (!app.hasAnyVoices) {
      context.toast('Download a voice pack first.');
      return;
    }
    await context.scope.player.stop();
    setState(() {
      _measuring = true;
      _speedReport = null;
    });
    const text = 'Light from the sun looks white, but it is really a mixture of every colour. '
        'Blue light is scattered far more than red, so the sky glows blue in every direction.';
    final voice = app.resolve('char:narrator').voice;
    final results = <int, double>{};
    final before = app.settings.threads;
    try {
      for (final threads in [2, 4, 6, 8]) {
        app.settings.threads = threads;
        await app.synth.speak(voice, 'Warm up.'); // loads the model with these threads
        final watch = Stopwatch()..start();
        final pcm = await app.synth.speak(voice, text);
        final audio = pcm.length / outputRate;
        results[threads] = audio / (watch.elapsedMilliseconds / 1000);
      }
      final best = results.entries.reduce((a, b) => a.value >= b.value ? a : b);
      await app.settings.update((s) => s.threads = best.key);
      setState(() {
        _speedReport = [
          for (final e in results.entries) '${e.key} threads: ${e.value.toStringAsFixed(1)}x real time',
          '',
          'Using ${best.key} threads. ${best.value >= 1.2 ? 'Fast enough to listen live.' : 'Slower than real time: '
              'playback may pause to catch up. Try the Supertonic voices, or export first and listen to the file.'}',
        ].join('\n');
      });
    } catch (e) {
      app.settings.threads = before;
      setState(() => _speedReport = 'Speed test failed: $e');
    } finally {
      if (mounted) setState(() => _measuring = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListenableBuilder(
        listenable: app,
        builder: (context, _) {
          final s = app.settings;
          var storage = 0;
          for (final id in voicePacks.keys) {
            storage += app.models.diskBytes(id);
          }
          return ListView(
            padding: const EdgeInsets.symmetric(vertical: 8),
            children: [
              const ListTile(
                title: Text('Voices'),
                subtitle: Padding(padding: EdgeInsets.only(top: 8), child: VoiceSetSwitch()),
              ),
              ListTile(
                title: const Text('Reading speed'),
                subtitle: Slider(
                  value: s.speed,
                  min: 0.7,
                  max: 1.5,
                  divisions: 16,
                  label: '${s.speed.toStringAsFixed(2)}x',
                  onChanged: (v) => s.update((x) => x.speed = v),
                ),
                trailing: Text('${s.speed.toStringAsFixed(2)}x'),
              ),
              SwitchListTile(
                title: const Text('Clarity (HD sound)'),
                subtitle: Text(app.models.isInstalled('clarity')
                    ? 'Rebuilds the crisp top end of muffled voices, like ones cloned from old or phone-quality '
                        'clips (full 48 kHz sound). Voices that already have it are left as they are.'
                    : 'Download Clarity in the Voices tab to make muffled cloned voices crisper.'),
                value: s.clarity,
                onChanged: (v) => s.update((x) => x.clarity = v),
              ),
              SwitchListTile(
                title: const Text('Best quality'),
                subtitle: const Text('More refinement steps for the Studio and cloned voices: cleaner, '
                    'slightly slower. Turn off only if playback can\'t keep up.'),
                value: s.bestQuality,
                onChanged: (v) => s.update((x) => x.bestQuality = v),
              ),
              SwitchListTile(
                title: const Text('Music and sound cues'),
                subtitle: const Text('Intro/outro music and transitions in podcast-style scripts'),
                value: s.music,
                onChanged: (v) => s.update((x) => x.music = v),
              ),
              const Divider(),
              ListTile(
                title: const Text('Speech speed test'),
                subtitle: Text(_speedReport ?? 'Finds the fastest setting for this phone (${s.threads} threads now).'),
                trailing: _measuring
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                    : FilledButton.tonal(onPressed: _measure, child: const Text('Run')),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Use the AI chip (experimental)'),
                subtitle: const Text(
                  'Lifelike voices: runs part of VibeVoice on the phone\'s Qualcomm AI chip (Hexagon NPU), if it has one, '
                  'and falls back to the processor if it can\'t. Run the test below first.',
                ),
                value: s.aiChip,
                onChanged: (v) => s.update((x) => x.aiChip = v),
              ),
              ListTile(
                title: const Text('Test the AI chip'),
                subtitle: Text(_accelReport ??
                    'Times the lifelike voices on the processor, the AI chip and the graphics chip, and saves a report to Downloads.'),
                trailing: _testingAccel
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                    : FilledButton.tonal(onPressed: _testAccel, child: const Text('Run')),
              ),
              const Divider(),
              ListTile(
                title: const Text('Voice files on this phone'),
                subtitle: Text(formatBytes(storage)),
              ),
              const Divider(),
              ListTile(
                title: const Text('About the voices'),
                subtitle: Text(
                  'Kokoro-82M (Apache-2.0) by hexgrad; Supertonic 3 (OpenRAIL) by Supertone; Pocket TTS (CC-BY-4.0) by '
                  'Kyutai; all run with sherpa-onnx (Apache-2.0). Clarity is LavaSR (Apache-2.0) by ysharma3501. Voice Gallery clips: kyutai/tts-voices (CC0 voice '
                  'donations, VCTK CC-BY-4.0, Voice-Zero CC0, Alba MacKenna CC-BY-4.0). Music is generated by the app.\n\n'
                  'Characters are voice styles, not imitations of real people. ${builtInCharacters.length} are built in.',
                  style: text.bodySmall,
                ),
              ),
              AboutListTile(
                applicationName: 'Narrator Studio',
                applicationVersion: '1.0',
                aboutBoxChildren: const [
                  Text('Reads your books, notes and scripts aloud with free on-device voices, and turns them into podcasts.'),
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}
