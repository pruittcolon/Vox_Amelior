import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;

import '../core/library/settings.dart';
import '../core/render/audio.dart';
import '../core/render/wav.dart';
import '../core/tts/voices.dart';
import 'scope.dart';
import 'voices_screen.dart';

/// The "Rainbow Passage" (public domain), used for voice samples because it
/// covers most English sounds.
const _readAloud =
    'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. '
    'The rainbow is a division of white light into many beautiful colours. These take the shape of a long '
    'round arch, with its path high above, and its two ends apparently beyond the horizon.';

class CloneScreen extends StatefulWidget {
  const CloneScreen({super.key});

  @override
  State<CloneScreen> createState() => _CloneScreenState();
}

class _CloneScreenState extends State<CloneScreen> {
  final _name = TextEditingController(text: 'My voice');
  bool _recording = false;
  double _level = 0;
  int _seconds = 0;
  Timer? _timer;
  Float32List? _clip;
  String? _tmpPath;
  bool _consent = false;
  late final _native = context.scope.native;

  @override
  void initState() {
    super.initState();
    _consent = context.app.settings.cloneConsent;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _name.dispose();
    if (_recording) _native.recordStop();
    super.dispose();
  }

  Future<void> _record() async {
    final native = _native;
    final player = context.scope.player;
    if (_recording) return _stop();
    if (!await native.requestPermission('mic')) {
      if (mounted) context.toast('Microphone permission is needed to record a sample.');
      return;
    }
    await player.stop();
    try {
      await native.recordStart(sampleRate: outputRate);
    } catch (e) {
      if (mounted) context.toast('$e');
      return;
    }
    setState(() {
      _recording = true;
      _seconds = 0;
      _clip = null;
    });
    final start = DateTime.now();
    _timer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      final level = await native.recordLevel();
      if (!mounted) return;
      final secs = DateTime.now().difference(start).inSeconds;
      setState(() {
        _level = level;
        _seconds = secs;
      });
      if (secs >= 14) await _stop();
    });
  }

  Future<void> _stop() async {
    _timer?.cancel();
    if (!_recording) return;
    final (bytes, rate) = await _native.recordStop();
    setState(() => _recording = false);
    await _useAudio(fromInt16Bytes(bytes), rate);
  }

  Future<void> _pickFile() async {
    final native = _native;
    final picked = await FilePicker.pickFiles(type: FileType.audio);
    final path = picked.isEmpty ? null : picked.first.path;
    if (path == null) return;
    try {
      final (pcm, rate) = await native.decodeAudio(path, maxSeconds: 30);
      await _useAudio(pcm, rate);
    } catch (e) {
      if (mounted) context.toast('Could not read that audio file: $e');
    }
  }

  /// Keeps the clearest ~12 s, at 24 kHz, normalized.
  Future<void> _useAudio(Float32List pcm, int rate) async {
    final preview = context.scope.preview;
    final dir = context.app.cacheDir..createSync(recursive: true);
    var x = rate == outputRate ? pcm : resample(pcm, rate, outputRate);
    x = trimSilence(x, outputRate, threshold: 0.02, keepMs: 80);
    if (x.length < outputRate * 3) {
      if (mounted) context.toast('That sample is too short. Aim for 8 to 12 seconds of clear speech.');
      return;
    }
    if (x.length > outputRate * 12) x = Float32List.sublistView(x, 0, outputRate * 12);
    x = normalizeSpeech(Float32List.fromList(x), outputRate, target: 0.12);
    fadeEdges(x, outputRate, ms: 20);
    final tmp = p.join(dir.path, 'clone-preview.wav');
    await File(tmp).writeAsBytes(encodeWav(x, outputRate));
    preview.forget('clone:preview');
    if (!mounted) return;
    setState(() {
      _clip = x;
      _tmpPath = tmp;
    });
  }

  Future<void> _try() async {
    final path = _tmpPath;
    if (path == null) return;
    final preview = context.scope.preview;
    await preview.play(Voice(
      id: 'clone:preview',
      name: _name.text,
      engine: Engine.pocket,
      blurb: '',
      refPath: path,
    ));
    if (preview.error != null && mounted) context.toast(preview.error!);
  }

  Future<void> _save() async {
    final app = context.app;
    final nav = Navigator.of(context);
    final clip = _clip;
    if (clip == null) return;
    final id = DateTime.now().millisecondsSinceEpoch.toRadixString(36);
    final file = '$id.wav';
    await app.clonesDir.create(recursive: true);
    await File(p.join(app.clonesDir.path, file)).writeAsBytes(encodeWav(clip, outputRate));
    final name = _name.text.trim().isEmpty ? 'My voice' : _name.text.trim();
    await app.settings.update((s) {
      s.cloneConsent = true;
      s.clones.add(ClonedVoice(id: id, name: name, file: file, created: DateTime.now()));
    });
    if (mounted) context.toast('$name saved. Pick it in "Voices & style" for any speaker.');
    nav.pop();
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final ready = app.models.isInstalled('pocket');
    return Scaffold(
      appBar: AppBar(title: const Text('Clone a voice')),
      body: ListenableBuilder(
        listenable: Listenable.merge([app, context.scope.preview]),
        builder: (context, _) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
          children: [
            if (!ready) ...[const PackCard(packId: 'pocket'), const SizedBox(height: 16)],
            Card(
              color: scheme.secondaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Only voices you have the right to use', style: text.titleSmall),
                    const SizedBox(height: 6),
                    const Text(
                      'Clone your own voice, someone who has given you permission, or a recording that is in the public '
                      'domain. Don\'t clone celebrities or anyone without consent, and say the voice is AI-generated '
                      'when you publish. For famous figures, use a character voice and a clearly labelled dramatization.',
                    ),
                    CheckboxListTile(
                      contentPadding: EdgeInsets.zero,
                      value: _consent,
                      onChanged: (v) => setState(() => _consent = v ?? false),
                      title: const Text('I have the right to use this voice'),
                      controlAffinity: ListTileControlAffinity.leading,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            TextField(controller: _name, decoration: const InputDecoration(labelText: 'Name for this voice')),
            const SizedBox(height: 16),
            Text('1. Record about 10 seconds', style: text.titleMedium),
            const SizedBox(height: 6),
            Text('In a quiet room, phone 20 cm away, read this naturally:', style: text.bodySmall),
            const SizedBox(height: 6),
            Card(child: Padding(padding: const EdgeInsets.all(14), child: Text(_readAloud, style: text.bodyLarge))),
            const SizedBox(height: 12),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: _consent && ready ? _record : null,
                  style: _recording ? FilledButton.styleFrom(backgroundColor: scheme.error) : null,
                  icon: Icon(_recording ? Icons.stop : Icons.mic),
                  label: Text(_recording ? 'Stop (${_seconds}s)' : 'Record'),
                ),
                const SizedBox(width: 12),
                if (_recording) Expanded(child: LinearProgressIndicator(value: _level)),
                if (!_recording)
                  TextButton.icon(
                    onPressed: _consent && ready ? _pickFile : null,
                    icon: const Icon(Icons.audio_file_outlined),
                    label: const Text('Or pick an audio file'),
                  ),
              ],
            ),
            if (_clip != null) ...[
              const SizedBox(height: 24),
              Text('2. Listen, then save', style: text.titleMedium),
              const SizedBox(height: 6),
              Text('Sample: ${(_clip!.length / outputRate).toStringAsFixed(1)} s', style: text.bodySmall),
              const SizedBox(height: 8),
              Row(
                children: [
                  OutlinedButton.icon(
                    onPressed: _try,
                    icon: Icon(context.scope.preview.busy == 'clone:preview' ? Icons.stop : Icons.play_arrow),
                    label: const Text('Hear it speak'),
                  ),
                  const SizedBox(width: 12),
                  FilledButton.icon(onPressed: _save, icon: const Icon(Icons.save_outlined), label: const Text('Save voice')),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
