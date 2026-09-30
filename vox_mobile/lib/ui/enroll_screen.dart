import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/speakers/enrollment_service.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

const _prompts = [
  'The quick brown fox jumps over the lazy dog near the riverbank.',
  'Could you remind me to call the plumber tomorrow morning?',
  'We need milk, eggs, bread and some fresh vegetables this week.',
  'My favourite place to relax is the garden on a sunny afternoon.',
  'Please turn the lights down and play something calm in the living room.',
  'Read any paragraph from a book or article in your normal voice.',
];

/// Teaches Vox a person's voice from recordings or imported WAV files.
/// Pass [person] to add more samples to someone already enrolled.
class EnrollScreen extends StatefulWidget {
  const EnrollScreen({super.key, required this.services, this.person});

  final AppServices services;
  final SpeakerProfile? person;

  @override
  State<EnrollScreen> createState() => _EnrollScreenState();
}

class _EnrollScreenState extends State<EnrollScreen> {
  static const _recordSeconds = 8;

  final _name = TextEditingController();
  final List<Float32List> _recorded = [];
  final List<String> _files = [];
  AudioRecorder? _recorder;
  StreamSubscription<Uint8List>? _mic;
  final BytesBuilder _buffer = BytesBuilder(copy: false);
  Timer? _timer;
  int _secondsLeft = 0;
  bool _busy = false;
  bool _resumeService = false;

  AppServices get s => widget.services;
  bool get _recording => _mic != null;

  @override
  void dispose() {
    _timer?.cancel();
    unawaited(_mic?.cancel());
    unawaited(_recorder?.dispose());
    if (_resumeService) s.listening.resume();
    _name.dispose();
    super.dispose();
  }

  Future<void> _record() async {
    if (_recording) return _finishRecording();
    final recorder = _recorder ??= AudioRecorder();
    if (!await recorder.hasPermission()) {
      if (mounted) showMessage(context, 'Microphone permission is needed.');
      return;
    }
    // The listening service holds the microphone; borrow it.
    _resumeService = await s.listening.pauseForRecording() || _resumeService;
    _buffer.clear();
    final stream = await recorder.startStream(const RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      androidConfig: AndroidRecordConfig(audioSource: AndroidAudioSource.voiceRecognition),
    ));
    setState(() {
      _secondsLeft = _recordSeconds;
      _mic = stream.listen(_buffer.add);
    });
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _secondsLeft--);
      if (_secondsLeft <= 0) unawaited(_finishRecording());
    });
  }

  Future<void> _finishRecording() async {
    _timer?.cancel();
    await _recorder?.stop();
    await _mic?.cancel();
    final bytes = _buffer.takeBytes();
    final data = ByteData.sublistView(bytes);
    final samples = Float32List(bytes.length ~/ 2);
    for (var i = 0; i < samples.length; i++) {
      samples[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
    }
    if (!mounted) return;
    setState(() {
      _mic = null;
      if (samples.length > 16000) _recorded.add(samples);
    });
  }

  Future<void> _pickFiles() async {
    final picked = await FilePicker.pickFiles(type: FileType.custom, allowedExtensions: const ['wav']);
    final paths = picked.map((f) => f.path).whereType<String>().toList();
    if (paths.isEmpty || !mounted) return;
    setState(() => _files.addAll(paths));
  }

  Future<void> _save() async {
    final worker = s.voiceprints;
    if (worker == null) {
      showMessage(context, 'Download the speech models first.');
      return;
    }
    final name = _name.text.trim();
    if (widget.person == null && name.isEmpty) {
      showMessage(context, 'Enter a name.');
      return;
    }
    if (_recorded.isEmpty && _files.isEmpty) {
      showMessage(context, 'Record or import some voice samples first.');
      return;
    }
    setState(() => _busy = true);
    try {
      var report = await worker.analyze(_recorded);
      var fileSamples = 0;
      if (_files.isNotEmpty) {
        final (fileReport, pieces) = await worker.analyzeWavFiles(_files);
        fileSamples = pieces;
        report = EnrollmentReport(
          embeddings: [...report.embeddings, ...fileReport.embeddings],
          rejected: [...report.rejected, ...fileReport.rejected],
          embeddingModel: report.embeddingModel,
        );
      }
      final service = EnrollmentService(s.speakers);
      final who = widget.person;
      if (who == null) {
        service.enroll(name, report);
      } else {
        service.addSamples(who.id, report);
      }
      s.dataChanged();
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Voice saved'),
          content: Text(
            '${report.embeddings.length} sample(s) used'
            '${fileSamples > 0 ? ' ($fileSamples from files)' : ''}.'
            '${report.rejected.isEmpty ? '' : '\n${report.rejected.length} skipped (too short, too quiet, or sounded like someone else).'}',
          ),
          actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text('Done'))],
        ),
      );
      if (mounted) Navigator.pop(context);
    } on EnrollmentException catch (e) {
      if (mounted) showMessage(context, e.message);
    } on Object catch (e) {
      if (mounted) showMessage(context, 'Could not process the audio: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final prompt = _prompts[_recorded.length % _prompts.length];
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(widget.person == null ? 'Add a person' : 'More samples for ${widget.person!.name}')),
      body: AbsorbPointer(
        absorbing: _busy,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (widget.person == null)
              TextField(
                controller: _name,
                textCapitalization: TextCapitalization.words,
                decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
              ),
            const SizedBox(height: 16),
            Text('Record', style: theme.textTheme.titleMedium),
            const Text('Record at least 3 samples (5 is better), one person at a time, in a quiet room.'),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    Text('Read aloud:', style: theme.textTheme.labelLarge),
                    const SizedBox(height: 8),
                    Text('"$prompt"', textAlign: TextAlign.center, style: theme.textTheme.titleMedium),
                    const SizedBox(height: 16),
                    FilledButton.icon(
                      onPressed: _record,
                      icon: Icon(_recording ? Icons.stop : Icons.mic),
                      label: Text(_recording ? 'Recording… $_secondsLeft s (tap to stop)' : 'Record sample ${_recorded.length + 1}'),
                    ),
                  ],
                ),
              ),
            ),
            for (var i = 0; i < _recorded.length; i++)
              ListTile(
                leading: const Icon(Icons.graphic_eq),
                title: Text('Sample ${i + 1}'),
                subtitle: Text('${(_recorded[i].length / 16000).toStringAsFixed(1)} s'),
                trailing: IconButton(icon: const Icon(Icons.delete_outline), onPressed: () => setState(() => _recorded.removeAt(i))),
              ),
            const Divider(height: 32),
            Text('Or import recordings', style: theme.textTheme.titleMedium),
            const Text('Pick WAV files of this person speaking. Long files are split automatically.'),
            const SizedBox(height: 8),
            OutlinedButton.icon(onPressed: _pickFiles, icon: const Icon(Icons.folder_open), label: const Text('Choose WAV files')),
            for (final f in _files)
              ListTile(
                leading: const Icon(Icons.audio_file),
                title: Text(f.split('/').last),
                trailing: IconButton(icon: const Icon(Icons.close), onPressed: () => setState(() => _files.remove(f))),
              ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _busy || _recording ? null : _save,
              child: _busy
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Save voice'),
            ),
          ],
        ),
      ),
    );
  }
}
