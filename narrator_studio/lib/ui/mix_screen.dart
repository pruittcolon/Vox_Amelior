import 'package:flutter/material.dart';

import '../core/tts/voice_mix.dart';
import '../core/tts/voices.dart';
import 'scope.dart';
import 'voices_screen.dart';

/// Blend two Kokoro voices into a new one.
class MixScreen extends StatefulWidget {
  const MixScreen({super.key});

  @override
  State<MixScreen> createState() => _MixScreenState();
}

class _MixScreenState extends State<MixScreen> {
  String _a = 'bm_george';
  String _b = 'am_onyx';
  double _weight = 0.6;
  final _name = TextEditingController(text: 'Old Professor');
  bool _working = false;

  /// Voices worth mixing (the English ones plus accents; Mandarin slots are
  /// reserved for mixes).
  List<Voice> get _choices => kokoroVoices.where((v) => !kokoroMixSlots.contains(v.id.substring(7))).toList();

  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  Future<VoiceMix?> _write({required bool preview}) async {
    final app = context.app;
    final slot = app.freeMixSlot();
    if (slot == null) {
      context.toast('You have ${kokoroMixSlots.length} mixes already. Remove one first.');
      return null;
    }
    final mix = VoiceMix(
      id: preview ? 'preview' : DateTime.now().millisecondsSinceEpoch.toRadixString(36),
      name: _name.text.trim().isEmpty ? 'My mix' : _name.text.trim(),
      parts: {_a: _weight, _b: 1 - _weight},
      slot: slot,
    );
    setState(() => _working = true);
    try {
      await app.settings.update((s) {
        s.mixes.removeWhere((m) => m.id == 'preview');
        s.mixes.add(mix);
      });
      await app.rebuildMixes();
      return mix;
    } finally {
      if (mounted) setState(() => _working = false);
    }
  }

  Future<void> _removePreview() async {
    final app = context.app;
    if (app.settings.mixes.any((m) => m.id == 'preview')) {
      await app.settings.update((s) => s.mixes.removeWhere((m) => m.id == 'preview'));
      await app.rebuildMixes();
    }
  }

  Future<void> _try() async {
    final preview = context.scope.preview;
    final mix = await _write(preview: true);
    if (mix == null) return;
    preview.forget(mix.voiceId);
    await preview.play(mix.toVoice());
    if (preview.error != null && mounted) context.toast(preview.error!);
  }

  Future<void> _save() async {
    final nav = Navigator.of(context);
    await _removePreview();
    final mix = await _write(preview: false);
    if (mix == null) return;
    if (mounted) context.toast('${mix.name} saved: ${mix.recipe}.');
    nav.pop();
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final text = Theme.of(context).textTheme;
    final ready = app.hasKokoro;
    return PopScope(
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) _removePreview();
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Mix voices')),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
          children: [
            if (!ready) ...[const PackCard(packId: 'kokoro'), const SizedBox(height: 16)],
            Text(
              'Kokoro describes every voice as a set of numbers. Blending two sets makes a voice that sits between '
              'them: try a British and an American voice, or a deep and a bright one.',
              style: text.bodyMedium,
            ),
            const SizedBox(height: 20),
            _VoiceDropdown(label: 'First voice', value: _a, choices: _choices, onChanged: (v) => setState(() => _a = v)),
            const SizedBox(height: 12),
            _VoiceDropdown(label: 'Second voice', value: _b, choices: _choices, onChanged: (v) => setState(() => _b = v)),
            const SizedBox(height: 16),
            Text(
              '${(_weight * 100).round()}% ${_nameOf(_a)} + ${((1 - _weight) * 100).round()}% ${_nameOf(_b)}',
              style: text.titleSmall,
            ),
            Slider(value: _weight, min: 0.1, max: 0.9, divisions: 16, onChanged: (v) => setState(() => _weight = v)),
            const SizedBox(height: 8),
            TextField(controller: _name, decoration: const InputDecoration(labelText: 'Name')),
            const SizedBox(height: 20),
            ListenableBuilder(
              listenable: context.scope.preview,
              builder: (context, _) => Row(
                children: [
                  OutlinedButton.icon(
                    onPressed: ready && !_working ? _try : null,
                    icon: _working
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.play_arrow),
                    label: const Text('Try it'),
                  ),
                  const SizedBox(width: 12),
                  FilledButton.icon(
                    onPressed: ready && !_working ? _save : null,
                    icon: const Icon(Icons.save_outlined),
                    label: const Text('Save mix'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _nameOf(String id) => kokoroVoices.firstWhere((v) => v.id == 'kokoro:$id').name;
}

class _VoiceDropdown extends StatelessWidget {
  const _VoiceDropdown({required this.label, required this.value, required this.choices, required this.onChanged});
  final String label;
  final String value;
  final List<Voice> choices;
  final void Function(String) onChanged;

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField<String>(
      initialValue: value,
      isExpanded: true,
      decoration: InputDecoration(labelText: label),
      items: [
        for (final v in choices)
          DropdownMenuItem(value: v.id.substring(7), child: Text('${v.name} · ${v.group}', overflow: TextOverflow.ellipsis)),
      ],
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
    );
  }
}
