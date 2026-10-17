import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/render/music.dart';
import '../core/render/styles.dart';
import 'scope.dart';
import 'theme.dart';
import 'voice_picker.dart';
import 'voice_set.dart';

/// Style, music and who reads what. Returns true when something changed.
Future<bool?> showCastSheet(BuildContext context, Session session) {
  return showModalBottomSheet<bool>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    useSafeArea: true,
    builder: (_) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.85,
      maxChildSize: 0.95,
      builder: (context, controller) => _CastSheet(session: session, controller: controller),
    ),
  );
}

class _CastSheet extends StatefulWidget {
  const _CastSheet({required this.session, required this.controller});
  final Session session;
  final ScrollController controller;

  @override
  State<_CastSheet> createState() => _CastSheetState();
}

class _CastSheetState extends State<_CastSheet> {
  late String _style = widget.session.style.id;
  late String _music = widget.session.music ? (widget.session.theme?.id ?? 'curious') : 'off';
  late final Map<String?, String> _refs = Map.of(widget.session.castRefs);

  /// Speakers whose voice you picked here (null = narrator).
  final _picked = <String?>{};
  bool _changed = false;

  Future<void> _apply() async {
    final doc = widget.session.doc;
    final app = context.app;
    final nav = Navigator.of(context);
    if (_changed) {
      if (_style != widget.session.style.id) {
        // A new style brings its own default cast, except voices you picked.
        doc.cast.clear();
        doc.narrator = null;
      }
      doc.style = _style;
      doc.music = _music;
      for (final speaker in _picked) {
        if (speaker == null) {
          doc.narrator = _refs[null];
        } else {
          doc.cast[speaker] = _refs[speaker]!;
        }
      }
      await app.library.changed(doc);
    }
    nav.pop(_changed);
  }

  /// A voice you chose yourself (here or before) is never swapped by the voice set.
  bool _pinned(String? speaker) {
    final doc = widget.session.doc;
    return _picked.contains(speaker) || (speaker == null ? doc.narrator != null : doc.cast.containsKey(speaker));
  }

  Future<void> _pick(String? speaker) async {
    final ref = await pickVoice(context, title: 'Voice for ${speaker ?? 'the narrator'}', current: _refs[speaker]);
    if (ref == null) return;
    setState(() {
      _refs[speaker] = ref;
      _picked.add(speaker);
      _changed = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    final text = Theme.of(context).textTheme;
    final speakers = widget.session.script.speakers;
    return Column(
      children: [
        Expanded(
          child: ListView(
            controller: widget.controller,
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            children: [
              Text('Style', style: text.titleMedium),
              const SizedBox(height: 8),
              RadioGroup<String>(
                groupValue: _style,
                onChanged: (v) => setState(() {
                  _style = v!;
                  _music = styleById(v).music;
                  _changed = true;
                }),
                child: Column(
                  children: [
                    for (final s in readingStyles)
                      RadioListTile<String>(
                        value: s.id,
                        contentPadding: EdgeInsets.zero,
                        title: Text(s.name),
                        subtitle: Text(s.tagline),
                      ),
                  ],
                ),
              ),
              const Divider(height: 28),
              Text('Music & sounds', style: text.titleMedium),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  ChoiceChip(
                    label: const Text('Off'),
                    selected: _music == 'off',
                    onSelected: (_) => setState(() {
                      _music = 'off';
                      _changed = true;
                    }),
                  ),
                  for (final t in musicThemes)
                    ChoiceChip(
                      label: Text(t.name),
                      tooltip: t.mood,
                      selected: _music == t.id,
                      onSelected: (_) => setState(() {
                        _music = t.id;
                        _changed = true;
                      }),
                    ),
                ],
              ),
              if (!app.settings.music)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text('Music is switched off in Settings.', style: text.bodySmall),
                ),
              const Divider(height: 28),
              Text('Voices', style: text.titleMedium),
              const SizedBox(height: 8),
              VoiceSetSwitch(
                reloadPlayer: false,
                onChanged: () => setState(() => _changed = true),
              ),
              const SizedBox(height: 4),
              _CastRow(label: 'Narrator', colorIndex: -1, ref: _refs[null]!, pinned: _pinned(null), onTap: () => _pick(null)),
              for (var i = 0; i < speakers.length; i++)
                _CastRow(
                  label: speakers[i],
                  colorIndex: i,
                  ref: _refs[speakers[i]] ?? _refs[null]!,
                  pinned: _pinned(speakers[i]),
                  onTap: () => _pick(speakers[i]),
                ),
              const SizedBox(height: 8),
              Text(
                'Tip: name voices in the script header (voices: HOST: Radio Host) so they stick when you re-import.',
                style: text.bodySmall,
              ),
            ],
          ),
        ),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 4, 20, 12),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton(onPressed: _apply, child: Text(_changed ? 'Apply' : 'Done')),
            ),
          ),
        ),
      ],
    );
  }
}

class _CastRow extends StatelessWidget {
  const _CastRow({required this.label, required this.colorIndex, required this.ref, required this.pinned, required this.onTap});
  final String label;
  final int colorIndex;
  final String ref;
  final bool pinned;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final app = context.app;
    String detail = '';
    try {
      detail = app.resolve(ref, pinned: pinned).voice.name;
    } catch (_) {}
    final name = app.refName(ref);
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: CircleAvatar(
        backgroundColor: speakerColor(context, colorIndex).withValues(alpha: 0.18),
        child: Icon(Icons.person, color: speakerColor(context, colorIndex)),
      ),
      title: Text(label),
      subtitle: Text(name == detail || detail.isEmpty ? name : '$name ($detail)'),
      trailing: const Icon(Icons.chevron_right),
      onTap: onTap,
    );
  }
}
