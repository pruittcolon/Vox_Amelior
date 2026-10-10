import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/native/sherpa_engines.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// What to do with a voice that has no name.
sealed class VoiceChoice {
  const VoiceChoice();
}

/// It is someone already in the household.
class NameAsPerson extends VoiceChoice {
  const NameAsPerson(this.speakerId);
  final String speakerId;
}

/// It is someone new, called [name].
class NameAsNew extends VoiceChoice {
  const NameAsNew(this.name);
  final String name;
}

/// It is the TV, radio or other background voice.
class MarkAsBackground extends VoiceChoice {
  const MarkAsBackground();
}

/// Asks who guest voice [clusterId] is (showing what it said) and names
/// everything it said in one go, with Undo. True if it was named or hidden.
Future<bool> nameVoice(BuildContext context, AppServices s, String clusterId) async {
  final voice = s.transcripts.voiceToName(clusterId);
  if (voice == null) {
    showMessage(context, 'That voice has no lines left.');
    return false;
  }
  final choice = await showModalBottomSheet<VoiceChoice>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    builder: (c) => SafeArea(
      child: Padding(
        // Room for the keyboard when typing a new name.
        padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(c).bottom),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(c).height * 0.9),
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            child: VoiceNamer(voice: voice, people: s.speakers.profiles(), onChoice: (choice) => Navigator.pop(c, choice)),
          ),
        ),
      ),
    ),
  );
  if (choice == null || !context.mounted) return false;
  return applyVoiceChoice(context, s, voice, choice);
}

/// Carries out [choice] for [voice], then offers Undo.
bool applyVoiceChoice(BuildContext context, AppServices s, VoiceToName voice, VoiceChoice choice) {
  final String done;
  final void Function() undo;
  try {
    switch (choice) {
      case NameAsPerson(:final speakerId):
        final n = s.speakers.nameVoice(voice.clusterId, speakerId);
        final name = s.speakers.profile(speakerId)?.name ?? 'them';
        done = '${voice.label} is $name: ${_lines(voice.lines)} named';
        undo = () => s.speakers.undoNaming(n);
      case NameAsNew(:final name):
        final n = s.speakers.nameVoiceAsNew(voice.clusterId, name, embeddingModel: SherpaSpeakerEmbedder.modelIdConst);
        done = 'Added $name: ${_lines(voice.lines)} named. Vox will know this voice from now on.';
        undo = () => s.speakers.undoNaming(n);
      case MarkAsBackground():
        s.speakers.setClusterBackground(voice.clusterId, true);
        done = '${voice.label} is hidden as TV / background';
        undo = () => s.speakers.setClusterBackground(voice.clusterId, false);
    }
  } on StateError catch (e) {
    showMessage(context, e.message);
    return false;
  }
  s.dataChanged();
  final messenger = ScaffoldMessenger.of(context);
  messenger
    ..hideCurrentSnackBar()
    ..showSnackBar(SnackBar(
      content: Text(done),
      action: SnackBarAction(
        label: 'Undo',
        onPressed: () {
          undo();
          s.dataChanged();
        },
      ),
    ));
  return true;
}

String _lines(int n) => n == 1 ? '1 line' : '$n lines';

/// Who a voice is: what it said, then one tap on a person (or a new name,
/// or TV / background). Used in a sheet and in [NameVoicesScreen].
class VoiceNamer extends StatefulWidget {
  const VoiceNamer({super.key, required this.voice, required this.people, required this.onChoice, this.onSkip, this.onOpenLine});

  final VoiceToName voice;
  final List<SpeakerProfile> people;
  final ValueChanged<VoiceChoice> onChoice;

  /// Shows a "Not now" button.
  final VoidCallback? onSkip;

  /// Opens a sample line in its conversation.
  final ValueChanged<SegmentView>? onOpenLine;

  @override
  State<VoiceNamer> createState() => _VoiceNamerState();
}

class _VoiceNamerState extends State<VoiceNamer> {
  final _name = TextEditingController();
  bool _typing = false;
  String? _error;

  @override
  void didUpdateWidget(VoiceNamer old) {
    super.didUpdateWidget(old);
    if (old.voice.clusterId != widget.voice.clusterId) {
      _name.clear();
      _typing = false;
      _error = null;
    }
  }

  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  void _saveNew() {
    final name = _name.text.trim();
    if (name.isEmpty) {
      setState(() => _error = 'Type their name');
      return;
    }
    final same = widget.people.where((p) => p.name.toLowerCase() == name.toLowerCase()).firstOrNull;
    if (same != null) {
      // Already in the household: that is who it is.
      widget.onChoice(NameAsPerson(same.id));
      return;
    }
    widget.onChoice(NameAsNew(name));
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final v = widget.voice;
    final muted = t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            SpeakerAvatar(label: v.label, known: false, radius: 24),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Who is ${v.label}?', style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
                  Text(
                    '${_lines(v.lines)} · last heard ${formatWhen(v.lastHeard)}'
                    '${v.heardWith.isEmpty ? '' : ' · with ${v.heardWith.join(', ')}'}',
                    style: muted,
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        for (final line in v.samples)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Material(
              color: t.colorScheme.surfaceContainerHigh,
              borderRadius: BorderRadius.circular(14),
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: widget.onOpenLine == null ? null : () => widget.onOpenLine!(line),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('"${line.text}"', maxLines: 3, overflow: TextOverflow.ellipsis, style: t.textTheme.bodyLarge),
                      const SizedBox(height: 2),
                      Text(formatWhen(line.startedAt), style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),
            ),
          ),
        const SizedBox(height: 8),
        Text("It's…", style: t.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final p in widget.people)
              ActionChip(
                avatar: SpeakerAvatar(label: p.name, radius: 12),
                label: Text(p.name),
                onPressed: () => widget.onChoice(NameAsPerson(p.id)),
              ),
            ActionChip(
              avatar: Icon(Icons.person_add_alt_1_rounded, size: 18, color: t.colorScheme.primary),
              label: const Text('Someone new'),
              onPressed: () => setState(() => _typing = true),
            ),
          ],
        ),
        if (_typing) ...[
          const SizedBox(height: 12),
          TextField(
            controller: _name,
            autofocus: true,
            textCapitalization: TextCapitalization.words,
            textInputAction: TextInputAction.done,
            decoration: InputDecoration(
              labelText: 'Their name',
              errorText: _error,
              suffixIcon: IconButton(tooltip: 'Save name', icon: const Icon(Icons.check_rounded), onPressed: _saveNew),
            ),
            onSubmitted: (_) => _saveNew(),
          ),
        ],
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          alignment: WrapAlignment.spaceBetween,
          children: [
            TextButton.icon(
              onPressed: () => widget.onChoice(const MarkAsBackground()),
              icon: const Icon(Icons.tv_rounded),
              label: const Text('TV or background'),
            ),
            if (widget.onSkip != null) TextButton(onPressed: widget.onSkip, child: const Text('Not now')),
          ],
        ),
      ],
    );
  }
}

/// Every voice without a name, one after another: what it said, then a
/// tap on who it is. Opened from People and Now.
class NameVoicesScreen extends StatefulWidget {
  const NameVoicesScreen({super.key, required this.services, required this.openLine});

  final AppServices services;

  /// Opens a line in its conversation (kept out of this file to avoid an import cycle).
  final void Function(BuildContext context, SegmentView line) openLine;

  @override
  State<NameVoicesScreen> createState() => _NameVoicesScreenState();
}

class _NameVoicesScreenState extends State<NameVoicesScreen> {
  final Set<String> _skipped = {};
  List<VoiceToName> _waiting = const [];
  VoiceToName? _current;
  int _named = 0;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _load();
    s.dataVersion.addListener(_load);
  }

  @override
  void dispose() {
    s.dataVersion.removeListener(_load);
    super.dispose();
  }

  void _load() {
    if (!mounted) return;
    final waiting = [for (final v in s.transcripts.voicesToName(detailed: 0)) if (!_skipped.contains(v.clusterId)) v];
    setState(() {
      _waiting = waiting;
      _current = waiting.isEmpty ? null : s.transcripts.voiceToName(waiting.first.clusterId, samples: 5);
    });
  }

  void _choose(VoiceChoice choice) {
    final v = _current;
    if (v == null) return;
    if (applyVoiceChoice(context, s, v, choice)) _named++;
  }

  void _skip() {
    final v = _current;
    if (v == null) return;
    _skipped.add(v.clusterId);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final v = _current;
    return Scaffold(
      appBar: AppBar(title: const Text("Who's this?")),
      body: v == null
          ? EmptyState(
              icon: Icons.how_to_reg_rounded,
              title: _skipped.isEmpty ? 'Every voice has a name' : 'That was every voice for now',
              message: _named == 0
                  ? 'New voices appear here as Vox hears them.'
                  : 'Named ${_named == 1 ? '1 voice' : '$_named voices'}. Thank you!',
              action: FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Done')),
            )
          : ListView(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 32),
              children: [
                Text(
                  _waiting.length == 1 ? 'Last voice to name' : '${_waiting.length} voices to name',
                  style: t.textTheme.labelLarge?.copyWith(color: t.colorScheme.primary),
                ),
                const SizedBox(height: 12),
                VoiceNamer(
                  key: ValueKey(v.clusterId),
                  voice: v,
                  people: s.speakers.profiles(),
                  onChoice: _choose,
                  onSkip: _skip,
                  onOpenLine: (line) => widget.openLine(context, line),
                ),
              ],
            ),
    );
  }
}

/// "3 voices need a name": opens [NameVoicesScreen]. Nothing when every
/// voice has a name.
class VoicesToNameCard extends StatelessWidget {
  const VoicesToNameCard({super.key, required this.count, required this.onOpen, this.margin = EdgeInsets.zero});

  final int count;
  final VoidCallback onOpen;
  final EdgeInsets margin;

  @override
  Widget build(BuildContext context) {
    if (count == 0) return const SizedBox.shrink();
    final t = Theme.of(context);
    return Padding(
      padding: margin,
      child: VoxCard(
        color: t.colorScheme.tertiaryContainer,
        onTap: onOpen,
        child: Row(
          children: [
            Icon(Icons.record_voice_over_rounded, color: t.colorScheme.onTertiaryContainer),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    count == 1 ? '1 voice needs a name' : '$count voices need a name',
                    style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700, color: t.colorScheme.onTertiaryContainer),
                  ),
                  Text(
                    'See what each said, then tap who it is.',
                    style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onTertiaryContainer),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: t.colorScheme.onTertiaryContainer),
          ],
        ),
      ),
    );
  }
}
