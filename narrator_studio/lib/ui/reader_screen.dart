import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/audio/player.dart';
import '../core/library/library.dart';
import '../core/render/music.dart';
import '../core/render/plan.dart';
import '../core/script/script.dart';
import 'art.dart';
import 'cast_sheet.dart';
import 'edit_text_screen.dart';
import 'export_sheet.dart';
import 'scope.dart';
import 'theme.dart';

class ReaderScreen extends StatefulWidget {
  const ReaderScreen({super.key, required this.doc});
  final Doc doc;

  @override
  State<ReaderScreen> createState() => _ReaderScreenState();
}

class _ReaderScreenState extends State<ReaderScreen> {
  Session? _session;
  Object? _error;
  final _scroll = ScrollController();
  final _keys = <int, GlobalKey>{};
  int _shown = -1;
  DateTime _userScrolledAt = DateTime(2000);
  bool _offTrack = false;
  late final Player _player;

  @override
  void initState() {
    super.initState();
    _player = context.scope.player;
    _player.addListener(_onPlayer);
    _open();
  }

  @override
  void dispose() {
    _player.removeListener(_onPlayer);
    _scroll.dispose();
    super.dispose();
  }

  bool get _mine => _player.session?.doc.id == widget.doc.id;

  Future<void> _open({bool reload = false}) async {
    final app = context.app;
    unawaited(app.library.touch(widget.doc));
    try {
      final s = !reload && _mine ? _player.session! : await app.openSession(widget.doc);
      if (!mounted) return;
      setState(() {
        _session = s;
        _error = null;
      });
      if (reload && _mine) await _player.reload(s);
      WidgetsBinding.instance.addPostFrameCallback((_) => _follow(force: true));
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  int get _current => _mine ? _player.current : widget.doc.position;

  void _onPlayer() {
    if (!mounted) return;
    setState(() {});
    if (_mine && _player.current != _shown) _follow();
  }

  /// Keeps the sentence being read on screen (unless you are browsing).
  void _follow({bool force = false}) {
    final s = _session;
    if (s == null || s.plan.utterances.isEmpty || !_scroll.hasClients) return;
    final cur = _current.clamp(0, s.plan.utterances.length - 1);
    _shown = cur;
    if (!force && DateTime.now().difference(_userScrolledAt).inSeconds < 6) {
      if (!_offTrack) setState(() => _offTrack = true);
      return;
    }
    final block = s.plan.utterances[cur].block;
    final ctx = _keys[block]?.currentContext;
    if (ctx != null) {
      Scrollable.ensureVisible(ctx, alignment: 0.3, duration: const Duration(milliseconds: 350), curve: Curves.easeOut);
    } else {
      // Not built yet: jump near it, then align once it exists.
      final fraction = block / s.script.blocks.length;
      _scroll.jumpTo((_scroll.position.maxScrollExtent * fraction).clamp(0, _scroll.position.maxScrollExtent));
      WidgetsBinding.instance.addPostFrameCallback((_) {
        final c = _keys[block]?.currentContext;
        if (c != null) Scrollable.ensureVisible(c, alignment: 0.3);
      });
    }
    if (_offTrack) setState(() => _offTrack = false);
  }

  Future<void> _playFrom(int utterance) async {
    final s = _session;
    if (s == null) return;
    _userScrolledAt = DateTime(2000);
    if (_mine) {
      await _player.jump(utterance);
      if (_player.state == PlayState.paused) await _player.resume();
    } else {
      await _player.play(s, from: utterance);
    }
  }

  Future<void> _togglePlay() async {
    final s = _session;
    if (s == null) return;
    if (_mine && _player.state != PlayState.ended && _player.state != PlayState.error) {
      await _player.toggle();
    } else {
      await _player.play(s, from: _player.state == PlayState.ended && _mine ? 0 : widget.doc.position);
    }
  }

  Future<void> _openCast() async {
    final s = _session;
    if (s == null) return;
    final changed = await showCastSheet(context, s);
    if (changed == true) await _open(reload: true);
  }

  Future<void> _setSpeed(double v) async {
    widget.doc.speed = v;
    await context.app.library.changed(widget.doc);
    await _open(reload: true);
  }

  @override
  Widget build(BuildContext context) {
    final s = _session;
    return Scaffold(
      appBar: AppBar(
        actions: [
          IconButton(tooltip: 'Voices & style', icon: const Icon(Icons.theater_comedy_outlined), onPressed: s == null ? null : _openCast),
          IconButton(
            tooltip: 'Export audio',
            icon: const Icon(Icons.ios_share),
            onPressed: s == null ? null : () => showExportSheet(context, s),
          ),
          PopupMenuButton<String>(
            onSelected: (v) async {
              if (v == 'edit') {
                await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => EditTextScreen(doc: widget.doc)));
                if (_mine) await _player.stop();
                await _open(reload: true);
              }
            },
            itemBuilder: (_) => const [PopupMenuItem(value: 'edit', child: Text('Edit text'))],
          ),
        ],
      ),
      body: _error != null
          ? _ErrorView(error: _error!, onRetry: _open)
          : s == null
              ? const Center(child: CircularProgressIndicator())
              : NotificationListener<UserScrollNotification>(
                  onNotification: (_) {
                    _userScrolledAt = DateTime.now();
                    return false;
                  },
                  child: Stack(
                    children: [
                      _TextView(
                        session: s,
                        controller: _scroll,
                        keys: _keys,
                        current: _current,
                        playing: _mine && _player.active,
                        onTap: _playFrom,
                        onCast: _openCast,
                      ),
                      if (_offTrack && _mine && _player.isPlaying)
                        Positioned(
                          right: 16,
                          bottom: 16,
                          child: FloatingActionButton.small(
                            tooltip: 'Back to the current sentence',
                            onPressed: () {
                              _userScrolledAt = DateTime(2000);
                              _follow(force: true);
                            },
                            child: const Icon(Icons.my_location),
                          ),
                        ),
                    ],
                  ),
                ),
      bottomNavigationBar: s == null
          ? null
          : _PlayerBar(
              session: s,
              mine: _mine,
              player: _player,
              current: _current,
              onToggle: _togglePlay,
              onSeek: _playFrom,
              onSpeed: _setSpeed,
              speed: widget.doc.speed ?? context.app.settings.speed,
            ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.error, required this.onRetry});
  final Object error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final noVoices = error is NoVoicesError;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(noVoices ? Icons.record_voice_over_outlined : Icons.error_outline, size: 48),
            const SizedBox(height: 12),
            Text(
              noVoices ? 'Download a voice pack first' : 'Could not open this document',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(noVoices ? 'Go to the Voices tab and download Kokoro (or Supertonic).' : '$error', textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton(onPressed: onRetry, child: const Text('Try again')),
          ],
        ),
      ),
    );
  }
}

class _TextView extends StatelessWidget {
  const _TextView({
    required this.session,
    required this.controller,
    required this.keys,
    required this.current,
    required this.playing,
    required this.onTap,
    required this.onCast,
  });

  final Session session;
  final ScrollController controller;
  final Map<int, GlobalKey> keys;
  final int current;
  final bool playing;
  final void Function(int utterance) onTap;
  final VoidCallback onCast;

  @override
  Widget build(BuildContext context) {
    final blocks = session.script.blocks;
    final byBlock = session.plan.byBlock;
    final speakers = session.script.speakers;
    final cur = session.plan.utterances.isEmpty ? null : session.plan.utterances[current.clamp(0, session.plan.utterances.length - 1)];
    return ListView.builder(
      controller: controller,
      padding: const EdgeInsets.fromLTRB(22, 0, 22, 40),
      itemCount: blocks.length + 1,
      itemBuilder: (context, i) {
        if (i == 0) return _Header(session: session, onCast: onCast);
        final bi = i - 1;
        final b = blocks[bi];
        final key = keys.putIfAbsent(bi, GlobalKey.new);
        final prevSpeaker = bi > 0 && blocks[bi - 1].kind == BlockKind.line ? blocks[bi - 1].speaker : '\u0000';
        return KeyedSubtree(
          key: key,
          child: switch (b.kind) {
            BlockKind.heading => _Heading(block: b, utterances: byBlock[bi] ?? const [], current: cur, onTap: onTap),
            BlockKind.line => _Paragraph(
                block: b,
                utterances: byBlock[bi] ?? const [],
                current: cur,
                playing: playing,
                showSpeaker: b.speaker != null && b.speaker != prevSpeaker,
                speakerIndex: b.speaker == null ? -1 : speakers.indexOf(b.speaker!),
                speakerVoice: session.cast[b.speaker]?.voice.name,
                speakerRef: session.castRefs[b.speaker],
                onTap: onTap,
              ),
            BlockKind.pause => _Cue(icon: Icons.more_horiz, label: 'pause ${_secs(b.seconds)}'),
            BlockKind.cue => _Cue(icon: b.cue!.startsWith('music') ? Icons.music_note : Icons.notifications_none, label: b.cue!.replaceAll(':', ' · ')),
          },
        );
      },
    );
  }

  static String _secs(double s) => s == s.roundToDouble() ? '${s.round()} s' : '${s.toStringAsFixed(1)} s';
}

class _Header extends StatelessWidget {
  const _Header({required this.session, required this.onCast});
  final Session session;
  final VoidCallback onCast;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final text = theme.textTheme;
    final speed = session.cast[null]?.speed ?? 1.0;
    final est = session.plan.estimate(speed);
    final speakers = session.script.speakers;
    final narrated = speakers.isEmpty || session.script.blocks.any((b) => b.kind == BlockKind.line && b.speaker == null);
    final title = session.doc.title;
    return Padding(
      padding: const EdgeInsets.only(bottom: 22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(22),
                  boxShadow: [
                    BoxShadow(color: paletteFor(session.doc.title).a.withValues(alpha: 0.35), blurRadius: 24, offset: const Offset(0, 10)),
                  ],
                ),
                child: CoverArt(seed: session.doc.title, style: session.style.id, size: 112, radius: 22),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      session.style.name.toUpperCase(),
                      style: text.labelSmall?.copyWith(color: scheme.secondary, letterSpacing: 1.4),
                    ),
                    const SizedBox(height: 4),
                    Text(title, maxLines: 4, overflow: TextOverflow.ellipsis, style: text.headlineSmall),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _Pill(icon: Icons.schedule_rounded, label: '${session.style.name} · ~${formatDuration(est)}'),
              if (session.music) _Pill(icon: Icons.music_note_rounded, label: themeById(session.theme?.id)?.name ?? 'Music'),
            ],
          ),
          const SizedBox(height: 14),
          Material(
            color: scheme.surfaceContainerLow,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18), side: BorderSide(color: scheme.outlineVariant)),
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: onCast,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(14, 12, 10, 12),
                child: Row(
                  children: [
                    Expanded(
                      child: Wrap(
                        spacing: 14,
                        runSpacing: 10,
                        children: [
                          if (narrated)
                            _CastMember(
                              label: 'Narrator',
                              voice: session.cast[null]!.voice.name,
                              reference: session.castRefs[null],
                              color: speakerColor(context, -1),
                            ),
                          for (var i = 0; i < speakers.length; i++)
                            _CastMember(
                              label: speakers[i],
                              voice: session.cast[speakers[i]]?.voice.name ?? '',
                              reference: session.castRefs[speakers[i]],
                              color: speakerColor(context, i),
                            ),
                        ],
                      ),
                    ),
                    Icon(Icons.chevron_right_rounded, color: scheme.onSurfaceVariant),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.icon, required this.label});
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(color: scheme.secondaryContainer.withValues(alpha: 0.7), borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 15, color: scheme.onSecondaryContainer),
          const SizedBox(width: 6),
          Flexible(
            child: Text(label, style: Theme.of(context).textTheme.labelMedium?.copyWith(color: scheme.onSecondaryContainer)),
          ),
        ],
      ),
    );
  }
}

String? characterIdOf(String? ref) => ref != null && ref.startsWith('char:') ? ref.substring(5) : null;

class _CastMember extends StatelessWidget {
  const _CastMember({required this.label, required this.voice, required this.reference, required this.color});
  final String label;
  final String voice;
  final String? reference;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        VoiceAvatar(name: voice.isEmpty ? label : voice, characterId: characterIdOf(reference), size: 34),
        const SizedBox(width: 8),
        Flexible(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: text.labelMedium?.copyWith(color: color, letterSpacing: 0.6),
              ),
              Text(voice, maxLines: 1, overflow: TextOverflow.ellipsis, style: text.bodySmall),
            ],
          ),
        ),
      ],
    );
  }
}

class _Heading extends StatelessWidget {
  const _Heading({required this.block, required this.utterances, required this.current, required this.onTap});
  final Block block;
  final List<Utterance> utterances;
  final Utterance? current;
  final void Function(int) onTap;

  @override
  Widget build(BuildContext context) {
    final active = utterances.isNotEmpty && current?.index == utterances.first.index;
    final style = block.level <= 1 ? Theme.of(context).textTheme.headlineSmall : Theme.of(context).textTheme.titleLarge;
    return Padding(
      padding: const EdgeInsets.only(top: 22, bottom: 12),
      child: InkWell(
        onTap: utterances.isEmpty ? null : () => onTap(utterances.first.index),
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          decoration: active ? BoxDecoration(color: readingHighlight(context).$1, borderRadius: BorderRadius.circular(8)) : null,
          child: Text(block.text, style: style),
        ),
      ),
    );
  }
}

class _Paragraph extends StatefulWidget {
  const _Paragraph({
    required this.block,
    required this.utterances,
    required this.current,
    required this.playing,
    required this.showSpeaker,
    required this.speakerIndex,
    required this.speakerVoice,
    required this.speakerRef,
    required this.onTap,
  });

  final Block block;
  final List<Utterance> utterances;
  final Utterance? current;
  final bool playing;
  final bool showSpeaker;
  final int speakerIndex;
  final String? speakerVoice;
  final String? speakerRef;
  final void Function(int) onTap;

  @override
  State<_Paragraph> createState() => _ParagraphState();
}

class _ParagraphState extends State<_Paragraph> {
  final _recognizers = <TapGestureRecognizer>[];

  @override
  void dispose() {
    for (final r in _recognizers) {
      r.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    for (final r in _recognizers) {
      r.dispose();
    }
    _recognizers.clear();
    final scheme = Theme.of(context).colorScheme;
    final base = Theme.of(context).textTheme.bodyLarge!.copyWith(fontFamily: readingFont, fontSize: 18.5, height: 1.62);
    final text = widget.block.text;
    final spans = <TextSpan>[];
    var at = 0;
    for (final u in widget.utterances) {
      if (u.start > at) spans.add(TextSpan(text: text.substring(at, u.start)));
      final active = widget.current?.index == u.index;
      final r = TapGestureRecognizer()..onTap = () => widget.onTap(u.index);
      _recognizers.add(r);
      final (bg, fg) = readingHighlight(context);
      spans.add(TextSpan(
        text: text.substring(u.start, u.end),
        recognizer: r,
        style: active ? TextStyle(backgroundColor: bg, color: fg) : null,
      ));
      at = u.end;
    }
    if (at < text.length) spans.add(TextSpan(text: text.substring(at)));
    final color = speakerColor(context, widget.speakerIndex);
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (widget.showSpeaker)
            Padding(
              padding: const EdgeInsets.only(bottom: 6, top: 4),
              child: Row(
                children: [
                  VoiceAvatar(
                    name: widget.speakerVoice ?? widget.block.speaker!,
                    characterId: characterIdOf(widget.speakerRef),
                    size: 24,
                  ),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text.rich(
                      TextSpan(children: [
                        TextSpan(text: widget.block.speaker, style: TextStyle(color: color, fontWeight: FontWeight.w700, letterSpacing: 0.8)),
                        if (widget.speakerVoice != null)
                          TextSpan(text: '  ${widget.speakerVoice}', style: TextStyle(color: scheme.onSurfaceVariant, fontWeight: FontWeight.w500)),
                      ]),
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                  ),
                ],
              ),
            ),
          Container(
            decoration: widget.block.speaker == null
                ? null
                : BoxDecoration(border: Border(left: BorderSide(color: color.withValues(alpha: 0.35), width: 2))),
            padding: widget.block.speaker == null ? EdgeInsets.zero : const EdgeInsets.only(left: 14),
            margin: widget.block.speaker == null ? EdgeInsets.zero : const EdgeInsets.only(left: 11),
            child: Text.rich(TextSpan(children: spans), style: base),
          ),
        ],
      ),
    );
  }
}

class _Cue extends StatelessWidget {
  const _Cue({required this.icon, required this.label});
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        children: [
          Expanded(child: Divider(color: scheme.outlineVariant, height: 1)),
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 10),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(borderRadius: BorderRadius.circular(20), border: Border.all(color: scheme.outlineVariant)),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 14, color: scheme.onSurfaceVariant),
                const SizedBox(width: 6),
                Text(label, style: Theme.of(context).textTheme.labelSmall?.copyWith(color: scheme.onSurfaceVariant)),
              ],
            ),
          ),
          Expanded(child: Divider(color: scheme.outlineVariant, height: 1)),
        ],
      ),
    );
  }
}

class _PlayerBar extends StatelessWidget {
  const _PlayerBar({
    required this.session,
    required this.mine,
    required this.player,
    required this.current,
    required this.onToggle,
    required this.onSeek,
    required this.onSpeed,
    required this.speed,
  });

  final Session session;
  final bool mine;
  final Player player;
  final int current;
  final VoidCallback onToggle;
  final void Function(int) onSeek;
  final void Function(double) onSpeed;
  final double speed;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final total = session.plan.utterances.length;
    final playing = mine && player.isPlaying;
    final buffering = mine && player.state == PlayState.buffering;
    final cur = total == 0 ? 0 : current.clamp(0, total - 1);
    final remaining = total == 0 ? Duration.zero : session.plan.estimate(session.cast[null]?.speed ?? 1.0) * ((total - cur) / total);
    final text = Theme.of(context).textTheme;
    final p = paletteFor(session.doc.title);
    return Container(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLow,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(26)),
        border: Border(top: BorderSide(color: scheme.outlineVariant)),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.08), blurRadius: 20, offset: const Offset(0, -4))],
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 6, 12, 8),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (mine && player.error != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 6, 8, 0),
                  child: Text(player.error!, style: TextStyle(color: scheme.error), maxLines: 3),
                ),
              if (total > 1) _SeekSlider(current: cur, total: total, onSeek: onSeek),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: Row(
                  children: [
                    if (playing) ...[Equalizer(playing: true, size: 12, color: scheme.secondary), const SizedBox(width: 8)],
                    Expanded(
                      child: Text(
                        buffering
                            ? 'Preparing the voice…'
                            : total == 0
                                ? 'Nothing to read'
                                : '${cur + 1} / $total · ${formatDuration(remaining)} left',
                        style: text.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _SpeedButton(speed: speed, onTap: () => _pickSpeed(context)),
                  IconButton(
                    tooltip: 'Previous sentence',
                    iconSize: 30,
                    onPressed: total == 0 ? null : () => onSeek(cur - 1),
                    icon: const Icon(Icons.skip_previous_rounded),
                  ),
                  Material(
                    shape: const CircleBorder(),
                    clipBehavior: Clip.antiAlias,
                    child: Ink(
                      width: 66,
                      height: 66,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [p.b, p.a]),
                      ),
                      child: InkWell(
                        onTap: total == 0 ? null : onToggle,
                        child: Center(
                          child: buffering
                              ? const SizedBox(
                                  width: 24,
                                  height: 24,
                                  child: CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white),
                                )
                              : Icon(
                                  playing ? Icons.pause_rounded : Icons.play_arrow_rounded,
                                  size: 38,
                                  color: Colors.white,
                                  semanticLabel: playing ? 'Pause' : 'Play',
                                ),
                        ),
                      ),
                    ),
                  ),
                  IconButton(
                    tooltip: 'Next sentence',
                    iconSize: 30,
                    onPressed: total == 0 ? null : () => onSeek(cur + 1),
                    icon: const Icon(Icons.skip_next_rounded),
                  ),
                  const SizedBox(width: 56),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _pickSpeed(BuildContext context) async {
    const options = [0.75, 0.85, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5];
    final v = await showModalBottomSheet<double>(
      context: context,
      showDragHandle: true,
      builder: (sheet) => SafeArea(
        child: Wrap(
          children: [
            for (final o in options)
              ListTile(
                title: Text('${o}x'),
                trailing: (o - speed).abs() < 0.001 ? const Icon(Icons.check) : null,
                onTap: () => Navigator.pop(sheet, o),
              ),
          ],
        ),
      ),
    );
    if (v != null) onSpeed(v);
  }
}

class _SpeedButton extends StatelessWidget {
  const _SpeedButton({required this.speed, required this.onTap});
  final double speed;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return SizedBox(
      width: 56,
      child: TextButton(
        style: TextButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 8),
          backgroundColor: scheme.surfaceContainerHigh,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        onPressed: onTap,
        child: Text('${speed.toStringAsFixed(speed == speed.roundToDouble() ? 1 : 2)}x'),
      ),
    );
  }
}

/// Drag to any sentence; it starts reading there when you let go.
class _SeekSlider extends StatefulWidget {
  const _SeekSlider({required this.current, required this.total, required this.onSeek});
  final int current;
  final int total;
  final void Function(int) onSeek;

  @override
  State<_SeekSlider> createState() => _SeekSliderState();
}

class _SeekSliderState extends State<_SeekSlider> {
  double? _drag;

  @override
  Widget build(BuildContext context) {
    final value = (_drag ?? widget.current.toDouble()).clamp(0, widget.total - 1).toDouble();
    return Slider(
      value: value,
      max: (widget.total - 1).toDouble(),
      label: 'Sentence ${value.round() + 1}',
      divisions: widget.total - 1,
      onChanged: (v) => setState(() => _drag = v),
      onChangeEnd: (v) {
        setState(() => _drag = null);
        widget.onSeek(v.round());
      },
    );
  }
}
