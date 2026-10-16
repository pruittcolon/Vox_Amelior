import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

import '../core/app_state.dart';
import '../core/audio/player.dart';
import '../core/library/library.dart';
import '../core/render/music.dart';
import '../core/render/plan.dart';
import '../core/script/script.dart';
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
        title: Text(widget.doc.title, maxLines: 1, overflow: TextOverflow.ellipsis),
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
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
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
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final speed = session.cast[null]?.speed ?? 1.0;
    final est = session.plan.estimate(speed);
    final speakers = session.script.speakers;
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Card(
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onCast,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.theater_comedy_outlined, size: 18, color: scheme.primary),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text.rich(
                        TextSpan(children: [
                          TextSpan(text: '${session.style.name} · ~${formatDuration(est)}'),
                          if (session.music) ...[
                            const TextSpan(text: '  '),
                            WidgetSpan(
                              alignment: PlaceholderAlignment.middle,
                              child: Icon(Icons.music_note, size: 16, color: scheme.primary),
                            ),
                            TextSpan(text: themeById(session.theme?.id)?.name ?? 'Music'),
                          ],
                        ]),
                        style: text.labelLarge?.copyWith(color: scheme.primary),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Icon(Icons.edit_outlined, size: 18, color: scheme.primary),
                  ],
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 6,
                  children: [
                    if (speakers.isEmpty || session.script.blocks.any((b) => b.kind == BlockKind.line && b.speaker == null))
                      _CastChip(label: 'Narrator', voice: session.cast[null]!.voice.name, color: speakerColor(context, -1)),
                    for (var i = 0; i < speakers.length; i++)
                      _CastChip(
                        label: speakers[i],
                        voice: session.cast[speakers[i]]?.voice.name ?? '',
                        color: speakerColor(context, i),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CastChip extends StatelessWidget {
  const _CastChip({required this.label, required this.voice, required this.color});
  final String label;
  final String voice;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text.rich(
        TextSpan(children: [
          TextSpan(text: label, style: TextStyle(fontWeight: FontWeight.w700, color: color)),
          TextSpan(text: '  $voice'),
        ]),
        style: Theme.of(context).textTheme.bodySmall,
      ),
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
    final style = (block.level <= 1 ? Theme.of(context).textTheme.headlineSmall : Theme.of(context).textTheme.titleLarge)
        ?.copyWith(fontWeight: FontWeight.w700);
    return Padding(
      padding: const EdgeInsets.only(top: 18, bottom: 10),
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
    required this.onTap,
  });

  final Block block;
  final List<Utterance> utterances;
  final Utterance? current;
  final bool playing;
  final bool showSpeaker;
  final int speakerIndex;
  final String? speakerVoice;
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
    final base = Theme.of(context).textTheme.bodyLarge!.copyWith(fontSize: 18, height: 1.55);
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
      padding: const EdgeInsets.only(bottom: 14),
      child: Container(
        decoration: widget.block.speaker == null
            ? null
            : BoxDecoration(border: Border(left: BorderSide(color: color.withValues(alpha: 0.6), width: 3))),
        padding: widget.block.speaker == null ? EdgeInsets.zero : const EdgeInsets.only(left: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (widget.showSpeaker)
              Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Text.rich(
                  TextSpan(children: [
                    TextSpan(text: widget.block.speaker, style: TextStyle(color: color, fontWeight: FontWeight.w800, letterSpacing: 0.6)),
                    if (widget.speakerVoice != null)
                      TextSpan(text: '  ${widget.speakerVoice}', style: TextStyle(color: scheme.onSurfaceVariant)),
                  ]),
                  style: Theme.of(context).textTheme.labelMedium,
                ),
              ),
            Text.rich(TextSpan(children: spans), style: base),
          ],
        ),
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
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        children: [
          Icon(icon, size: 16, color: scheme.outline),
          const SizedBox(width: 6),
          Text(label, style: Theme.of(context).textTheme.labelMedium?.copyWith(color: scheme.outline)),
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
    return Material(
      color: scheme.surfaceContainer,
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8, 4, 8, 6),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (mine && player.error != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 6, 8, 0),
                  child: Text(player.error!, style: TextStyle(color: scheme.error), maxLines: 3),
                ),
              if (total > 1) _SeekSlider(current: cur, total: total, onSeek: onSeek),
              Row(
                children: [
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      buffering
                          ? 'Preparing the voice…'
                          : total == 0
                              ? 'Nothing to read'
                              : '${cur + 1} / $total · ${formatDuration(remaining)} left',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                  IconButton(tooltip: 'Previous sentence', onPressed: total == 0 ? null : () => onSeek(cur - 1), icon: const Icon(Icons.skip_previous_rounded)),
                  SizedBox(
                    width: 56,
                    height: 56,
                    child: FilledButton(
                      style: FilledButton.styleFrom(shape: const CircleBorder(), padding: EdgeInsets.zero),
                      onPressed: total == 0 ? null : onToggle,
                      child: buffering
                          ? SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(strokeWidth: 2.5, color: scheme.onPrimary),
                            )
                          : Icon(playing ? Icons.pause_rounded : Icons.play_arrow_rounded, size: 32),
                    ),
                  ),
                  IconButton(tooltip: 'Next sentence', onPressed: total == 0 ? null : () => onSeek(cur + 1), icon: const Icon(Icons.skip_next_rounded)),
                  TextButton(
                    onPressed: () => _pickSpeed(context),
                    child: Text('${speed.toStringAsFixed(speed == speed.roundToDouble() ? 1 : 2)}x'),
                  ),
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
