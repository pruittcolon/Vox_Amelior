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
import 'chapters.dart';
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

  /// The whole index; picking a chapter plays from there.
  Future<void> _showChapters() async {
    final s = _session;
    if (s == null) return;
    final target = await showChaptersSheet(context, s, current: _current);
    if (target != null) await _playFrom(target);
  }

  /// Next or previous chapter. Back goes to the start of this chapter first,
  /// unless you're right at its beginning.
  Future<void> _skipChapter(int direction) async {
    final s = _session;
    if (s == null || !s.plan.hasChapters) return;
    final chapters = s.plan.chapters;
    final here = s.plan.chapterOf(_current)!;
    int target;
    if (direction > 0) {
      if (here.index + 1 >= chapters.length) return;
      target = chapters[here.index + 1].first;
    } else {
      target = _current > here.first + 1 || here.index == 0 ? here.first : chapters[here.index - 1].first;
    }
    await _playFrom(target);
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
                        onShowChapters: _showChapters,
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
              onSkipChapter: _skipChapter,
              onShowChapters: _showChapters,
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
            Text(noVoices ? 'Go to the Voices tab and download a voice pack: Natural, Studio or Lifelike.' : '$error', textAlign: TextAlign.center),
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
    required this.onShowChapters,
  });

  final Session session;
  final ScrollController controller;
  final Map<int, GlobalKey> keys;
  final int current;
  final bool playing;
  final void Function(int utterance) onTap;
  final VoidCallback onCast;
  final VoidCallback onShowChapters;

  @override
  Widget build(BuildContext context) {
    final blocks = session.script.blocks;
    final byBlock = session.plan.byBlock;
    final speakers = session.script.speakers;
    final cur = session.plan.utterances.isEmpty ? null : session.plan.utterances[current.clamp(0, session.plan.utterances.length - 1)];
    final chapterOfHeading = {
      if (session.plan.hasChapters)
        for (final c in session.plan.chapters)
          if (c.chapter.heading != null) c.chapter.heading!: c,
    };
    return ListView.builder(
      controller: controller,
      padding: const EdgeInsets.fromLTRB(22, 0, 22, 40),
      itemCount: blocks.length + 1,
      itemBuilder: (context, i) {
        if (i == 0) {
          return _Header(session: session, onCast: onCast, current: current, onJump: onTap, onShowChapters: onShowChapters);
        }
        final bi = i - 1;
        final b = blocks[bi];
        final key = keys.putIfAbsent(bi, GlobalKey.new);
        final prevSpeaker = bi > 0 && blocks[bi - 1].kind == BlockKind.line ? blocks[bi - 1].speaker : '\u0000';
        return KeyedSubtree(
          key: key,
          child: switch (b.kind) {
            BlockKind.heading => _Heading(
                block: b,
                utterances: byBlock[bi] ?? const [],
                current: cur,
                chapter: chapterOfHeading[bi],
                session: session,
                onTap: onTap,
              ),
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
  const _Header({
    required this.session,
    required this.onCast,
    required this.current,
    required this.onJump,
    required this.onShowChapters,
  });
  final Session session;
  final VoidCallback onCast;
  final int current;
  final void Function(int utterance) onJump;
  final VoidCallback onShowChapters;

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
          if (session.script.description != null) ...[
            const SizedBox(height: 14),
            Text(
              session.script.description!,
              style: text.bodyLarge?.copyWith(fontFamily: readingFont, height: 1.5, color: scheme.onSurfaceVariant),
            ),
          ],
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _Pill(icon: Icons.schedule_rounded, label: '${session.style.name} · ~${formatDuration(est)}'),
              if (session.music) _Pill(icon: Icons.music_note_rounded, label: themeById(session.theme?.id)?.name ?? 'Music'),
              for (final t in session.script.topics) _Pill(icon: Icons.tag_rounded, label: topicLabel(t)),
            ],
          ),
          if (session.plan.hasChapters) ...[
            const SizedBox(height: 14),
            ChapterCard(session: session, current: current, onJump: onJump, onShowAll: onShowChapters),
          ],
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
  const _Heading({
    required this.block,
    required this.utterances,
    required this.current,
    required this.chapter,
    required this.session,
    required this.onTap,
  });
  final Block block;
  final List<Utterance> utterances;
  final Utterance? current;
  final PlanChapter? chapter;
  final Session session;
  final void Function(int) onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final active = utterances.isNotEmpty && current?.index == utterances.first.index;
    final style = block.level <= 1 ? theme.textTheme.headlineSmall : theme.textTheme.titleLarge;
    final c = chapter;
    final target = c?.first ?? (utterances.isEmpty ? null : utterances.first.index);
    return Padding(
      padding: const EdgeInsets.only(top: 22, bottom: 12),
      child: InkWell(
        onTap: target == null ? null : () => onTap(target),
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          decoration: active ? BoxDecoration(color: readingHighlight(context).$1, borderRadius: BorderRadius.circular(8)) : null,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (c != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(
                    'CHAPTER ${c.index + 1} · ${formatClock(chapterStart(session, c))}',
                    style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.secondary, letterSpacing: 1.2),
                  ),
                ),
              Text(block.text, style: style),
              if (c?.summary != null)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    c!.summary!,
                    style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant, height: 1.45),
                  ),
                ),
            ],
          ),
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

class _PlayerBar extends StatefulWidget {
  const _PlayerBar({
    required this.session,
    required this.mine,
    required this.player,
    required this.current,
    required this.onToggle,
    required this.onSeek,
    required this.onSkipChapter,
    required this.onShowChapters,
    required this.onSpeed,
    required this.speed,
  });

  final Session session;
  final bool mine;
  final Player player;
  final int current;
  final VoidCallback onToggle;
  final void Function(int) onSeek;
  final void Function(int direction) onSkipChapter;
  final VoidCallback onShowChapters;
  final void Function(double) onSpeed;
  final double speed;

  @override
  State<_PlayerBar> createState() => _PlayerBarState();
}

class _PlayerBarState extends State<_PlayerBar> {
  /// The spot under your finger while dragging the seek bar.
  int? _preview;

  @override
  Widget build(BuildContext context) {
    final session = widget.session;
    final player = widget.player;
    final scheme = Theme.of(context).colorScheme;
    final total = session.plan.utterances.length;
    final playing = widget.mine && player.isPlaying;
    final buffering = widget.mine && player.state == PlayState.buffering;
    final cur = total == 0 ? 0 : widget.current.clamp(0, total - 1);
    final shown = _preview ?? cur;
    final times = sessionTimes(session);
    final at = total == 0 ? Duration.zero : Duration(milliseconds: times.starts[shown]);
    final length = Duration(milliseconds: times.total);
    final text = Theme.of(context).textTheme;
    final p = paletteFor(session.doc.title);
    final chapters = session.plan.hasChapters;
    final chapter = chapters ? session.plan.chapterOf(shown) : null;
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
              if (widget.mine && player.error != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 6, 8, 0),
                  child: Text(player.error!, style: TextStyle(color: scheme.error), maxLines: 3),
                ),
              if (chapter != null)
                Row(
                  children: [
                    IconButton(
                      tooltip: 'Previous chapter',
                      visualDensity: VisualDensity.compact,
                      onPressed: () => widget.onSkipChapter(-1),
                      icon: const Icon(Icons.keyboard_double_arrow_left_rounded),
                    ),
                    Expanded(
                      child: InkWell(
                        borderRadius: BorderRadius.circular(10),
                        onTap: widget.onShowChapters,
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 4),
                          child: Column(
                            children: [
                              Text(
                                'CHAPTER ${chapter.index + 1} OF ${session.plan.chapters.length}',
                                style: text.labelSmall?.copyWith(color: scheme.secondary, letterSpacing: 1.2),
                              ),
                              Text(
                                chapter.title,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                textAlign: TextAlign.center,
                                style: text.titleSmall,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    IconButton(
                      tooltip: 'Next chapter',
                      visualDensity: VisualDensity.compact,
                      onPressed: chapter.index + 1 < session.plan.chapters.length ? () => widget.onSkipChapter(1) : null,
                      icon: const Icon(Icons.keyboard_double_arrow_right_rounded),
                    ),
                  ],
                ),
              if (widget.mine && player.preparing != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(12, 4, 12, 6),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(value: player.preparing, minHeight: 6),
                  ),
                ),
              if (total > 1)
                ChapterSeekBar(
                  session: session,
                  current: cur,
                  onSeek: widget.onSeek,
                  onPreview: (u) => setState(() => _preview = u),
                ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: Row(
                  children: [
                    if (playing) ...[Equalizer(playing: true, size: 12, color: scheme.secondary), const SizedBox(width: 8)],
                    Expanded(
                      child: Text(
                        buffering && _preview == null
                            ? (widget.mine && player.preparing != null
                                ? 'Getting ahead${player.preparingLeft == null ? '' : ': about ${formatClock(player.preparingLeft!)} left'}, '
                                    'then it plays to the end'
                                : 'Preparing the voice…')
                            : total == 0
                                ? 'Nothing to read'
                                : '${formatClock(at)} / ${formatClock(length)}',
                        style: text.bodySmall,
                      ),
                    ),
                    if (total > 0)
                      Text(
                        '${formatClock(length - at)} left',
                        style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _SpeedButton(speed: widget.speed, onTap: () => _pickSpeed(context)),
                  IconButton(
                    tooltip: 'Previous sentence',
                    iconSize: 30,
                    onPressed: total == 0 ? null : () => widget.onSeek(cur - 1),
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
                        onTap: total == 0 ? null : widget.onToggle,
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
                    onPressed: total == 0 ? null : () => widget.onSeek(cur + 1),
                    icon: const Icon(Icons.skip_next_rounded),
                  ),
                  SizedBox(
                    width: 56,
                    child: chapters
                        ? IconButton(
                            tooltip: 'Chapters',
                            onPressed: widget.onShowChapters,
                            icon: const Icon(Icons.format_list_numbered_rounded),
                          )
                        : null,
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
    final speed = widget.speed;
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
    if (v != null) widget.onSpeed(v);
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
