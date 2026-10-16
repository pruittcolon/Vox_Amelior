import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;

import '../core/app_state.dart';
import '../core/audio/exporter.dart';
import 'scope.dart';

Future<void> showExportSheet(BuildContext context, Session session) {
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (_) => _ExportSheet(session: session),
  );
}

class _ExportSheet extends StatefulWidget {
  const _ExportSheet({required this.session});
  final Session session;

  @override
  State<_ExportSheet> createState() => _ExportSheetState();
}

class _ExportSheetState extends State<_ExportSheet> {
  late String _format = context.app.settings.exportFormat;

  @override
  Widget build(BuildContext context) {
    final exporter = context.scope.exporter;
    final text = Theme.of(context).textTheme;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: ListenableBuilder(
          listenable: exporter,
          builder: (context, _) {
            final mine = exporter.title == widget.session.doc.title;
            if (exporter.busy && mine) return _progress(context, exporter);
            if (exporter.state == ExportState.done && mine && exporter.output != null) return _done(context, exporter);
            final est = widget.session.plan.estimate(widget.session.cast[null]?.speed ?? 1.0);
            return Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Export audio', style: text.titleLarge),
                const SizedBox(height: 6),
                Text(
                  'Renders "${widget.session.doc.title}" (about ${formatDuration(est)}) with the voices, '
                  'pauses and music you hear, into a file you can upload as a podcast.',
                ),
                const SizedBox(height: 12),
                RadioGroup<String>(
                  groupValue: _format,
                  onChanged: (v) => setState(() => _format = v!),
                  child: const Column(
                    children: [
                      RadioListTile<String>(
                        value: 'm4a',
                        contentPadding: EdgeInsets.zero,
                        title: Text('M4A (AAC) — recommended'),
                        subtitle: Text('Small, good quality; accepted by Spotify, Apple Podcasts and YouTube.'),
                      ),
                      RadioListTile<String>(
                        value: 'wav',
                        contentPadding: EdgeInsets.zero,
                        title: Text('WAV'),
                        subtitle: Text('Uncompressed, about 10x bigger. For editing in another app.'),
                      ),
                    ],
                  ),
                ),
                if (exporter.state == ExportState.failed && mine)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text('Last export failed: ${exporter.error}', style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  ),
                if (exporter.busy && !mine)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text('Another export ("${exporter.title}") is running. Wait for it to finish.', style: text.bodySmall),
                  ),
                const SizedBox(height: 8),
                Text(
                  'Tip: exporting keeps the screen on; it takes roughly as long as the speech generator needs '
                  '(often a fraction of the listening time). You can leave the app; a notification shows progress.',
                  style: text.bodySmall,
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: exporter.busy
                        ? null
                        : () async {
                            final app = context.app;
                            final player = context.scope.player;
                            await app.settings.update((s) => s.exportFormat = _format);
                            exporter.reset();
                            if (player.isPlaying) await player.pause();
                            await exporter.export(widget.session, format: _format);
                          },
                    icon: const Icon(Icons.graphic_eq),
                    label: const Text('Start export'),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _progress(BuildContext context, Exporter e) {
    final eta = e.eta;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(e.state == ExportState.encoding ? 'Compressing…' : 'Rendering speech…', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 16),
        LinearProgressIndicator(value: e.progress),
        const SizedBox(height: 8),
        Text('${(e.progress * 100).floor()}%${eta == null ? '' : ' · about ${formatDuration(eta)} left'}'),
        const SizedBox(height: 16),
        Align(
          alignment: Alignment.centerRight,
          child: TextButton(onPressed: e.cancel, child: const Text('Cancel')),
        ),
      ],
    );
  }

  Widget _done(BuildContext context, Exporter e) {
    final path = e.output!;
    final name = p.basename(path);
    final mime = path.endsWith('.wav') ? 'audio/wav' : 'audio/mp4';
    final native = context.scope.native;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.check_circle, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 8),
            Text('Exported', style: Theme.of(context).textTheme.titleLarge),
          ],
        ),
        const SizedBox(height: 8),
        Text('$name · ${formatDuration(e.length)}'),
        const SizedBox(height: 16),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            FilledButton.icon(
              onPressed: () async {
                try {
                  final where = await native.saveToDownloads(path, name, mime);
                  if (context.mounted) context.toast('Saved to $where');
                } catch (err) {
                  if (context.mounted) context.toast('$err');
                }
              },
              icon: const Icon(Icons.download),
              label: const Text('Save to Downloads'),
            ),
            OutlinedButton.icon(
              onPressed: () => native.share(path, mime, subject: e.title),
              icon: const Icon(Icons.share),
              label: const Text('Share'),
            ),
            TextButton(onPressed: e.reset, child: const Text('Export again')),
          ],
        ),
      ],
    );
  }
}
