import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/data/models.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

/// Chat-like list of utterances, newest first, with day headers.
class TranscriptList extends StatelessWidget {
  const TranscriptList({super.key, required this.segments, this.onTap, this.emptyText = 'Nothing heard yet.'});

  final List<SegmentView> segments;
  final void Function(SegmentView segment)? onTap;
  final String emptyText;

  @override
  Widget build(BuildContext context) {
    if (segments.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Text(emptyText, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyLarge),
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.only(bottom: 96),
      itemCount: segments.length,
      itemBuilder: (context, i) {
        final s = segments[i];
        final header = i == 0 || formatDay(segments[i - 1].startedAt) != formatDay(s.startedAt);
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (header)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                child: Text(formatDay(s.startedAt), style: Theme.of(context).textTheme.labelLarge),
              ),
            SegmentTile(segment: s, onTap: onTap == null ? null : () => onTap!(s)),
          ],
        );
      },
    );
  }
}

class SegmentTile extends StatelessWidget {
  const SegmentTile({super.key, required this.segment, this.onTap});

  final SegmentView segment;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final color = speakerColor(segment.speakerLabel, known: segment.isKnownSpeaker);
    return ListTile(
      onTap: onTap,
      leading: CircleAvatar(
        backgroundColor: color.withValues(alpha: 0.15),
        foregroundColor: color,
        child: Text(segment.speakerLabel.characters.first.toUpperCase()),
      ),
      title: Row(
        children: [
          Flexible(
            child: Text(
              segment.speakerLabel,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: color, fontWeight: FontWeight.w600),
            ),
          ),
          const SizedBox(width: 8),
          Text(formatTime(segment.startedAt), style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
      subtitle: Text(segment.text, style: Theme.of(context).textTheme.bodyLarge),
    );
  }
}
