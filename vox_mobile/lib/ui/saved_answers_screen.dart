import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vox_amelior_mobile/assistant/assistant_requests.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Every question asked of Gemma (typed or out loud) with its answer, newest
/// first. Search, filter by how it was asked, expand, select and copy.
class SavedAnswersScreen extends StatefulWidget {
  const SavedAnswersScreen({super.key, required this.requests});

  final AssistantRequestRepository requests;

  @override
  State<SavedAnswersScreen> createState() => _SavedAnswersScreenState();
}

class _SavedAnswersScreenState extends State<SavedAnswersScreen> {
  final _search = TextEditingController();
  RequestSource? _source;
  late final List<AssistantRequest> _all = widget.requests.recent(limit: 500);
  final Set<int> _open = {};

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  List<AssistantRequest> get _shown {
    final words = _search.text.trim().toLowerCase().split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
    return [
      for (final r in _all)
        if ((_source == null || r.source == _source) &&
            words.every((w) => r.text.toLowerCase().contains(w) || (r.answer ?? '').toLowerCase().contains(w)))
          r,
    ];
  }

  Future<void> _copy(AssistantRequest r) async {
    await Clipboard.setData(ClipboardData(text: 'Q: ${r.text}\nA: ${r.answer ?? ''}'));
    if (mounted) showMessage(context, 'Copied');
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final shown = _shown;
    return Scaffold(
      appBar: AppBar(title: const Text('Saved answers')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: TextField(
              controller: _search,
              textInputAction: TextInputAction.search,
              decoration: const InputDecoration(hintText: 'Search questions and answers', prefixIcon: Icon(Icons.search_rounded)),
              onChanged: (_) => setState(() {}),
            ),
          ),
          SizedBox(
            height: 44,
            child: ListView(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              children: [
                for (final (label, source) in const [('All', null), ('Typed', RequestSource.app), ('Out loud', RequestSource.voice)])
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: ChoiceChip(label: Text(label), selected: _source == source, onSelected: (_) => setState(() => _source = source)),
                  ),
              ],
            ),
          ),
          Expanded(
            child: _all.isEmpty
                ? const EmptyState(
                    icon: Icons.bookmark_border_rounded,
                    title: 'No answers yet',
                    message: 'Questions you ask Gemma — typed here or said out loud — are kept here with their answers.',
                  )
                : shown.isEmpty
                    ? const EmptyState(icon: Icons.search_off_rounded, title: 'No matches')
                    : ListView.separated(
                        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                        itemCount: shown.length,
                        separatorBuilder: (_, _) => const SizedBox(height: 10),
                        itemBuilder: (context, i) {
                          final r = shown[i];
                          final open = _open.contains(r.id);
                          final failed = r.status == RequestStatus.failed;
                          final answer = r.answer ?? (r.status == RequestStatus.pending ? 'Waiting…' : '');
                          return VoxCard(
                            onTap: () => setState(() => open ? _open.remove(r.id) : _open.add(r.id)),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(r.source == RequestSource.voice ? Icons.mic_rounded : Icons.keyboard_rounded,
                                        size: 16, color: t.colorScheme.onSurfaceVariant),
                                    const SizedBox(width: 6),
                                    Expanded(
                                      child: Text(formatWhen(r.createdAt),
                                          style: t.textTheme.labelMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                                    ),
                                    if (r.sourceSegmentIds.isNotEmpty)
                                      Text('${r.sourceSegmentIds.length} lines',
                                          style: t.textTheme.labelSmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                                  ],
                                ),
                                const SizedBox(height: 6),
                                Text(r.text, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                                const SizedBox(height: 6),
                                if (open)
                                  SelectableText(answer, style: t.textTheme.bodyMedium?.copyWith(color: failed ? t.colorScheme.error : null))
                                else
                                  Text(answer,
                                      maxLines: 3,
                                      overflow: TextOverflow.ellipsis,
                                      style: t.textTheme.bodyMedium?.copyWith(color: failed ? t.colorScheme.error : null)),
                                if (open && r.answer != null)
                                  Align(
                                    alignment: Alignment.centerRight,
                                    child: TextButton.icon(
                                      onPressed: () => _copy(r),
                                      icon: const Icon(Icons.copy_rounded, size: 18),
                                      label: const Text('Copy'),
                                    ),
                                  ),
                              ],
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }
}
