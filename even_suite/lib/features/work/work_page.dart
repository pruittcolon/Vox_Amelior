import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/bridge/bridge_client.dart';
import '../../core/bridge/work_controller.dart';
import '../../widgets/common.dart';

/// Claude Code jobs that finished in Termux. A Claude Code hook reports each
/// one; the app says so on the phone and the glasses - briefly, and only when
/// Claude is done.
class WorkPage extends StatefulWidget {
  const WorkPage({super.key, required this.active});

  final bool active;

  @override
  State<WorkPage> createState() => _WorkPageState();
}

class _WorkPageState extends State<WorkPage> {
  String? _open;
  bool? _hook;
  bool _installing = false;

  @override
  void didUpdateWidget(WorkPage old) {
    super.didUpdateWidget(old);
    if (widget.active && !old.active) {
      SuiteScope.of(context).work.markRead();
      _checkHook();
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_hook == null) _checkHook();
  }

  Future<void> _checkHook() async {
    final suite = SuiteScope.of(context);
    try {
      final info = await suite.client.ping();
      if (mounted) setState(() => _hook = info.hook);
    } catch (_) {}
  }

  Future<void> _install() async {
    final suite = SuiteScope.of(context);
    setState(() => _installing = true);
    try {
      final ok = await suite.client.installHook();
      if (!mounted) return;
      setState(() => _hook = ok);
      showToast(context, ok ? 'Done: Claude Code now reports finished jobs here' : 'Could not install the hook');
    } catch (e) {
      if (mounted) showToast(context, '$e');
    } finally {
      if (mounted) setState(() => _installing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final work = suite.work;
    final s = suite.settings;
    return ListenableBuilder(
      listenable: Listenable.merge([work, s]),
      builder: (context, _) {
        final items = work.items.reversed.toList();
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            Row(children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Work', style: Theme.of(context).textTheme.headlineSmall),
                  const Text('When Claude Code finishes a job', style: TextStyle(color: AppColors.muted)),
                ]),
              ),
              Switch(value: s.workAlerts, onChanged: (v) => s.set('workAlerts', v)),
            ]),
            const SizedBox(height: 12),
            SectionCard(
              title: 'Tell me when Claude is done',
              icon: Icons.notifications_active_rounded,
              iconColor: AppColors.claude,
              trailing: StatusPill(
                work.online ? (_hook == false ? 'Hook missing' : 'Watching') : 'Offline',
                color: work.online ? (_hook == false ? AppColors.warn : AppColors.ok) : AppColors.error,
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                const Text(
                    'Any Claude Code session in Termux (the one you type into too) shows here while it works '
                    '- and on the glasses when you look up. When it finishes you get a one-line note on the '
                    'glasses and a phone notification - only at the end.',
                    style: TextStyle(color: AppColors.muted, fontSize: 13.5)),
                if (_hook == false) ...[
                  const SizedBox(height: 10),
                  FilledButton.icon(
                    onPressed: _installing ? null : _install,
                    icon: const Icon(Icons.download_done_rounded),
                    label: Text(_installing ? 'Installing…' : 'Set up Claude Code (one tap)'),
                  ),
                  const SizedBox(height: 4),
                  const Text('Or in Termux: even setup',
                      textAlign: TextAlign.center, style: TextStyle(color: AppColors.muted, fontSize: 12)),
                ],
                const SizedBox(height: 12),
                const Text('Only jobs that took at least', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Wrap(spacing: 8, runSpacing: 8, children: [
                  for (final (sec, label) in const [(0, 'Any'), (30, '30 s'), (60, '1 min'), (300, '5 min')])
                    ChoiceChip(
                      label: Text(label),
                      selected: s.workMinSec == sec,
                      onSelected: (_) => s.set('workMinSec', sec),
                    ),
                ]),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Phone notification', style: TextStyle(fontWeight: FontWeight.w600)),
                  value: s.workPhone,
                  onChanged: (v) async {
                    if (v) await suite.system.permissions(['notifications']);
                    s.set('workPhone', v);
                  },
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Note on the glasses', style: TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: const Text('Pops up briefly; also kept for when you look up.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                  value: s.workGlasses,
                  onChanged: (v) => s.set('workGlasses', v),
                ),
              ]),
            ),
            if (work.active.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text('Working now', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 6),
              for (final job in work.active) _LiveJob(job: job),
              const Padding(
                padding: EdgeInsets.only(top: 2),
                child: Text('Look up on the glasses to see it there too; it keeps updating while you look.',
                    style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
              ),
            ],
            const SizedBox(height: 16),
            Row(children: [
              Expanded(child: Text('Finished', style: Theme.of(context).textTheme.titleMedium)),
              if (items.isNotEmpty)
                TextButton(onPressed: () => work.clear(), child: const Text('Clear')),
            ]),
            const SizedBox(height: 6),
            if (items.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 24),
                child: Text('Nothing yet. Ask Claude Code to do something in Termux; it shows up here when it is done.',
                    textAlign: TextAlign.center, style: TextStyle(color: AppColors.muted)),
              )
            else
              for (final item in items)
                _JobCard(
                  item: item,
                  open: _open == item.id,
                  onTap: () => setState(() => _open = _open == item.id ? null : item.id),
                  onShow: () {
                    suite.display.showText('${item.summary}\n\n${item.text}', title: 'Claude');
                    showToast(context, suite.ble.isConnected ? 'Showing on your glasses' : 'Glasses are not connected');
                  },
                ),
          ],
        );
      },
    );
  }
}

/// A job in progress: spinner, what Claude is doing, what you asked, time.
class _LiveJob extends StatefulWidget {
  const _LiveJob({required this.job});
  final WorkJob job;
  @override
  State<_LiveJob> createState() => _LiveJobState();
}

class _LiveJobState extends State<_LiveJob> {
  Timer? _tick;

  @override
  void initState() {
    super.initState();
    _tick = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _tick?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final j = widget.job;
    final meta = [
      if (j.project.isNotEmpty) j.project,
      formatDuration(j.elapsed()),
      if (j.steps > 0) '${j.steps} step${j.steps == 1 ? '' : 's'}',
    ].join(' · ');
    return Card(
      color: AppColors.claude.withOpacity(0.08),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: AppColors.claude.withOpacity(0.35)),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 12, 12),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Padding(
            padding: EdgeInsets.only(top: 2),
            child: SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2.4, color: AppColors.claude),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(j.step.isEmpty ? 'Thinking' : j.step,
                  style: const TextStyle(fontWeight: FontWeight.w700, height: 1.3)),
              if (j.prompt.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 3),
                  child: Text('“${j.prompt}”',
                      maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 13.5)),
                ),
              const SizedBox(height: 4),
              Text(meta, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
            ]),
          ),
        ]),
      ),
    );
  }
}

class _JobCard extends StatelessWidget {
  const _JobCard({required this.item, required this.open, required this.onTap, required this.onShow});
  final WorkItem item;
  final bool open;
  final VoidCallback onTap;
  final VoidCallback onShow;

  @override
  Widget build(BuildContext context) {
    final t = item.time;
    final when = '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
    final meta = [
      if (item.project.isNotEmpty) item.project,
      if (item.ms > 0) formatDuration(item.took),
      if (item.tools > 0) '${item.tools} step${item.tools == 1 ? '' : 's'}',
      when,
    ].join(' · ');
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        onLongPress: () {
          Clipboard.setData(ClipboardData(text: item.text));
          showToast(context, 'Copied');
        },
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 12, 12),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Padding(
                padding: EdgeInsets.only(top: 2),
                child: Icon(Icons.check_circle_rounded, color: AppColors.ok, size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(item.summary, style: const TextStyle(fontWeight: FontWeight.w700, height: 1.3)),
                  if (item.prompt.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text('You asked: ${item.prompt}',
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(color: AppColors.muted, fontSize: 13)),
                    ),
                  const SizedBox(height: 4),
                  Text(meta, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
                ]),
              ),
              Icon(open ? Icons.expand_less_rounded : Icons.expand_more_rounded, color: AppColors.muted),
            ]),
            if (open) ...[
              const SizedBox(height: 10),
              SelectableText(item.text, style: const TextStyle(fontSize: 14, height: 1.4)),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: onShow,
                  icon: const Icon(Icons.visibility_rounded),
                  label: const Text('Show on glasses'),
                ),
              ),
            ],
          ]),
        ),
      ),
    );
  }
}
