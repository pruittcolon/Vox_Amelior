import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/context_budget.dart';
import 'package:vox_amelior_mobile/assistant/context_probe.dart';
import 'package:vox_amelior_mobile/assistant/llm_engine.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

String formatTokens(int n) => n.toString().replaceAllMapped(RegExp(r'\B(?=(\d{3})+$)'), (_) => ',');

/// How much Gemma can read at once on this phone: test it, then use half of
/// it for each part of a "go through everything" review.
class CapacityScreen extends StatefulWidget {
  const CapacityScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<CapacityScreen> createState() => _CapacityScreenState();
}

class _CapacityScreenState extends State<CapacityScreen> {
  final Map<int, ProbeStep> _steps = {};
  bool _testing = false;
  String? _error;

  AppServices get s => widget.services;

  Future<void> _test() async {
    setState(() {
      _testing = true;
      _error = null;
      _steps.clear();
    });
    try {
      await s.testContext((step) {
        if (mounted) setState(() => _steps[step.size] = step);
      });
    } on Object catch (e) {
      if (mounted) setState(() => _error = friendlyLlmError(e));
    } finally {
      if (mounted) setState(() => _testing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Gemma capacity')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: s.settings,
        builder: (context, st, _) {
          final t = Theme.of(context);
          final b = st.budget;
          final ready = s.assistantReady;
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              VoxCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.memory_rounded, color: t.colorScheme.primary),
                        const SizedBox(width: 10),
                        Expanded(child: Text('Reads at once', style: t.textTheme.titleMedium)),
                        Text('${formatTokens(st.contextTokens)} tokens',
                            style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800, color: t.colorScheme.primary)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      st.contextTested == 0
                          ? 'Not tested yet. Using ${formatTokens(ContextBudget.defaultContext)}, which works on most phones.'
                          : st.contextTestNote,
                      style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
                    ),
                    const Divider(height: 28),
                    Row(
                      children: [
                        Icon(Icons.splitscreen_rounded, color: t.colorScheme.tertiary),
                        const SizedBox(width: 10),
                        Expanded(child: Text('Recommended per review part', style: t.textTheme.titleSmall)),
                        Text('${formatTokens(b.recommendedChunkTokens)} tokens', style: const TextStyle(fontWeight: FontWeight.w700)),
                      ],
                    ),
                    Padding(
                      padding: const EdgeInsets.only(left: 34, top: 2),
                      child: Text('Half of what Gemma can read, about ${ContextBudget.linesFor(b.recommendedChunkTokens)} lines. '
                          'The other half holds your instructions, notes and Gemma\'s reply.'),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: !ready || _testing ? null : _test,
                icon: _testing
                    ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.speed_rounded),
                label: Text(_testing ? 'Testing…' : 'Test this phone'),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(4, 8, 4, 0),
                child: Text(
                  ready
                      ? 'Tries ${ContextBudget.testSizes.map(formatTokens).join(', ')} tokens and keeps the largest that works. '
                          'Takes a few minutes; transcription waits meanwhile (nothing is lost). If the app closes during the test, '
                          'that size is marked as too big.'
                      : 'Download Gemma first (More → Models).',
                  style: t.textTheme.bodySmall,
                ),
              ),
              if (_steps.isNotEmpty || _error != null) ...[
                const SectionHeader('Test', padding: EdgeInsets.fromLTRB(4, 20, 4, 8)),
                VoxCard(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Column(
                    children: [
                      for (final size in ContextBudget.testSizes)
                        if (_steps[size] != null) _stepTile(context, _steps[size]!),
                      if (_error != null)
                        ListTile(
                          leading: Icon(Icons.error_outline_rounded, color: t.colorScheme.error),
                          title: Text(_error!),
                        ),
                    ],
                  ),
                ),
              ],
              const SectionHeader('Adjust', padding: EdgeInsets.fromLTRB(4, 24, 4, 8)),
              VoxCard(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Expanded(child: Text('Context size')),
                        DropdownButton<int>(
                          value: ContextBudget.testSizes.contains(st.contextTokens) ? st.contextTokens : null,
                          hint: Text(formatTokens(st.contextTokens)),
                          underline: const SizedBox.shrink(),
                          items: [
                            for (final n in ContextBudget.testSizes)
                              DropdownMenuItem(
                                value: n,
                                child: Text(
                                  '${formatTokens(n)}${st.contextTested > 0 && n > st.contextTested ? '  (failed test)' : ''}',
                                ),
                              ),
                          ],
                          onChanged: (v) {
                            if (v != null) s.updateSettings(st.copyWith(contextTokens: v, reviewChunkTokens: 0));
                          },
                        ),
                      ],
                    ),
                    if (st.contextTested > 0 && st.contextTokens > st.contextTested)
                      Text('Larger than what passed the test — Gemma may fail or crash.',
                          style: TextStyle(color: t.colorScheme.error)),
                    const SizedBox(height: 12),
                    const Text('Review part size'),
                    Text('${formatTokens(b.chunkTokens)} tokens · about ${ContextBudget.linesFor(b.chunkTokens)} lines',
                        style: TextStyle(fontWeight: FontWeight.w700, color: t.colorScheme.primary)),
                    Slider(
                      value: b.chunkTokens.toDouble().clamp(512, b.maxChunkTokens.toDouble()),
                      min: 512,
                      max: b.maxChunkTokens.toDouble().clamp(513, double.infinity),
                      divisions: ((b.maxChunkTokens - 512) ~/ 128).clamp(1, 400),
                      label: formatTokens(b.chunkTokens),
                      onChanged: (v) => s.updateSettings(st.copyWith(reviewChunkTokens: (v ~/ 64) * 64)),
                    ),
                    Align(
                      alignment: Alignment.centerRight,
                      child: TextButton(
                        onPressed: st.reviewChunkTokens == 0 ? null : () => s.updateSettings(st.copyWith(reviewChunkTokens: 0)),
                        child: const Text('Use recommended (half)'),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _stepTile(BuildContext context, ProbeStep step) {
    final t = Theme.of(context);
    final (Widget icon, String label) = switch (step.status) {
      ProbeStatus.testing => (
          const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5)),
          'Testing…',
        ),
      ProbeStatus.passed => (Icon(Icons.check_circle_rounded, color: t.colorScheme.primary), 'Works'),
      ProbeStatus.failed => (Icon(Icons.cancel_rounded, color: t.colorScheme.error), 'Too big: ${step.detail}'),
    };
    return ListTile(
      leading: icon,
      title: Text('${formatTokens(step.size)} tokens'),
      subtitle: Text(label),
      trailing: step.status == ProbeStatus.passed ? Text('~${ContextBudget.linesFor(step.size ~/ 2)} lines/part') : null,
    );
  }
}

/// Short summary for settings lists.
String capacitySummary(AppSettings st) => st.contextTested == 0
    ? '${formatTokens(st.contextTokens)} tokens · not tested'
    : '${formatTokens(st.contextTokens)} tokens · tested on this phone';
