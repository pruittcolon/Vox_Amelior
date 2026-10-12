import 'dart:async';

import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/log.dart';
import '../../widgets/common.dart';

/// Live view of the timing log, and where it is saved.
class DiagnosticsPage extends StatefulWidget {
  const DiagnosticsPage({super.key});

  @override
  State<DiagnosticsPage> createState() => _DiagnosticsPageState();
}

class _DiagnosticsPageState extends State<DiagnosticsPage> {
  Timer? _tick;
  String _filter = 'all';
  bool _uploading = false;

  static const filters = ['all', 'ble', 'display', 'bmp', 'screen', 'voice', 'chat', 'glasses', 'native'];

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

  /// Median and slowest of `ms` for events of one kind.
  static String _ms(List<LogEvent> events, String cat, String name) {
    final ms = [
      for (final e in events)
        if (e.cat == cat && e.name == name && e.data['ms'] is num) (e.data['ms'] as num).toInt(),
    ]..sort();
    if (ms.isEmpty) return '–';
    return '${ms[ms.length ~/ 2]} ms (max ${ms.last})';
  }

  @override
  Widget build(BuildContext context) {
    final log = EventLog.instance;
    final suite = SuiteScope.of(context);
    final events = log.recent;
    final shown = [
      for (final e in events.reversed)
        if (_filter == 'all' || e.cat == _filter) e,
    ];
    return Scaffold(
      appBar: AppBar(title: const Text('Diagnostics')),
      body: ListView.builder(
        padding: const EdgeInsets.only(bottom: 16),
        itemCount: 2 + (shown.isEmpty ? 1 : shown.length),
        itemBuilder: (context, index) {
          if (index == 0) {
            return Column(children: [
              const _LabCard(),
              Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
          child: SectionCard(
            title: 'Timing log',
            icon: Icons.insights_rounded,
            iconColor: AppColors.glasses,
            trailing: Switch(
              value: suite.settings.diagnostics,
              onChanged: (v) => setState(() => suite.settings.set('diagnostics', v)),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Text(
                'Saved in Termux at ~/.agent-hub/logs/ so Claude can read the raw data. '
                '${log.uploaded} uploaded · ${log.pending} waiting'
                '${log.dropped > 0 ? ' · ${log.dropped} dropped' : ''}'
                '${log.lastError != null ? '\nLast upload failed: ${log.lastError}' : ''}',
                style: const TextStyle(color: AppColors.muted, fontSize: 13),
              ),
              const SizedBox(height: 10),
              Text('Glasses screen send: ${_ms(events, 'display', 'screen')}\n'
                  'Picture frame: ${_ms(events, 'bmp', 'frame')}\n'
                  'Bluetooth round trip: ${_ms(events, 'ble', 'req')}\n'
                  'Claude reply: ${_ms(events, 'chat', 'reply')}',
                  style: const TextStyle(fontSize: 13, height: 1.5)),
              const SizedBox(height: 10),
              const Text('Picture packet spacing (tuning)', style: TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final ms in const [0, 3, 5, 8, 12])
                  ChoiceChip(
                    label: Text('$ms ms'),
                    selected: suite.settings.bmpGapMs == ms,
                    onSelected: (_) => setState(() => suite.settings.set('bmpGapMs', ms)),
                  ),
              ]),
              const SizedBox(height: 10),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Fast radio (Bluetooth 5 2M)', style: TextStyle(fontWeight: FontWeight.w600)),
                subtitle: const Text('Up to twice as fast if the glasses accept it. Applies on the next connect.',
                    style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                value: suite.settings.ble2m,
                onChanged: (v) => setState(() => suite.settings.set('ble2m', v)),
              ),
              const Text('Picture packet size (tuning)', style: TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final (size, label) in const [(194, '194 B (demo)'), (236, '236 B (fewer packets)')])
                  ChoiceChip(
                    label: Text(label),
                    selected: suite.settings.bmpChunk == size,
                    onSelected: (_) => setState(() => suite.settings.set('bmpChunk', size)),
                  ),
              ]),
              const SizedBox(height: 10),
              Row(children: [
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _uploading
                        ? null
                        : () async {
                            setState(() => _uploading = true);
                            final ok = await log.flush();
                            if (!context.mounted) return;
                            setState(() => _uploading = false);
                            showToast(context, ok ? 'Uploaded to Termux' : 'Upload failed: is Agent Hub running?');
                          },
                    icon: const Icon(Icons.upload_rounded),
                    label: const Text('Upload now'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: () => setState(log.clear),
                  child: const Text('Clear'),
                ),
              ]),
            ]),
          ),
        ),
            ]);
          }
          if (index == 1) {
            return
        SizedBox(
          height: 48,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            children: [
              for (final f in filters)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: Text(f),
                    selected: _filter == f,
                    onSelected: (_) => setState(() => _filter = f),
                  ),
                ),
            ],
          ),
        );
          }
          if (shown.isEmpty) {
            return const Padding(
              padding: EdgeInsets.all(32),
              child: Center(child: Text('No events yet', style: TextStyle(color: AppColors.muted))),
            );
          }
          final i = index - 2;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Builder(builder: (context) {
                    final e = shown[i];
                    final t = e.time;
                    final ts = '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:'
                        '${t.second.toString().padLeft(2, '0')}.${(t.millisecond ~/ 100)}';
                    final bad = e.data['ok'] == false || e.name == 'error';
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 3),
                      child: Text.rich(
                        TextSpan(children: [
                          TextSpan(text: '$ts  ', style: const TextStyle(color: AppColors.muted)),
                          TextSpan(
                              text: '${e.cat}.${e.name}  ',
                              style: TextStyle(
                                  color: bad ? AppColors.warn : AppColors.glasses, fontWeight: FontWeight.w700)),
                          TextSpan(text: e.summary),
                        ]),
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 12, height: 1.35),
                      ),
                    );
            }),
          );
        },
      ),
    );
  }
}

/// Picture test: run it, see per-trial answers, results go to the log.
class _LabCard extends StatelessWidget {
  const _LabCard();

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final lab = suite.lab;
    return ListenableBuilder(
      listenable: Listenable.merge([lab, suite.ble]),
      builder: (context, _) {
        // c9 count per trial and arm
        final summary = <String, Map<String, (int, int)>>{};
        for (final r in lab.results) {
          if (!r.step.startsWith('crc')) continue;
          final t = summary.putIfAbsent(r.trial, () => {});
          final (ok, n) = t[r.arm] ?? (0, 0);
          t[r.arm] = (ok + (r.crc == 'c9' ? 1 : 0), n + 1);
        }
        String arm(Map<String, (int, int)> m, String a) {
          final v = m[a];
          return v == null ? '-' : '${v.$1}/${v.$2}';
        }

        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          child: SectionCard(
            title: 'Picture test',
            icon: Icons.science_rounded,
            iconColor: AppColors.claude,
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              const Text(
                  'Sends test pictures 12 different ways (about 30 s) and logs exactly what each side of the '
                  'glasses answers, so the sending can be fixed from facts. Stops screen sharing while it runs.',
                  style: TextStyle(color: AppColors.muted, fontSize: 13)),
              const SizedBox(height: 10),
              if (lab.running) ...[
                LinearProgressIndicator(value: lab.progress),
                const SizedBox(height: 6),
                Text(lab.status, style: const TextStyle(fontSize: 13)),
              ] else
                FilledButton.icon(
                  onPressed: suite.ble.isConnected ? suite.runPictureTest : null,
                  icon: const Icon(Icons.play_arrow_rounded),
                  label: Text(suite.ble.isConnected ? 'Run picture test' : 'Connect the glasses first'),
                ),
              if (summary.isNotEmpty) ...[
                const SizedBox(height: 10),
                const Text('Accepted (left / right):', style: TextStyle(fontWeight: FontWeight.w600)),
                for (final e in summary.entries)
                  Padding(
                    padding: const EdgeInsets.only(top: 3),
                    child: Text('${e.key}:  L ${arm(e.value, 'L')}  R ${arm(e.value, 'R')}',
                        style: const TextStyle(fontSize: 12.5, fontFamily: 'monospace')),
                  ),
              ],
            ]),
          ),
        );
      },
    );
  }
}
