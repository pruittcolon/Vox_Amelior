import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/glasses/claude_panel.dart';
import '../../core/glasses/g1_font.dart';
import '../../core/glasses/hud_frame.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';
import 'layout_page.dart';
import 'look_designer.dart';

/// What the glasses show when you look up: clock, weather, layout, and four
/// notes you pick - Claude's finished jobs, the last answer, your phone's
/// notifications, or your own text. Phone notifications can also pop up on
/// the glasses as they arrive.
class LookUpPage extends StatefulWidget {
  const LookUpPage({super.key});

  @override
  State<LookUpPage> createState() => _LookUpPageState();
}

class _LookUpPageState extends State<LookUpPage> {
  bool _access = false;
  String _sample = 'live'; // what the preview shows: live, working, answering, done
  final _custom = TextEditingController();
  final _city = TextEditingController();
  bool _ready = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_ready) return;
    _ready = true;
    final s = SuiteScope.of(context).settings;
    _custom.text = s.dashCustom;
    _city.text = s.dashCity;
    _checkAccess();
  }

  Future<void> _checkAccess() async {
    final ok = await SuiteScope.of(context).system.notificationAccess();
    if (mounted) setState(() => _access = ok);
  }

  @override
  void dispose() {
    _custom.dispose();
    _city.dispose();
    super.dispose();
  }

  static const _sources = [
    ('work', 'Claude’s finished jobs'),
    ('reply', 'Claude’s last answer'),
    ('phone', 'Phone notifications'),
    ('custom', 'My note'),
    ('off', 'Nothing'),
  ];

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    final s = suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Look up screen')),
      body: ListenableBuilder(
        listenable: Listenable.merge([s, suite.dashboard, suite.phone]),
        builder: (context, _) {
          final slots = s.noteSlots;
          final custom = s.lookMode == 'custom';
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              SegmentedButton<String>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: 'custom', icon: Icon(Icons.dashboard_customize_rounded), label: Text('My layout')),
                  ButtonSegment(value: 'dashboard', icon: Icon(Icons.view_quilt_rounded), label: Text('Built-in')),
                ],
                selected: {s.lookMode},
                onSelectionChanged: (v) => s.set('lookMode', v.first),
              ),
              const SizedBox(height: 14),
              if (custom) ...[
                const LookDesigner(),
                const SizedBox(height: 12),
                SectionCard(
                  title: 'HUD frame',
                  icon: Icons.filter_frames_rounded,
                  iconColor: AppColors.hud,
                  child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                    const Text('Thin lines drawn around your layout on the glasses, for a heads-up display look.',
                        style: TextStyle(color: AppColors.muted, fontSize: 13)),
                    const SizedBox(height: 10),
                    Wrap(spacing: 8, runSpacing: 8, children: [
                      for (final (key, label) in HudFrame.styles)
                        ChoiceChip(
                          label: Text(label),
                          selected: s.lookFrame == key,
                          onSelected: (_) {
                            s.set('lookFrame', key);
                            if (suite.ble.isConnected) suite.showLookNow();
                          },
                        ),
                    ]),
                  ]),
                ),
              ] else
                ..._dashboard(context, suite),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Clock and weather',
                icon: Icons.wb_sunny_rounded,
                iconColor: AppColors.warn,
                child: _clockAndWeather(context, suite),
              ),
              const SizedBox(height: 12),
              if (!custom && s.dashRight == 'notes') ...[_notes(context, suite, slots), const SizedBox(height: 12)],
              _notifications(context, suite),
            ],
          );
        },
      ),
    );
  }

  List<Widget> _dashboard(BuildContext context, Suite suite) {
    final s = suite.settings;
    final dash = suite.dashboard;
    return [
              if (dash.foreignWrites > 0) ...[_OtherApp(suite: suite), const SizedBox(height: 12)],
              _Preview(suite: suite, sample: s.dashRight == 'claude' ? _sample : 'live'),
              if (s.dashRight == 'claude') ...[
                const SizedBox(height: 8),
                Wrap(spacing: 6, runSpacing: 6, children: [
                  for (final (v, label) in const [('live', 'Now'), ('working', 'Working'), ('answering', 'Answering'), ('done', 'Done')])
                    ChoiceChip(
                      label: Text(label),
                      visualDensity: VisualDensity.compact,
                      selected: _sample == v,
                      onSelected: (_) => setState(() => _sample = v),
                    ),
                ]),
              ],
              const SizedBox(height: 10),
              Row(children: [
                Expanded(
                  child: Text(
                    dash.error != null
                        ? 'Last update failed: ${dash.error}'
                        : (dash.lastSync == null ? 'Not sent yet' : 'Sent at ${_clock(dash.lastSync!)}'),
                    style: TextStyle(color: dash.error != null ? AppColors.warn : AppColors.muted, fontSize: 12.5),
                  ),
                ),
                FilledButton.icon(
                  onPressed: dash.syncing
                      ? null
                      : () async {
                          if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
                          await suite.dashboard.sync(force: true);
                          if (context.mounted) showToast(context, 'Look up to see it');
                        },
                  icon: const Icon(Icons.upload_rounded),
                  label: Text(dash.syncing ? 'Sending…' : 'Update now'),
                ),
              ]),
              const SizedBox(height: 12),
              SectionCard(
                title: 'Right side',
                icon: Icons.dashboard_customize_rounded,
                iconColor: AppColors.hud,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  _Choices('dashRight', s.dashRight, const [
                    ('claude', 'Claude (live)'),
                    ('notes', 'My notes'),
                    ('calendar', 'Calendar'),
                    ('news', 'News'),
                    ('stocks', 'Stocks'),
                    ('map', 'Map'),
                  ]),
                  const SizedBox(height: 8),
                  Text(
                    switch (s.dashRight) {
                      'claude' => 'Time on the left, Claude on the right: what it is doing now, what you asked, '
                          'then the result. Updates every few seconds while you look up.',
                      'notes' => 'Up to four notes you pick below; tap on the glasses to page through them.',
                      _ => 'The glasses’ own panel, filled by the Even app.',
                    },
                    style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                  ),
                  if (s.dashRight != 'claude') ...[
                    const SizedBox(height: 12),
                    const Text('Layout', style: TextStyle(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 8),
                    _Choices('dashMode', s.dashMode, const [('dual', 'Dual'), ('full', 'Full'), ('minimal', 'Minimal')]),
                  ],
                ]),
              ),
              if (s.dashRight == 'claude') ...[
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  onPressed: () =>
                      Navigator.of(context).push(MaterialPageRoute(builder: (_) => const LayoutPage())),
                  icon: const Icon(Icons.drag_indicator_rounded),
                  label: const Text('Arrange the layout (drag and drop)'),
                ),
              ],
    ];
  }

  Widget _clockAndWeather(BuildContext context, Suite suite) {
    final s = suite.settings;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  Row(children: [
                    Expanded(
                      child: SegmentedButton<bool>(
                        showSelectedIcon: false,
                        segments: const [
                          ButtonSegment(value: false, label: Text('12 h')),
                          ButtonSegment(value: true, label: Text('24 h')),
                        ],
                        selected: {s.dash24h},
                        onSelectionChanged: (v) => s.set('dash24h', v.first),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: SegmentedButton<bool>(
                        showSelectedIcon: false,
                        segments: const [
                          ButtonSegment(value: true, label: Text('°F')),
                          ButtonSegment(value: false, label: Text('°C')),
                        ],
                        selected: {s.dashFahrenheit},
                        onSelectionChanged: (v) => s.set('dashFahrenheit', v.first),
                      ),
                    ),
                  ]),
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Weather', style: TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: const Text('From Open-Meteo (free, no account).',
                        style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                    value: s.dashWeather,
                    onChanged: (v) async {
                      if (v && s.dashCity.isEmpty) await suite.system.permissions(['location']);
                      s.set('dashWeather', v);
                    },
                  ),
                  if (s.dashWeather)
                    TextField(
                      controller: _city,
                      decoration: const InputDecoration(
                          labelText: 'City', helperText: 'Leave empty to use where the phone is.'),
                      onSubmitted: (v) => s.set('dashCity', v.trim()),
                      onTapOutside: (_) {
                        if (_city.text.trim() != s.dashCity) s.set('dashCity', _city.text.trim());
                        FocusScope.of(context).unfocus();
                      },
                    ),
                ]);
  }

  Widget _notes(BuildContext context, Suite suite, List<String> slots) {
    final s = suite.settings;
    return SectionCard(
                title: 'Notes',
                icon: Icons.sticky_note_2_rounded,
                iconColor: AppColors.claude,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  const Text('Four notes show in the Notes panel. Pick what each one shows.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  for (var i = 0; i < 4; i++)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: DropdownButtonFormField<String>(
                        value: _sources.any((x) => x.$1 == slots[i]) ? slots[i] : 'off',
                        isExpanded: true,
                        decoration: InputDecoration(labelText: 'Note ${i + 1}'),
                        items: [for (final (v, label) in _sources) DropdownMenuItem(value: v, child: Text(label))],
                        onChanged: (v) {
                          final next = List.of(slots)..[i] = v ?? 'off';
                          s.set('dashNotes', next.join(','));
                        },
                      ),
                    ),
                  if (slots.contains('custom')) ...[
                    const SizedBox(height: 12),
                    TextField(
                      controller: _custom,
                      minLines: 2,
                      maxLines: 5,
                      decoration: const InputDecoration(
                          labelText: 'My note', helperText: 'First line is the title. Anything you like.'),
                      onChanged: (v) => s.set('dashCustom', v),
                    ),
                  ],
                ]),
              );
  }

  Widget _notifications(BuildContext context, Suite suite) {
    final s = suite.settings;
    return SectionCard(
                title: 'Phone notifications',
                icon: Icons.notifications_rounded,
                iconColor: AppColors.glasses,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Show them on the glasses', style: TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: const Text('They pop up as they arrive and wait for you when you look up.',
                        style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                    value: s.notifyForward && _access,
                    onChanged: (v) async {
                      if (v && !_access) {
                        await suite.system.openNotificationAccess();
                        if (context.mounted) {
                          showToast(context, 'Turn on Even Suite in the list, then come back');
                        }
                      }
                      s.set('notifyForward', v);
                    },
                  ),
                  if (!_access)
                    OutlinedButton.icon(
                      onPressed: () async {
                        await suite.system.openNotificationAccess();
                        await Future<void>.delayed(const Duration(seconds: 1));
                        await _checkAccess();
                      },
                      icon: const Icon(Icons.lock_open_rounded),
                      label: const Text('Give notification access'),
                    ),
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Private', style: TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: const Text('Only the app and sender, not the message.',
                        style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                    value: s.notifyPrivate,
                    onChanged: (v) => s.set('notifyPrivate', v),
                  ),
                  if (suite.phone.seen.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    Text(s.allowedApps.isEmpty ? 'All apps (tap to choose only some)' : 'Only these apps',
                        style: const TextStyle(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 8),
                    Wrap(spacing: 8, runSpacing: 8, children: [
                      for (final e in suite.phone.seen.entries)
                        FilterChip(
                          label: Text(e.value),
                          selected: s.allowedApps.contains(e.key),
                          onSelected: (on) {
                            final next = Set.of(s.allowedApps);
                            on ? next.add(e.key) : next.remove(e.key);
                            s.set('notifyApps', next.join(','));
                          },
                        ),
                    ]),
                  ],
                ]),
              );
  }

  static String _clock(DateTime t) => '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
}

/// Another app (the Even app) keeps rewriting the glasses' dashboard - every
/// minute - so what you pick here seems to snap back. Even Suite puts yours
/// back right after it (with "Keep it up to date" on); force-stopping the
/// Even app ends the tug of war.
class _OtherApp extends StatelessWidget {
  const _OtherApp({required this.suite});
  final Suite suite;

  @override
  Widget build(BuildContext context) {
    final at = suite.dashboard.foreignAt;
    final ago = at == null ? '' : DateTime.now().difference(at).inSeconds < 90 ? ' (just now)' : '';
    return SectionCard(
      title: 'Another app changes your dashboard',
      icon: Icons.sync_problem_rounded,
      iconColor: AppColors.warn,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Text(
          'The Even app is running too and resets the glasses’ dashboard about every minute$ago. '
          'Even Suite puts yours back within a second, but they keep taking turns until the Even app is stopped.',
          style: const TextStyle(color: AppColors.muted, fontSize: 13),
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          onPressed: () async {
            await suite.system.openAppSettings('com.even.g1');
            if (context.mounted) showToast(context, 'Tap Force stop - the Even app stops changing it');
          },
          icon: const Icon(Icons.block_rounded),
          label: const Text('Stop the Even app'),
        ),
      ]),
    );
  }
}

class _Choices extends StatelessWidget {
  const _Choices(this.keyName, this.value, this.options);
  final String keyName;
  final String value;
  final List<(String, String)> options;

  @override
  Widget build(BuildContext context) {
    final s = SuiteScope.of(context).settings;
    return Wrap(spacing: 8, runSpacing: 8, children: [
      for (final (v, label) in options)
        ChoiceChip(label: Text(label), selected: value == v, onSelected: (_) => s.set(keyName, v)),
    ]);
  }
}

/// The look-up screen as the glasses will draw it (dual layout, G1 font widths).
class _Preview extends StatelessWidget {
  const _Preview({required this.suite, this.sample = 'live'});
  final Suite suite;
  final String sample;

  @override
  Widget build(BuildContext context) {
    final s = suite.settings;
    final dash = suite.dashboard;
    final w = dash.weather;
    final temp = !s.dashWeather || w == null
        ? ''
        : (s.dashFahrenheit ? '${(w.tempC * 9 / 5 + 32).round()}°F' : '${w.tempC}°C');
    String title = '';
    List<PanelLine> rows = const [];
    String? label;
    if (s.dashRight == 'claude') {
      final p = sample == 'live' ? dash.panel : ClaudePanel.example(sample, PanelLayout.parse(s.panelLayout));
      (title, rows) = (p.title, p.rows);
    } else if (s.dashRight == 'notes') {
      final notes = dash.buildNotes().where((n) => n.$1.isNotEmpty || n.$2.isNotEmpty).toList();
      if (notes.isNotEmpty) {
        title = G1Font.fit(notes.first.$1, DashLayout.titleMaxPx);
        rows = [for (final l in G1Font.wrapMax(notes.first.$2, DashLayout.bodyMaxPx, DashLayout.bodyLines)) PanelLine(l)];
      }
    } else {
      label = '${s.dashRight[0].toUpperCase()}${s.dashRight.substring(1)} (from the Even app)';
    }
    return Container(
      // A margin inside the rounded frame, so its corners do not cut the glasses' edges.
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: AppColors.hudBg,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.hud.withOpacity(0.18)),
      ),
      child: GlassesDashboardView(
        now: DateTime.now(),
        h24: s.dash24h,
        temp: temp,
        title: title,
        rows: rows,
        paneLabel: label,
      ),
    );
  }
}

