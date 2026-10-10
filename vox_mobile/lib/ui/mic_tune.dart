import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/service/protocol.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/controls.dart';

/// What the microphone is hearing right now (both 0–1 on a dB scale).
class MicReading {
  const MicReading({this.level = 0, this.peak = 0, this.clipping = false, this.live = false});

  final double level;
  final double peak;
  final bool clipping;

  /// Fresh readings are arriving from the listening service.
  final bool live;
}

/// "+15%", "−10%" or "None" for a mic boost factor.
String formatBoost(double gain) {
  final pct = ((gain - 1) * 100).round();
  if (pct == 0) return 'None';
  return pct > 0 ? '+$pct%' : '−${-pct}%';
}

/// Reads a typed percentage change ("+15%", "15", "-10", "−10%") as a boost factor.
double? parseBoost(String text) {
  final v = double.tryParse(text.trim().replaceAll('%', '').replaceAll('+', '').replaceAll('−', '-'));
  return v == null ? null : 1 + v / 100;
}

/// One-tap starting points for how much the microphone picks up.
class MicPreset {
  const MicPreset(this.name, this.icon, this.gain, this.threshold, this.hint);

  final String name;
  final IconData icon;
  final double gain;
  final double threshold;
  final String hint;

  bool matches(AppSettings s) => (s.micGain - gain).abs() < 0.005 && (s.vadThreshold - threshold).abs() < 0.005;

  static const List<MicPreset> all = [
    MicPreset('Sensitive', Icons.hearing_rounded, 1.5, 0.4, 'Far-away and quiet voices'),
    MicPreset('Balanced', Icons.balance_rounded, 1.15, 0.5, 'Everyday rooms'),
    MicPreset('Noise-proof', Icons.noise_control_off_rounded, 1.0, 0.65, 'TV, traffic, crowds'),
  ];

  static MicPreset? of(AppSettings s) {
    for (final p in all) {
      if (p.matches(s)) return p;
    }
    return null;
  }
}

/// Turns the service's live level on while a screen is showing it.
///
/// Readings are requested only while this widget is visible (not a hidden
/// tab or a page underneath another), the app is in the foreground and Vox
/// is listening. The request is renewed every couple of seconds, so the
/// service stops by itself if the app goes away without saying so.
class MicMeterBinding extends StatefulWidget {
  const MicMeterBinding({super.key, required this.services, required this.builder});

  final AppServices services;
  final Widget Function(BuildContext context, MicReading reading) builder;

  @override
  State<MicMeterBinding> createState() => _MicMeterBindingState();
}

class _MicMeterBindingState extends State<MicMeterBinding> with WidgetsBindingObserver {
  StreamSubscription<Map<Object?, Object?>>? _sub;
  Timer? _tick;
  DateTime _last = DateTime.fromMillisecondsSinceEpoch(0);
  MicReading _reading = const MicReading();
  bool _asked = false;
  bool _visible = true;
  bool _foreground = true;
  int _ticks = 0;

  /// Meters requesting readings right now; the service is told to stop only when the last one goes.
  static int _users = 0;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    final state = WidgetsBinding.instance.lifecycleState;
    _foreground = state == null || state == AppLifecycleState.resumed;
    _sub = s.listening.events.listen(_onEvent);
    s.listening.addListener(_sync);
    _tick = Timer.periodic(const Duration(milliseconds: 500), (_) => _onTick());
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // False for a tab that is not selected and for a page covered by another.
    _visible = TickerMode.valuesOf(context).enabled;
    _sync();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _foreground = state == AppLifecycleState.resumed;
    _sync();
  }

  void _sync() {
    final want = s.listening.isListening && _visible && _foreground;
    if (want != _asked) {
      _asked = want;
      _users += want ? 1 : -1;
      if (want || _users <= 0) s.listening.levelMeter(want);
    }
    if (!want && _reading.live && mounted) setState(() => _reading = const MicReading());
  }

  void _onEvent(Map<Object?, Object?> e) {
    if (e['type'] != ServiceEvents.level || !mounted || !_asked) return;
    _last = DateTime.now();
    setState(() => _reading = MicReading(
          level: (e['level'] as num?)?.toDouble() ?? 0,
          peak: (e['peak'] as num?)?.toDouble() ?? 0,
          clipping: e['clipping'] == true,
          live: true,
        ));
  }

  void _onTick() {
    if (!mounted) return;
    // Renew the request (it lapses in the service after a few seconds).
    if (_asked && ++_ticks % 4 == 0) s.listening.levelMeter(true);
    if (_reading.live && DateTime.now().difference(_last) > const Duration(milliseconds: 1500)) {
      setState(() => _reading = const MicReading());
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_sub?.cancel());
    _tick?.cancel();
    s.listening.removeListener(_sync);
    if (_asked) {
      _users--;
      if (_users <= 0) s.listening.levelMeter(false);
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => widget.builder(context, _reading);
}

/// A horizontal loudness bar with a "good" zone and a peak marker.
class LevelMeterBar extends StatelessWidget {
  const LevelMeterBar({super.key, required this.reading, this.height = 18});

  final MicReading reading;
  final double height;

  static const double goodFrom = 0.4;
  static const double goodTo = 0.85;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final level = reading.level.clamp(0.0, 1.0);
    final color = reading.clipping || level > goodTo
        ? t.colorScheme.error
        : (level < goodFrom ? const Color(0xFFB08800) : const Color(0xFF0E9F6E));
    return Semantics(
      label: 'Microphone level',
      value: '${(level * 100).round()} percent',
      child: LayoutBuilder(
        builder: (context, box) {
          final w = box.maxWidth;
          return SizedBox(
            width: double.infinity,
            height: height,
            child: Stack(
              children: [
                Positioned.fill(
                  child: DecoratedBox(
                    decoration: BoxDecoration(color: t.colorScheme.outlineVariant.withValues(alpha: 0.45), borderRadius: BorderRadius.circular(height)),
                  ),
                ),
                Positioned(
                  left: w * goodFrom,
                  width: w * (goodTo - goodFrom),
                  top: 0,
                  bottom: 0,
                  child: DecoratedBox(decoration: BoxDecoration(color: const Color(0xFF0E9F6E).withValues(alpha: 0.14))),
                ),
                Positioned(
                  left: 0,
                  top: 0,
                  bottom: 0,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 140),
                    curve: Curves.easeOut,
                    width: reading.live ? (w * level).clamp(height, w) : 0,
                    decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(height)),
                  ),
                ),
                if (reading.live)
                  AnimatedPositioned(
                    duration: const Duration(milliseconds: 140),
                    left: (w * reading.peak.clamp(0.0, 1.0) - 2).clamp(0, w - 3),
                    top: 2,
                    bottom: 2,
                    width: 3,
                    child: DecoratedBox(decoration: BoxDecoration(color: t.colorScheme.onSurface.withValues(alpha: 0.55), borderRadius: BorderRadius.circular(2))),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}

/// What to tell the person about the recent loudness.
({String text, IconData icon, Color? color}) micVerdict(BuildContext context, MicReading r, {required bool listening, required double recentLevel}) {
  final t = Theme.of(context);
  if (!listening) return (text: 'Turn Vox on to see what the microphone hears.', icon: Icons.mic_off_rounded, color: null);
  if (!r.live) return (text: 'Waiting for sound…', icon: Icons.hourglass_empty_rounded, color: null);
  if (r.clipping) return (text: 'Too loud: the sound is distorting. Lower the boost.', icon: Icons.warning_amber_rounded, color: t.colorScheme.error);
  if (recentLevel < LevelMeterBar.goodFrom - 0.05) {
    return (text: 'Quiet. Raise the boost if people are far away.', icon: Icons.volume_down_rounded, color: const Color(0xFFB08800));
  }
  return (text: 'Good level.', icon: Icons.check_circle_rounded, color: const Color(0xFF0E9F6E));
}

/// The microphone card: live level, one-tap presets and the boost slider.
class MicTuneCard extends StatefulWidget {
  const MicTuneCard({super.key, required this.services});

  final AppServices services;

  @override
  State<MicTuneCard> createState() => _MicTuneCardState();
}

class _MicTuneCardState extends State<MicTuneCard> {
  // Loudest recent reading, so the message does not flicker between words.
  double _recent = 0;
  DateTime _recentAt = DateTime.fromMillisecondsSinceEpoch(0);

  AppServices get s => widget.services;

  double _hold(MicReading r) {
    final now = DateTime.now();
    if (r.live && (r.level >= _recent || now.difference(_recentAt) > const Duration(seconds: 4))) {
      _recent = r.level;
      _recentAt = now;
    }
    if (!r.live) _recent = 0;
    return _recent;
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return ListenableBuilder(
      listenable: Listenable.merge([s.settings, s.listening]),
      builder: (context, _) {
        final st = s.settings.value;
        final preset = MicPreset.of(st);
        return Card(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 8, 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                MicMeterBinding(
                  services: s,
                  builder: (context, r) {
                    final v = micVerdict(context, r, listening: s.listening.isListening, recentLevel: _hold(r));
                    return Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const IconBadge(Icons.mic_rounded),
                              const SizedBox(width: 12),
                              Expanded(child: Text('Microphone', style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800))),
                              _LivePill(live: s.listening.isListening),
                            ],
                          ),
                          const SizedBox(height: 14),
                          LevelMeterBar(reading: r),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Icon(v.icon, size: 16, color: v.color ?? t.colorScheme.onSurfaceVariant),
                              const SizedBox(width: 6),
                              Expanded(child: Text(v.text, style: t.textTheme.bodyMedium?.copyWith(color: v.color ?? t.colorScheme.onSurfaceVariant))),
                            ],
                          ),
                        ],
                      ),
                    );
                  },
                ),
                const SizedBox(height: 14),
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 4,
                    children: [
                      for (final p in MicPreset.all)
                        ChoiceChip(
                          avatar: Icon(p.icon, size: 18),
                          showCheckmark: false,
                          label: Text(p.name),
                          selected: preset == p,
                          onSelected: (_) {
                            if (preset != p) s.updateSettings(st.copyWith(micGain: p.gain, vadThreshold: p.threshold));
                          },
                        ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 6, right: 8),
                  child: Text(
                    preset?.hint ?? 'Custom: you have set your own values.',
                    style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
                  ),
                ),
                SettingSlider(
                  padding: const EdgeInsets.fromLTRB(0, 12, 0, 4),
                  title: 'Mic boost',
                  subtitle: 'Helps Vox notice quiet or faraway speech. Transcripts always use the clean recording.',
                  value: st.micGain,
                  min: 0.5,
                  max: 4.0,
                  step: 0.05,
                  defaultValue: 1.15,
                  format: formatBoost,
                  parse: parseBoost,
                  unitHint: 'Percent change, for example 15 for +15%',
                  lowLabel: '−50%',
                  highLabel: '+300%',
                  onChanged: (v) => s.updateSettings(st.copyWith(micGain: v)),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _LivePill extends StatelessWidget {
  const _LivePill({required this.live});

  final bool live;

  @override
  Widget build(BuildContext context) {
    final c = live ? const Color(0xFF0E9F6E) : Theme.of(context).colorScheme.outline;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: c.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(99)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.circle, size: 8, color: c),
          const SizedBox(width: 6),
          Text(live ? 'Live' : 'Off', style: TextStyle(color: c, fontWeight: FontWeight.w700, fontSize: 12)),
        ],
      ),
    );
  }
}

/// Quick access to the microphone controls from anywhere.
Future<void> showMicTuneSheet(BuildContext context, AppServices services) => showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (_) => SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(12, 0, 12, 24),
        child: MicTuneCard(services: services),
      ),
    );
