import 'dart:async';

import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/voice/voice_controller.dart';

String clockTime(DateTime t) =>
    '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

/// The big round talk button. Idle: mic. Taking a message: glowing coral
/// ring that follows your voice; tap to stop.
class MicButton extends StatelessWidget {
  const MicButton({super.key, required this.voice, required this.onTap, this.size = 84});

  final VoiceController voice;
  final VoidCallback onTap;
  final double size;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: voice,
      builder: (context, _) {
        final on = voice.dictating;
        final waiting = voice.mode == VoiceMode.waiting;
        final color = on ? AppColors.claude : (waiting ? AppColors.glasses : AppColors.surfaceHigh);
        return Semantics(
          button: true,
          label: on ? 'Stop talking' : 'Talk to Claude',
          child: ValueListenableBuilder<double>(
            valueListenable: voice.level,
            builder: (context, level, child) {
              final glow = on ? 6 + level * 18 : (waiting ? 4.0 : 0.0);
              return AnimatedContainer(
                duration: const Duration(milliseconds: 120),
                width: size,
                height: size,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: color,
                  border: Border.all(color: on ? Colors.white24 : AppColors.outline, width: 1.5),
                  boxShadow: [
                    if (glow > 0)
                      BoxShadow(color: color.withOpacity(0.45), blurRadius: glow * 2, spreadRadius: glow / 2),
                  ],
                ),
                child: child,
              );
            },
            child: Material(
              type: MaterialType.transparency,
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: onTap,
                child: Icon(
                  on ? Icons.stop_rounded : Icons.mic_rounded,
                  size: size * 0.42,
                  color: on ? const Color(0xFF1F0E08) : AppColors.text,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// One line telling the user what the voice system is doing.
class VoiceStatus extends StatelessWidget {
  const VoiceStatus({super.key, required this.voice, this.target = 'Claude'});

  final VoiceController voice;
  final String target;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: voice,
      builder: (context, _) {
        String text;
        Color color = AppColors.muted;
        if (voice.error != null) {
          text = voice.error!;
          color = AppColors.warn;
        } else if (voice.dictating) {
          text = voice.holding
              ? 'Listening... let go of the TouchBar to send'
              : 'Listening... say “send” when done';
          color = AppColors.claude;
        } else if (voice.mode == VoiceMode.waiting) {
          text = 'Say “Claude”, your message, then “send”';
          color = AppColors.glasses;
        } else if (voice.hint != null) {
          text = voice.hint!;
        } else {
          text = 'Tap the mic to talk to $target';
        }
        return AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: Text(text,
              key: ValueKey(text),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: color, fontSize: 13.5, fontWeight: FontWeight.w600)),
        );
      },
    );
  }
}

/// Hands-free on/off with the time it switches itself off.
class HandsFreeTile extends StatelessWidget {
  const HandsFreeTile({super.key, required this.voice, required this.onChanged, this.autoOffMinutes = 10});

  final VoiceController voice;
  final ValueChanged<bool> onChanged;
  final int autoOffMinutes;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: voice,
      builder: (context, _) {
        final on = voice.handsFree;
        final off = voice.autoOffAt;
        return Container(
          padding: const EdgeInsets.fromLTRB(14, 6, 6, 6),
          decoration: BoxDecoration(
            color: on ? AppColors.claude.withOpacity(0.10) : AppColors.surfaceHigh,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: on ? AppColors.claude.withOpacity(0.4) : AppColors.outline),
          ),
          child: Row(children: [
            Icon(on ? Icons.hearing_rounded : Icons.hearing_disabled_rounded,
                color: on ? AppColors.claude : AppColors.muted),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Hands-free “Claude”', style: TextStyle(fontWeight: FontWeight.w700)),
                Text(
                  on && off != null
                      ? 'Listening · turns off at ${clockTime(off)}'
                      : 'Off · turns itself off after $autoOffMinutes min',
                  style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
                ),
              ]),
            ),
            Switch(value: on, onChanged: onChanged),
          ]),
        );
      },
    );
  }
}

/// Slim bar shown on every tab while hands-free is on, with a Stop button.
class HandsFreeBanner extends StatefulWidget {
  const HandsFreeBanner({super.key, required this.voice, required this.onStop});

  final VoiceController voice;
  final VoidCallback onStop;

  @override
  State<HandsFreeBanner> createState() => _HandsFreeBannerState();
}

class _HandsFreeBannerState extends State<HandsFreeBanner> {
  Timer? _tick;

  @override
  void initState() {
    super.initState();
    // Refresh the "min left" once a minute.
    _tick = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted && widget.voice.handsFree) setState(() {});
    });
  }

  @override
  void dispose() {
    _tick?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.voice,
      builder: (context, _) {
        final v = widget.voice;
        final show = v.handsFree || v.dictating;
        final off = v.autoOffAt;
        final left = off == null ? null : off.difference(DateTime.now()).inMinutes + 1;
        return AnimatedSize(
          duration: const Duration(milliseconds: 200),
          child: !show
              ? const SizedBox(width: double.infinity)
              : Container(
                  width: double.infinity,
                  color: (v.dictating ? AppColors.claude : AppColors.glasses).withOpacity(0.14),
                  padding: const EdgeInsets.fromLTRB(16, 4, 4, 4),
                  child: Row(children: [
                    Icon(v.dictating ? Icons.graphic_eq_rounded : Icons.hearing_rounded,
                        size: 18, color: v.dictating ? AppColors.claude : AppColors.glasses),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        v.dictating
                            ? 'Taking your message…'
                            : 'Listening for “Claude”${left != null ? ' · $left min left' : ''}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                      ),
                    ),
                    TextButton(onPressed: widget.onStop, child: const Text('Stop')),
                  ]),
                ),
        );
      },
    );
  }
}
