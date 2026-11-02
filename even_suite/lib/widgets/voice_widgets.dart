import 'dart:async';

import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/bridge/bridge_client.dart';
import '../core/settings.dart';
import '../core/talk/conversation.dart';
import '../core/voice/speaker.dart';
import '../core/voice/voice_controller.dart';

String clockTime(DateTime t) =>
    '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

/// Who answers: Claude, or Google Antigravity (agy) for when Claude has run
/// out. The same voice and glasses; each keeps its own conversation.
class AgentChooser extends StatelessWidget {
  const AgentChooser({super.key, required this.settings, required this.onChanged, this.hub});
  final Settings settings;
  final void Function(String agent) onChanged;

  /// What the running Agent Hub can do (null: not checked yet).
  final HubInfo? hub;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
        listenable: settings,
        builder: (context, _) {
          final agy = settings.agent == 'agy';
          return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            SegmentedButton<String>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(value: 'claude', icon: Icon(Icons.auto_awesome_rounded), label: Text('Claude')),
                ButtonSegment(value: 'agy', icon: Icon(Icons.rocket_launch_rounded), label: Text('Antigravity')),
              ],
              selected: {settings.agent},
              onSelectionChanged: (v) => onChanged(v.first),
            ),
            const SizedBox(height: 6),
            Text(
              agy
                  ? 'Google Antigravity answers instead - for when Claude has run out. Same voice, same glasses; '
                      'say “use Claude” to switch back.'
                  : 'Claude Code answers. If you run out, switch to Antigravity here or say “use agy”.',
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            if (agy && hub != null && !hub!.agyLive)
              const Padding(
                padding: EdgeInsets.only(top: 6),
                child: Text(
                  'Your Agent Hub is older: in Termux type  even restart  so Antigravity answers live '
                  'and knows it is on the glasses.',
                  style: TextStyle(color: AppColors.warn, fontSize: 12.5),
                ),
              ),
          ]);
        },
      );
}

/// One question (the standard: ask, read the answer, it goes) or back and
/// forth (after an answer it listens for your follow-up).
class TalkModeChooser extends StatelessWidget {
  const TalkModeChooser({super.key, required this.settings, this.glassesMic = false});
  final Settings settings;

  /// Follow-ups are heard through the glasses' microphone.
  final bool glassesMic;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
        listenable: settings,
        builder: (context, _) {
          final chat = settings.talkMode == 'chat';
          return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            SegmentedButton<String>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(value: 'single', icon: Icon(Icons.chat_bubble_outline_rounded), label: Text('One question')),
                ButtonSegment(value: 'chat', icon: Icon(Icons.forum_outlined), label: Text('Back and forth')),
              ],
              selected: {settings.talkMode},
              onSelectionChanged: (v) => settings.set('talkMode', v.first),
            ),
            const SizedBox(height: 6),
            Text(
              chat
                  ? 'After each answer it listens for ${settings.followUpSec} s for a follow-up'
                      '${glassesMic ? ' through the glasses\' mic' : ''}. Silence or “thanks” ends it, and it '
                      'stops after a few turns in a row.'
                  : 'Ask, read the answer, and it goes away by itself. Nothing listens afterwards.',
              style: const TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
          ]);
        },
      );
}

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
              : voice.pauseSends
                  ? 'Listening... it sends when you pause (tap the glasses to send now)'
                  : 'Listening... say “send” when done';
          color = AppColors.claude;
        } else if (voice.mode == VoiceMode.waiting) {
          text = 'Say “${voice.wake.label}”, your message, then “send”';
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
                Text('Hands-free “${voice.wake.label}”', style: const TextStyle(fontWeight: FontWeight.w700)),
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
                            : 'Listening for “${v.wake.label}”${left != null ? ' · $left min left' : ''}',
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

/// The talk button with the whole back-and-forth on it: listening (glow that
/// follows your voice), Claude working (spinning ring), the phone speaking
/// (tap to stop), and "your turn" for a follow-up.
class TalkButton extends StatelessWidget {
  const TalkButton({
    super.key,
    required this.voice,
    required this.talk,
    required this.narrator,
    required this.onTap,
    this.size = 84,
  });

  final VoiceController voice;
  final Conversation talk;
  final Narrator narrator;
  final VoidCallback onTap;
  final double size;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([voice, talk, narrator]),
      builder: (context, _) {
        final speaking = narrator.speaking;
        final followUp = voice.followUp && voice.dictating && voice.draft.isEmpty;
        final listening = voice.dictating && !followUp;
        final working = talk.busy && !speaking && !listening;
        final waiting = voice.mode == VoiceMode.waiting;
        Color color;
        IconData icon;
        String label;
        if (speaking) {
          (color, icon, label) = (AppColors.claude, Icons.volume_up_rounded, 'Stop talking');
        } else if (followUp) {
          (color, icon, label) = (AppColors.glasses, Icons.mic_rounded, 'Your turn: ask a follow-up');
        } else if (listening) {
          (color, icon, label) = (AppColors.claude, Icons.stop_rounded, 'Stop talking');
        } else if (waiting) {
          (color, icon, label) = (AppColors.glasses, Icons.mic_rounded, 'Talk to Claude');
        } else {
          (color, icon, label) = (AppColors.surfaceHigh, Icons.mic_rounded, 'Talk to Claude');
        }
        final dark = speaking || listening || followUp;
        return Semantics(
          button: true,
          label: label,
          child: SizedBox(
            width: size + 16,
            height: size + 16,
            child: Stack(alignment: Alignment.center, children: [
              if (working || followUp)
                SizedBox(
                  width: size + 12,
                  height: size + 12,
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: working ? AppColors.claude : AppColors.glasses,
                  ),
                ),
              if (speaking) _Ripple(size: size, color: AppColors.claude),
              ValueListenableBuilder<double>(
                valueListenable: voice.level,
                builder: (context, level, child) {
                  final glow = listening ? 6 + level * 18 : (waiting || followUp ? 4.0 : 0.0);
                  return AnimatedContainer(
                    duration: const Duration(milliseconds: 120),
                    width: size,
                    height: size,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: color,
                      border: Border.all(color: dark ? Colors.white24 : AppColors.outline, width: 1.5),
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
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 180),
                      transitionBuilder: (c, a) => ScaleTransition(scale: a, child: c),
                      child: Icon(icon,
                          key: ValueKey(icon),
                          size: size * 0.42,
                          color: dark ? const Color(0xFF1F0E08) : AppColors.text),
                    ),
                  ),
                ),
              ),
            ]),
          ),
        );
      },
    );
  }
}

/// Soft rings spreading out while the phone talks.
class _Ripple extends StatefulWidget {
  const _Ripple({required this.size, required this.color});
  final double size;
  final Color color;
  @override
  State<_Ripple> createState() => _RippleState();
}

class _RippleState extends State<_Ripple> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
        animation: _c,
        builder: (context, _) => Container(
          width: widget.size * (1 + 0.18 * _c.value),
          height: widget.size * (1 + 0.18 * _c.value),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: widget.color.withOpacity(0.6 * (1 - _c.value)), width: 3),
          ),
        ),
      );
}

/// One line saying where the back-and-forth is.
class TalkStatus extends StatelessWidget {
  const TalkStatus({super.key, required this.voice, required this.talk, required this.narrator, this.target = 'Claude'});

  final VoiceController voice;
  final Conversation talk;
  final Narrator narrator;
  final String target;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([voice, talk, narrator]),
      builder: (context, _) {
        String? text;
        var color = AppColors.claude;
        final secs = talk.elapsed.inSeconds;
        if (voice.error != null) {
          // the voice line below says it
        } else if (narrator.speaking) {
          text = 'Speaking… tap the button to stop';
        } else if (voice.followUp && voice.dictating) {
          text = voice.draft.isEmpty ? 'Your turn: ask a follow-up, or say “thanks”' : 'Listening…';
          color = AppColors.glasses;
        } else if (talk.stage == TalkStage.sent) {
          text = 'Got it · sending';
        } else if (talk.stage == TalkStage.working) {
          text = '${talk.step.isEmpty ? 'Thinking' : talk.step} · $secs s';
        } else if (talk.stage == TalkStage.answering) {
          text = 'Answering · $secs s';
        } else if (talk.stage == TalkStage.error) {
          text = 'Claude did not answer: ${talk.error}';
          color = AppColors.warn;
        }
        if (text == null) return VoiceStatus(voice: voice, target: target);
        return AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: Text(text,
              key: ValueKey(text.split('·').first),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: color, fontSize: 13.5, fontWeight: FontWeight.w600)),
        );
      },
    );
  }
}

/// Listening → Got it → Working → Answer → Speaking, lit up in turn.
class StageStrip extends StatelessWidget {
  const StageStrip({super.key, required this.talk, required this.voice, required this.narrator});

  final Conversation talk;
  final VoiceController voice;
  final Narrator narrator;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([talk, voice, narrator]),
      builder: (context, _) {
        final s = talk.stage;
        int current;
        if (voice.dictating && !(voice.followUp && voice.draft.isEmpty)) {
          current = 0;
        } else if (s == TalkStage.sent) {
          current = 1;
        } else if (s == TalkStage.working) {
          current = 2;
        } else if (s == TalkStage.answering) {
          current = 3;
        } else if (narrator.speaking || s == TalkStage.speaking) {
          current = 4;
        } else if (s == TalkStage.followUp) {
          current = 5;
        } else {
          current = -1;
        }
        final show = current >= 0 || s == TalkStage.error;
        const labels = ['Listen', 'Got it', 'Work', 'Answer', 'Speak', 'Your turn'];
        return AnimatedSize(
          duration: const Duration(milliseconds: 220),
          child: !show
              ? const SizedBox(width: double.infinity)
              : Padding(
                  padding: const EdgeInsets.fromLTRB(12, 0, 12, 6),
                  child: Row(children: [
                    for (var i = 0; i < labels.length; i++) ...[
                      if (i > 0)
                        Expanded(
                          child: Container(
                            height: 2,
                            color: i <= current ? AppColors.claude.withOpacity(0.7) : AppColors.outline,
                          ),
                        ),
                      _StageDot(label: labels[i], state: s == TalkStage.error && i == 2 ? -2 : (i < current ? 1 : (i == current ? 0 : -1))),
                    ],
                  ]),
                ),
        );
      },
    );
  }
}

class _StageDot extends StatelessWidget {
  const _StageDot({required this.label, required this.state});
  final String label;

  /// 1 done, 0 now, -1 not yet, -2 failed.
  final int state;

  @override
  Widget build(BuildContext context) {
    final color = switch (state) {
      1 => AppColors.claude,
      0 => AppColors.claude,
      -2 => AppColors.error,
      _ => AppColors.muted,
    };
    return Column(mainAxisSize: MainAxisSize.min, children: [
      AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        width: state == 0 ? 14 : 10,
        height: state == 0 ? 14 : 10,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: state >= 0 || state == -2 ? color : Colors.transparent,
          border: Border.all(color: color, width: 1.5),
          boxShadow: [if (state == 0) BoxShadow(color: color.withOpacity(0.6), blurRadius: 8)],
        ),
      ),
      const SizedBox(height: 3),
      Text(label,
          style: TextStyle(
              fontSize: 10, color: state == 0 ? AppColors.text : AppColors.muted, fontWeight: FontWeight.w600)),
    ]);
  }
}
