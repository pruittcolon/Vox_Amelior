import 'dart:math';

import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../core/home/voice_monkey.dart';
import '../../core/suite.dart';
import '../../widgets/common.dart';

/// The lights (Alexa routines through Voice Monkey) and the Vizio TV: set up
/// once here, then by voice - "all lights off", "TV off", "TV remote".
class SmartHomePage extends StatefulWidget {
  const SmartHomePage({super.key});

  @override
  State<SmartHomePage> createState() => _SmartHomePageState();
}

class _SmartHomePageState extends State<SmartHomePage> {
  late final Suite _suite;
  late final TextEditingController _token;
  late final TextEditingController _devices;
  late final TextEditingController _ip;
  bool _started = false;
  bool _showToken = false;
  bool _busy = false;
  bool? _tvOn;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_started) return;
    _started = true;
    _suite = SuiteScope.of(context);
    final s = _suite.settings;
    _token = TextEditingController(text: s.vmToken);
    _devices = TextEditingController(text: s.vmDevices);
    _ip = TextEditingController(text: s.tvIp);
    _checkTv();
  }

  @override
  void dispose() {
    _token.dispose();
    _devices.dispose();
    _ip.dispose();
    super.dispose();
  }

  Future<void> _checkTv() async {
    final s = _suite.settings;
    if (s.tvIp.isEmpty || s.tvAuth.isEmpty) return;
    final on = await _suite.tv.power(s.tvIp, s.tvAuth);
    if (mounted) setState(() => _tvOn = on);
  }

  Future<void> _run(Future<String> Function() job) async {
    setState(() => _busy = true);
    try {
      final msg = await job();
      if (mounted && msg.isNotEmpty) showToast(context, msg);
    } catch (e) {
      if (mounted) showToast(context, '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _find() => _run(() async {
        final found = await _suite.tv.discover();
        if (found == null) return 'No Vizio TV answered on this Wi-Fi. Type its address instead.';
        _suite.settings
          ..set('tvIp', found.$1)
          ..set('tvName', found.$2);
        _ip.text = found.$1;
        return 'Found ${found.$2} at ${found.$1}';
      });

  Future<void> _pair() async {
    final s = _suite.settings;
    final ip = _ip.text.trim();
    if (ip.isEmpty) return showToast(context, 'Find the TV or type its address first');
    if (ip != s.tvIp) s.set('tvIp', ip);
    final id = s.tvDeviceId.isNotEmpty
        ? s.tvDeviceId
        : 'even-suite-${List.generate(6, (_) => Random().nextInt(16).toRadixString(16)).join()}';
    setState(() => _busy = true);
    try {
      await _suite.tv.startPairing(ip, id);
    } catch (e) {
      if (mounted) {
        setState(() => _busy = false);
        showToast(context, 'The TV did not start pairing: $e');
      }
      return;
    }
    if (!mounted) return;
    setState(() => _busy = false);
    final pin = await showDialog<String>(
      context: context,
      builder: (context) {
        final c = TextEditingController();
        return AlertDialog(
          title: const Text('PIN on the TV'),
          content: TextField(
            controller: c,
            autofocus: true,
            keyboardType: TextInputType.number,
            maxLength: 4,
            decoration: const InputDecoration(hintText: '4 digits'),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(context, c.text), child: const Text('Pair')),
          ],
        );
      },
    );
    if (pin == null || pin.trim().isEmpty) {
      await _suite.tv.cancelPairing(ip);
      return;
    }
    await _run(() async {
      final token = await _suite.tv.finishPairing(ip, pin);
      s
        ..set('tvAuth', token)
        ..set('tvDeviceId', id);
      if (s.tvName.isEmpty) s.set('tvName', await _suite.tv.name(ip) ?? 'TV');
      await _checkTv();
      return 'Paired with ${s.tvName}';
    });
  }

  Future<void> _key(List<String> keys) => _run(() async {
        final s = _suite.settings;
        await _suite.tv.keys(s.tvIp, s.tvAuth, keys);
        if (keys.first.startsWith('POWER')) Future.delayed(const Duration(seconds: 2), _checkTv);
        return '';
      }).then((_) {});

  @override
  Widget build(BuildContext context) {
    final s = _suite.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Lights and TV')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) {
          final devices = VoiceMonkey.parse(s.vmDevices);
          final paired = s.tvIp.isNotEmpty && s.tvAuth.isNotEmpty;
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
            children: [
              SectionCard(
                title: 'Lights',
                icon: Icons.lightbulb_rounded,
                iconColor: AppColors.warn,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  const Text(
                      'Alexa routines through Voice Monkey, as in the old app. Paste your token from the Voice Monkey '
                      'dashboard (or a device’s whole trigger URL). It stays on this phone.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                  const SizedBox(height: 10),
                  TextField(
                    controller: _token,
                    obscureText: !_showToken,
                    decoration: InputDecoration(
                      labelText: 'Voice Monkey token',
                      suffixIcon: IconButton(
                        tooltip: _showToken ? 'Hide' : 'Show',
                        icon: Icon(_showToken ? Icons.visibility_off_rounded : Icons.visibility_rounded),
                        onPressed: () => setState(() => _showToken = !_showToken),
                      ),
                    ),
                    onChanged: (v) => s.set('vmToken', v.trim()),
                  ),
                  const SizedBox(height: 12),
                  for (final d in devices)
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: const Icon(Icons.lightbulb_outline_rounded, color: AppColors.warn),
                      title: Text(d.name, style: const TextStyle(fontWeight: FontWeight.w700)),
                      subtitle: Text('${d.device}\nSay: ${d.phrases.map((p) => '“$p”').join(', ')} (on or off)',
                          style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
                      isThreeLine: true,
                      trailing: OutlinedButton(
                        onPressed: _busy ? null : () => _run(() => _suite.voiceMonkey.trigger(s.vmToken, d)),
                        child: const Text('Run'),
                      ),
                    ),
                  const SizedBox(height: 4),
                  ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    title: const Text('Devices and what to say', style: TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: const Text('One per line: Name = device id: phrase, phrase',
                        style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                    children: [
                      TextField(
                        controller: _devices,
                        minLines: 3,
                        maxLines: 8,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 12.5),
                        onChanged: (v) => s.set('vmDevices', v),
                      ),
                      Align(
                        alignment: Alignment.centerRight,
                        child: TextButton(
                          onPressed: () {
                            _devices.text = VoiceMonkey.defaultDevices;
                            s.set('vmDevices', VoiceMonkey.defaultDevices);
                          },
                          child: const Text('Back to the start'),
                        ),
                      ),
                    ],
                  ),
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: s.tvName.isEmpty ? 'Vizio TV' : s.tvName,
                icon: Icons.tv_rounded,
                iconColor: AppColors.glasses,
                trailing: paired
                    ? StatusPill(_tvOn == null ? 'Paired' : (_tvOn! ? 'On' : 'Off'),
                        color: _tvOn == false ? AppColors.muted : AppColors.ok)
                    : const StatusPill('Not paired', color: AppColors.muted),
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  Row(children: [
                    Expanded(
                      child: TextField(
                        controller: _ip,
                        keyboardType: TextInputType.url,
                        decoration: const InputDecoration(labelText: 'TV address', hintText: '192.168.x.x'),
                        onSubmitted: (v) => s.set('tvIp', v.trim()),
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(onPressed: _busy ? null : _find, child: const Text('Find')),
                  ]),
                  const SizedBox(height: 10),
                  FilledButton.tonalIcon(
                    onPressed: _busy ? null : _pair,
                    icon: const Icon(Icons.link_rounded),
                    label: Text(paired ? 'Pair again' : 'Pair (the TV shows a PIN)'),
                  ),
                  if (paired) ...[
                    const SizedBox(height: 14),
                    _Remote(onKeys: _busy ? null : _key),
                  ],
                ]),
              ),
              const SizedBox(height: 12),
              SectionCard(
                title: 'By voice',
                icon: Icons.record_voice_over_rounded,
                iconColor: AppColors.claude,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Without the wake word', style: TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: const Text(
                        'While hands-free listens, these work on their own - no “Claude” first. Anything else '
                        'still needs the wake word.',
                        style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                    value: s.triggerWords,
                    onChanged: (v) => s.set('triggerWords', v),
                  ),
                  const SizedBox(height: 6),
                  for (final (say, does) in const [
                    ('“Set a timer for 5 minutes”', 'a timer on the phone, shown on the glasses when it is done'),
                    ('“Remind me in 20 minutes to …”', 'a reminder'),
                    ('“All lights off”, “Turn on the kitchen light”', 'your Voice Monkey devices'),
                    ('“TV off”, “TV on”, “TV volume up”, “Mute the TV”, “TV HDMI 2”', 'the TV, at once'),
                    ('“TV remote”', 'then just “up”, “down”, “OK”, “back”, “home”, “volume up”, “pause”… until “done"'),
                  ])
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text.rich(TextSpan(children: [
                        TextSpan(text: '$say\n', style: const TextStyle(fontWeight: FontWeight.w600)),
                        TextSpan(text: does, style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
                      ])),
                    ),
                  const Text('With the wake word or the TouchBar they work too - “Claude, all lights off”.',
                      style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
                ]),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// A TV remote: power, volume, the arrows and OK, back, home, input, play.
class _Remote extends StatelessWidget {
  const _Remote({required this.onKeys});
  final Future<void> Function(List<String> keys)? onKeys;

  @override
  Widget build(BuildContext context) {
    Widget b(IconData icon, String tip, List<String> keys, {Color? color}) => IconButton.filledTonal(
          tooltip: tip,
          onPressed: onKeys == null ? null : () => onKeys!(keys),
          icon: Icon(icon, color: color),
        );
    return Column(children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
        b(Icons.power_settings_new_rounded, 'Off', const ['POWER_OFF'], color: AppColors.error),
        b(Icons.power_rounded, 'On', const ['POWER_ON'], color: AppColors.ok),
        b(Icons.input_rounded, 'Input', const ['INPUT']),
        b(Icons.home_rounded, 'Home', const ['HOME']),
      ]),
      const SizedBox(height: 6),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [b(Icons.keyboard_arrow_up_rounded, 'Up', const ['UP'])]),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        b(Icons.keyboard_arrow_left_rounded, 'Left', const ['LEFT']),
        const SizedBox(width: 8),
        b(Icons.circle_outlined, 'OK', const ['OK']),
        const SizedBox(width: 8),
        b(Icons.keyboard_arrow_right_rounded, 'Right', const ['RIGHT']),
      ]),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        b(Icons.keyboard_arrow_down_rounded, 'Down', const ['DOWN'])
      ]),
      const SizedBox(height: 6),
      Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
        b(Icons.arrow_back_rounded, 'Back', const ['BACK']),
        b(Icons.volume_down_rounded, 'Volume down', const ['VOL_DOWN']),
        b(Icons.volume_off_rounded, 'Mute', const ['MUTE']),
        b(Icons.volume_up_rounded, 'Volume up', const ['VOL_UP']),
        b(Icons.play_arrow_rounded, 'Play', const ['PLAY']),
        b(Icons.pause_rounded, 'Pause', const ['PAUSE']),
      ]),
    ]);
  }
}
