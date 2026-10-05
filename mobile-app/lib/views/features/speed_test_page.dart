import 'dart:async';
import 'dart:typed_data';

import 'package:demo_ai_even/services/fast_bmp.dart';
import 'package:demo_ai_even/services/proto.dart';
import 'package:flutter/material.dart';

/// Streams generated frames (clock, frame counter, moving bar) to the glasses
/// as fast as they are accepted and shows the measured frame rate.
class SpeedTestPage extends StatefulWidget {
  const SpeedTestPage({super.key});

  @override
  State<SpeedTestPage> createState() => _SpeedTestPageState();
}

class _SpeedTestPageState extends State<SpeedTestPage> {
  final streamer = FrameStreamer();
  final canvas = G1Canvas();
  Timer? _timer;
  int _tick = 0;

  @override
  void initState() {
    super.initState();
    streamer.onStats = () {
      if (mounted) setState(() {});
    };
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _start() {
    streamer.reset();
    _tick = 0;
    // Draw faster than the link can send; the streamer keeps only the newest.
    _timer = Timer.periodic(const Duration(milliseconds: 50), (_) {
      streamer.push(_frame(_tick++));
    });
    setState(() {});
  }

  Future<void> _stop() async {
    _timer?.cancel();
    _timer = null;
    streamer.dropPending();
    setState(() {});
    await Proto.exit();
  }

  Uint8List _frame(int tick) {
    final now = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    canvas.clear();
    canvas.text(4, 4, '${two(now.hour)}:${two(now.minute)}:${two(now.second)}');
    canvas.text(330, 4, 'FPS ${streamer.fps.toStringAsFixed(1)}');
    canvas.rect(0, 24, G1Canvas.width, 25);
    canvas.text(4, 40, 'FRAME ${streamer.framesOk + 1}', scale: 4);
    final x = (tick * 12) % (G1Canvas.width - 40);
    canvas.rect(4, 112, G1Canvas.width - 4, 130, fill: false);
    canvas.rect(6 + x, 114, 6 + x + 36, 128);
    return canvas.toBmp();
  }

  Widget _stat(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [Text(label), Text(value, style: const TextStyle(fontSize: 18))],
        ),
      );

  @override
  Widget build(BuildContext context) {
    final running = _timer != null;
    return Scaffold(
      appBar: AppBar(title: const Text('BMP Speed Test')),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          children: [
            _stat('Frames per second', streamer.fps.toStringAsFixed(2)),
            _stat('Last frame', '${streamer.lastFrameMs} ms'),
            _stat('Frames shown', '${streamer.framesOk}'),
            _stat('Frames failed', '${streamer.framesFailed}'),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Wait for end-command reply'),
              subtitle: const Text('Off = send CRC immediately (experimental)'),
              value: streamer.waitForEnd,
              onChanged: (v) => setState(() => streamer.waitForEnd = v),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton(
                onPressed: running ? _stop : _start,
                child: Text(running ? 'Stop' : 'Start streaming'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
