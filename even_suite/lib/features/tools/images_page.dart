import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../app/shell.dart';
import '../../app/theme.dart';
import '../../widgets/common.dart';

/// The demo's BMP page: show a 576x136 1-bit picture on both lenses.
class ImagesPage extends StatefulWidget {
  const ImagesPage({super.key});

  @override
  State<ImagesPage> createState() => _ImagesPageState();
}

class _ImagesPageState extends State<ImagesPage> {
  static const images = ['assets/images/image_1.bmp', 'assets/images/image_2.bmp'];
  String? _sending;

  Future<void> _show(String asset) async {
    final suite = SuiteScope.of(context);
    if (!suite.ble.isConnected) return showToast(context, 'Connect your glasses first');
    setState(() => _sending = asset);
    final sw = Stopwatch()..start();
    final data = await rootBundle.load(asset);
    final ok = await suite.proto.sendBmp(data.buffer.asUint8List(), requireCrc: false);
    if (!mounted) return;
    setState(() => _sending = null);
    showToast(context, ok ? 'On your glasses (${sw.elapsedMilliseconds} ms)' : 'The glasses did not confirm the picture');
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Images')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
        children: [
          const Text('Pictures are sent to both lenses at once with flow control, so they arrive fast and whole.',
              style: TextStyle(color: AppColors.muted)),
          const SizedBox(height: 14),
          for (final (i, asset) in images.indexed)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Card(
                clipBehavior: Clip.antiAlias,
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  Container(
                    color: AppColors.hudBg,
                    padding: const EdgeInsets.all(12),
                    child: AspectRatio(
                      aspectRatio: 576 / 136,
                      child: ColorFiltered(
                        // White pixels are the lit ones; the glasses draw them green on black.
                        colorFilter: const ColorFilter.matrix([
                          0, 0, 0, 0, 0x6C, //
                          0, 0, 0, 0, 0xF5, //
                          0, 0, 0, 0, 0xA0, //
                          0.333, 0.333, 0.333, 0, 0, //
                        ]),
                        child: Image.asset(asset, fit: BoxFit.contain, filterQuality: FilterQuality.none),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
                    child: Row(children: [
                      Expanded(child: Text('Sample ${i + 1}', style: const TextStyle(fontWeight: FontWeight.w700))),
                      FilledButton.tonalIcon(
                        onPressed: _sending != null ? null : () => _show(asset),
                        icon: _sending == asset
                            ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.visibility_rounded),
                        label: Text(_sending == asset ? 'Sending…' : 'Show'),
                      ),
                    ]),
                  ),
                ]),
              ),
            ),
          OutlinedButton.icon(
            onPressed: suite.display.clear,
            icon: const Icon(Icons.layers_clear_rounded),
            label: const Text('Clear the glasses'),
          ),
        ],
      ),
    );
  }
}
