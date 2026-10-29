import 'package:flutter/material.dart';

import '../../app/shell.dart';
import '../../widgets/common.dart';
import '../../widgets/glasses_widgets.dart';

/// The demo's Text page: show any text on the glasses, five lines a page.
class TextPage extends StatefulWidget {
  const TextPage({super.key});

  @override
  State<TextPage> createState() => _TextPageState();
}

class _TextPageState extends State<TextPage> {
  static const sample = '''Welcome to G1.

You're holding the first eyewear ever designed to blend stunning aesthetics, amazing wearability and useful functionality.

At Even Realities we continuously explore the human relationship with technology. And our breakthrough is a pair of glasses that are unique, clever and capable but are still everyday glasses. The ones you'll reach for every morning and want to wear all day.

So you can see what matters. When it matters.''';

  final _text = TextEditingController(text: sample);

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final suite = SuiteScope.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Send text')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
        children: [
          TextField(
            controller: _text,
            minLines: 6,
            maxLines: 12,
            decoration: const InputDecoration(hintText: 'Text to show on the glasses'),
          ),
          const SizedBox(height: 12),
          ListenableBuilder(
            listenable: _text,
            builder: (context, _) => FilledButton.icon(
              onPressed: _text.text.trim().isEmpty
                  ? null
                  : () {
                      suite.display.showText(_text.text, title: 'Text');
                      if (!suite.ble.isConnected) showToast(context, 'Glasses are not connected');
                    },
              icon: const Icon(Icons.send_to_mobile_rounded),
              label: const Text('Show on glasses'),
            ),
          ),
          const SizedBox(height: 20),
          SectionCard(
            title: 'Preview',
            child: GlassesPreview(display: suite.display, connected: suite.ble.isConnected),
          ),
        ],
      ),
    );
  }
}
