import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../core/script/prompts.dart';
import 'edit_text_screen.dart';
import 'guide_screen.dart';
import 'scope.dart';

/// Make scripts with Claude: copy a prompt, paste the reply back.
class WriteScreen extends StatefulWidget {
  const WriteScreen({super.key});

  @override
  State<WriteScreen> createState() => _WriteScreenState();
}

class _WriteScreenState extends State<WriteScreen> {
  final _topic = TextEditingController();
  double _minutes = 10;

  @override
  void dispose() {
    _topic.dispose();
    super.dispose();
  }

  Future<void> _copy(ScriptPrompt prompt) async {
    await Clipboard.setData(ClipboardData(text: prompt.fill(topic: _topic.text, minutes: _minutes.round())));
    if (!mounted) return;
    context.toast(
      'Prompt copied. Paste it into Claude.',
      action: SnackBarAction(label: 'Open Claude', onPressed: () => context.scope.native.openClaude()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Write with Claude'),
        actions: [
          IconButton(
            tooltip: 'Script format guide',
            icon: const Icon(Icons.help_outline),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const GuideScreen())),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
        children: [
          Card(
            clipBehavior: Clip.antiAlias,
            child: ExpansionTile(
              shape: const Border(),
              leading: Icon(Icons.auto_awesome, color: scheme.primary),
              title: const Text('Copy a prompt, paste it into Claude, paste the reply back here.'),
              subtitle: const Text('How it works'),
              childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              children: [
                for (var i = 0; i < howItWorks.length; i++)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 5),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        CircleAvatar(
                          radius: 12,
                          backgroundColor: scheme.primary,
                          child: Text('${i + 1}', style: TextStyle(fontSize: 12, color: scheme.onPrimary)),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text.rich(TextSpan(children: [
                            TextSpan(text: '${howItWorks[i].$1}. ', style: const TextStyle(fontWeight: FontWeight.w700)),
                            TextSpan(text: howItWorks[i].$2),
                          ])),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _topic,
            decoration: const InputDecoration(
              labelText: 'Your topic',
              hintText: 'e.g. how black holes evaporate',
            ),
            textCapitalization: TextCapitalization.sentences,
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Text('Length: ${_minutes.round()} min', style: text.bodyMedium),
              Expanded(
                child: Slider(
                  value: _minutes,
                  min: 3,
                  max: 30,
                  divisions: 27,
                  label: '${_minutes.round()} min',
                  onChanged: (v) => setState(() => _minutes = v),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          for (final prompt in scriptPrompts) ...[
            _PromptCard(prompt: prompt, onCopy: () => _copy(prompt)),
            const SizedBox(height: 10),
          ],
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const EditTextScreen())),
            icon: const Icon(Icons.content_paste),
            label: const Text('Paste script from Claude'),
          ),
          const SizedBox(height: 12),
          Text(
            'Text or PDF? Ask Claude for plain text and paste it: the speaker tags, pauses and music cues survive. '
            'PDFs are for things that already exist (a book, a paper): open them directly and they read as an audiobook, '
            'or attach the PDF to one of the prompts above to turn it into a podcast.',
            style: text.bodySmall,
          ),
        ],
      ),
    );
  }
}

class _PromptCard extends StatelessWidget {
  const _PromptCard({required this.prompt, required this.onCopy});
  final ScriptPrompt prompt;
  final VoidCallback onCopy;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(prompt.title, style: text.titleMedium),
                  const SizedBox(height: 2),
                  Text(prompt.when, style: text.bodySmall),
                ],
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.tonalIcon(onPressed: onCopy, icon: const Icon(Icons.copy, size: 18), label: const Text('Copy prompt')),
          ],
        ),
      ),
    );
  }
}
