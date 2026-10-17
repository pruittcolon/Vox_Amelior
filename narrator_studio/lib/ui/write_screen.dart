import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../core/script/prompts.dart';
import 'art.dart';
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
      body: SafeArea(
        bottom: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: Text('Write with Claude', style: text.headlineLarge)),
                IconButton(
                  tooltip: 'Script format guide',
                  icon: const Icon(Icons.help_outline_rounded),
                  onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const GuideScreen())),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Claude writes the episode, the voices perform it.',
              style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant),
            ),
            const SizedBox(height: 16),
            Card(
              clipBehavior: Clip.antiAlias,
              child: ExpansionTile(
                leading: const IconTile(icon: Icons.auto_awesome_rounded, seed: 'claude-write', size: 38),
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
                            child: Text('${i + 1}', style: TextStyle(fontSize: 12, color: scheme.onPrimary, fontWeight: FontWeight.w700)),
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
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onCopy,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  CoverArt(seed: 'style-${prompt.style}', style: prompt.style, size: 52, radius: 14),
                  const SizedBox(width: 14),
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
                ],
              ),
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.tonalIcon(
                  onPressed: onCopy,
                  icon: const Icon(Icons.copy_rounded, size: 18),
                  label: const Text('Copy prompt'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
