import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/assistant/prompt_builder.dart';

const List<(String, String)> _presets = [
  ('Short & plain', PromptBuilder.defaultInstructions),
  (
    'Detailed',
    'Answer thoroughly from the conversation excerpts and tool results only; never invent details. '
        'Explain who said what and when, and quote the key lines.'
  ),
  (
    'Bullet points',
    'Answer as short bullet points from the conversation excerpts and tool results only; never invent details. '
        'Start each bullet with who said it and when.'
  ),
  (
    'Coach',
    'Answer from the conversation excerpts and tool results only; never invent details. Be kind and constructive: '
        'point out patterns and suggest one practical next step.'
  ),
];

/// Edits the instructions Gemma gets with every question.
Future<void> showPromptEditor(BuildContext context, AppServices services) async {
  final st = services.settings.value;
  final controller = TextEditingController(text: st.instructions.isEmpty ? PromptBuilder.defaultInstructions : st.instructions);
  final saved = await showModalBottomSheet<String>(
    context: context,
    isScrollControlled: true,
    builder: (c) {
      final t = Theme.of(c);
      return Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(c).bottom),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text('How Gemma answers', style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('Sent with every question. Vox also tells Gemma the date and time, who lives here, and the tools it may use.',
                    style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant)),
                const SizedBox(height: 14),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    for (final (name, text) in _presets) ActionChip(label: Text(name), onPressed: () => controller.text = text),
                  ],
                ),
                const SizedBox(height: 14),
                TextField(
                  controller: controller,
                  minLines: 5,
                  maxLines: 12,
                  decoration: const InputDecoration(hintText: 'How should Vox answer?'),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    TextButton(onPressed: () => Navigator.pop(c, ''), child: const Text('Reset to default')),
                    const Spacer(),
                    FilledButton(onPressed: () => Navigator.pop(c, controller.text.trim()), child: const Text('Save')),
                  ],
                ),
              ],
            ),
          ),
        ),
      );
    },
  );
  controller.dispose();
  if (saved == null) return;
  final value = saved == PromptBuilder.defaultInstructions ? '' : saved;
  await services.updateSettings(services.settings.value.copyWith(instructions: value));
}
