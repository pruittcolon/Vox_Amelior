import 'package:flutter/material.dart';

import '../../app/theme.dart';
import '../../core/bridge/chat_controller.dart';
import '../../core/suite.dart';

/// The two sections of the Terminal tab: Claude, and Antigravity (Google's
/// agent, for when Claude has run out). Each has its own chat, its own
/// sessions in Termux, its own colour. Tap one to move there.
class AgentSections extends StatelessWidget {
  const AgentSections({super.key, required this.suite});

  final Suite suite;

  static Color colorOf(String agent) => agent == 'agy' ? AppColors.agy : AppColors.claude;

  static IconData iconOf(String agent) => agent == 'agy' ? Icons.rocket_launch_rounded : Icons.auto_awesome_rounded;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 6),
      child: ListenableBuilder(
        listenable: Listenable.merge([suite.settings, suite.terminal, suite.chat]),
        builder: (context, _) {
          final agy = suite.settings.agent == 'agy';
          final hub = suite.chat.hub;
          return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Row(children: [
              Expanded(child: _Section(suite: suite, agent: 'claude')),
              const SizedBox(width: 8),
              Expanded(child: _Section(suite: suite, agent: 'agy')),
            ]),
            if (agy && hub != null && !hub.agyLive)
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
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.suite, required this.agent});

  final Suite suite;
  final String agent;

  @override
  Widget build(BuildContext context) {
    final selected = suite.settings.agent == agent;
    final color = AgentSections.colorOf(agent);
    final open = suite.terminal.ofAgent(agent);
    final working = open.any((s) => s.state == 'working' || s.state == 'asking');
    return Semantics(
      button: true,
      selected: selected,
      label: ChatController.agentLabel(agent),
      child: Material(
        color: selected ? color.withOpacity(0.16) : AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          key: ValueKey('section-$agent'),
          borderRadius: BorderRadius.circular(12),
          onTap: () => suite.openSection(agent),
          child: Container(
            height: 42,
            padding: const EdgeInsets.symmetric(horizontal: 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: selected ? color : AppColors.outline, width: selected ? 1.5 : 1),
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(AgentSections.iconOf(agent), size: 18, color: selected ? color : AppColors.muted),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  ChatController.agentLabel(agent),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontWeight: FontWeight.w800, color: selected ? AppColors.text : AppColors.muted),
                ),
              ),
              if (open.isNotEmpty) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                  decoration: BoxDecoration(
                    color: working ? AppColors.warn.withOpacity(0.25) : AppColors.outline,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text('${open.length}',
                      style: TextStyle(
                          fontSize: 11.5, fontWeight: FontWeight.w700, color: working ? AppColors.warn : AppColors.muted)),
                ),
              ],
            ]),
          ),
        ),
      ),
    );
  }
}
