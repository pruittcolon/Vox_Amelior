import 'package:flutter/material.dart';

import '../app/theme.dart';
import '../core/lab/glasses_lab.dart';

/// While Claude lab is on: says so on every screen, with a way to turn it off.
class LabBanner extends StatelessWidget {
  const LabBanner({super.key, required this.lab});

  final GlassesLab lab;

  static String _at(DateTime t) => '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) => ListenableBuilder(
        listenable: lab,
        builder: (context, _) => AnimatedSize(
          duration: const Duration(milliseconds: 200),
          child: !lab.on
              ? const SizedBox(width: double.infinity)
              : Material(
                  color: AppColors.claude.withOpacity(0.16),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(14, 4, 6, 4),
                    child: Row(children: [
                      const Icon(Icons.science_rounded, size: 18, color: AppColors.claude),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          lab.error != null
                              ? 'Claude lab is on - not reaching Agent Hub'
                              : lab.lastAt == null
                                  ? 'Claude lab is on - Claude can control your glasses'
                                  : 'Claude lab is on - last: ${lab.lastCommand} at ${_at(lab.lastAt!)}',
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                        ),
                      ),
                      TextButton(onPressed: () => lab.setOn(false), child: const Text('Turn off')),
                    ]),
                  ),
                ),
        ),
      );
}
