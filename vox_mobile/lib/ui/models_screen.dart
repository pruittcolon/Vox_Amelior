import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/settings/app_settings.dart';
import 'package:vox_amelior_mobile/ui/format.dart';
import 'package:vox_amelior_mobile/ui/widgets.dart';

/// Download and choose the on-device models. Also the first-run setup.
///
/// Downloads belong to the app, not this screen: leaving it (or the app)
/// does not stop them.
class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key, required this.services, this.onDone});

  final AppServices services;

  /// First-run mode: shows a "Continue" button once speech models are ready.
  final VoidCallback? onDone;

  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen> {
  final _token = TextEditingController();
  bool _hasToken = false;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    s.tokens.huggingFace().then((t) {
      if (mounted) setState(() => _hasToken = t != null);
    });
  }

  @override
  void dispose() {
    _token.dispose();
    super.dispose();
  }

  Future<void> _saveToken() async {
    final t = _token.text.trim();
    if (!t.startsWith('hf_')) {
      showMessage(context, 'Hugging Face tokens start with "hf_".');
      return;
    }
    await s.tokens.saveHuggingFace(t);
    _token.clear();
    if (!mounted) return;
    setState(() => _hasToken = true);
    showMessage(context, 'Token saved securely on this phone.');
  }

  Future<void> _open(String url) async {
    if (!await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication) && mounted) {
      showMessage(context, 'Could not open $url');
    }
  }

  Future<void> _editCustom() async {
    final st = s.settings.value;
    final result = await showDialog<AppSettings>(context: context, builder: (_) => _CustomModelDialog(settings: st));
    if (result != null) await s.updateSettings(result.copyWith(llmId: ModelCatalog.customLlmId));
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(widget.onDone == null ? 'Models' : 'Set up Vox')),
      body: ListenableBuilder(
        listenable: Listenable.merge([s.downloads, s.settings]),
        builder: (context, _) {
          final st = s.settings.value;
          final speechAll = [...ModelCatalog.speech, ...ModelCatalog.speechExtras];
          final speechBytes = speechAll.fold<int>(0, (a, m) => a + m.approxDownloadBytes);
          final speechBusy = speechAll.any((m) => s.downloads.stateOf(m).isBusy);
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
            children: [
              Text(
                'Everything runs on this phone and nothing you say leaves it. Downloads continue in the background '
                'and resume automatically if interrupted. Wi-Fi recommended.',
                style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
              const SectionHeader('Speech', padding: EdgeInsets.fromLTRB(4, 20, 4, 8)),
              VoxCard(
                padding: const EdgeInsets.fromLTRB(4, 8, 4, 12),
                child: Column(
                  children: [
                    for (final m in speechAll) _ModelTile(asset: m, downloads: s.downloads),
                    // Optional smaller recognizer; downloads only when tapped. Also chosen in
                    // Settings → Microphone & hearing.
                    _ModelTile(asset: ModelCatalog.parakeet, downloads: s.downloads),
                    _ModelTile(asset: ModelCatalog.parakeetFp32, downloads: s.downloads),
                    // Optional: tone of voice and sounds per line; downloads only when tapped.
                    _ModelTile(asset: ModelCatalog.toneModel, downloads: s.downloads),
                    if (!s.downloads.speechReady)
                      Padding(
                        padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
                        child: SizedBox(
                          width: double.infinity,
                          child: FilledButton.icon(
                            onPressed: speechBusy ? null : s.downloads.downloadSpeechModels,
                            icon: const Icon(Icons.download_rounded),
                            label: Text('Download speech models (${formatBytes(speechBytes)})'),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SectionHeader('Assistant', padding: EdgeInsets.fromLTRB(4, 24, 4, 8)),
              VoxCard(
                padding: const EdgeInsets.fromLTRB(4, 8, 4, 8),
                child: Column(
                  children: [
                    for (final m in ModelCatalog.assistants)
                      _ModelTile(
                        asset: m,
                        downloads: s.downloads,
                        selected: st.llmId == m.id,
                        onSelect: () => s.updateSettings(st.copyWith(llmId: m.id)),
                      ),
                    if (st.customLlmUrl.trim().isNotEmpty)
                      _ModelTile(
                        asset: ModelCatalog.custom(
                          url: st.customLlmUrl,
                          name: st.customLlmName,
                          llmType: st.customLlmType,
                          supportsTools: st.customLlmTools,
                          requiresToken: st.customLlmNeedsToken,
                        ),
                        downloads: s.downloads,
                        selected: st.llmId == ModelCatalog.customLlmId,
                        onSelect: () => s.updateSettings(st.copyWith(llmId: ModelCatalog.customLlmId)),
                        onEdit: _editCustom,
                      ),
                    ListTile(
                      leading: const Icon(Icons.add_link_rounded),
                      title: Text(st.customLlmUrl.trim().isEmpty ? 'Use another model…' : 'Edit custom model…'),
                      subtitle: const Text('Any LiteRT-LM (.litertlm) model URL'),
                      onTap: _editCustom,
                    ),
                  ],
                ),
              ),
              const SectionHeader('Hugging Face (optional)', padding: EdgeInsets.fromLTRB(4, 24, 4, 8)),
              VoxCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Only needed for gated or private custom models. Gemma 4 downloads without it.'),
                    const SizedBox(height: 12),
                    if (_hasToken)
                      Row(
                        children: [
                          const Icon(Icons.verified_user_rounded, color: Color(0xFF0E9F6E)),
                          const SizedBox(width: 8),
                          const Expanded(child: Text('Token saved')),
                          TextButton(
                            onPressed: () async {
                              await s.tokens.clearHuggingFace();
                              if (mounted) setState(() => _hasToken = false);
                            },
                            child: const Text('Remove'),
                          ),
                        ],
                      )
                    else ...[
                      TextField(
                        controller: _token,
                        obscureText: true,
                        autocorrect: false,
                        enableSuggestions: false,
                        decoration: InputDecoration(
                          hintText: 'hf_…',
                          suffixIcon: IconButton(icon: const Icon(Icons.save_rounded), onPressed: _saveToken),
                        ),
                        onSubmitted: (_) => _saveToken(),
                      ),
                      TextButton.icon(
                        onPressed: () => _open('https://huggingface.co/settings/tokens'),
                        icon: const Icon(Icons.open_in_new_rounded, size: 18),
                        label: const Text('Create a read token'),
                      ),
                    ],
                  ],
                ),
              ),
              if (widget.onDone != null) ...[
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: s.downloads.speechReady ? widget.onDone : null,
                  child: Text(s.downloads.speechReady ? 'Continue' : 'Waiting for speech models…'),
                ),
                const SizedBox(height: 8),
                Text('You can leave this screen; downloads keep going.',
                    textAlign: TextAlign.center, style: t.textTheme.bodySmall),
              ],
            ],
          );
        },
      ),
    );
  }
}

class _ModelTile extends StatelessWidget {
  const _ModelTile({required this.asset, required this.downloads, this.selected, this.onSelect, this.onEdit});

  final ModelAsset asset;
  final ModelDownloads downloads;

  /// Non-null for choosable models (the assistant).
  final bool? selected;
  final VoidCallback? onSelect;
  final VoidCallback? onEdit;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final st = downloads.stateOf(asset);
    Widget status;
    List<Widget> actions = [];
    switch (st.status) {
      case DownloadStatus.installed:
        status = const Pill('Installed', icon: Icons.check_circle_rounded, color: Color(0xFF0E9F6E));
        actions = [
          IconButton(
            tooltip: 'Delete to free space',
            icon: const Icon(Icons.delete_outline_rounded),
            onPressed: () async {
              if (await confirm(context, 'Delete ${asset.title}?', 'You can download it again later.')) downloads.remove(asset);
            },
          ),
        ];
      case DownloadStatus.queued:
        status = const Pill('Waiting to download', icon: Icons.schedule_rounded);
        actions = [IconButton(tooltip: 'Cancel', icon: const Icon(Icons.close_rounded), onPressed: () => downloads.cancel(asset))];
      case DownloadStatus.downloading:
      case DownloadStatus.unpacking:
        final unpacking = st.status == DownloadStatus.unpacking;
        status = Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(value: st.progress, minHeight: 8),
            ),
            const SizedBox(height: 6),
            Text(describeDownload(st)),
            if (unpacking)
              Text(
                'Downloaded. Getting it ready to use — this can take a few minutes on a phone.',
                style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
          ],
        );
        actions = [
          if (!unpacking)
            IconButton(tooltip: 'Pause', icon: const Icon(Icons.pause_rounded), onPressed: () => downloads.cancel(asset)),
        ];
      case DownloadStatus.failed:
        status = Text(st.error ?? 'Failed', style: TextStyle(color: t.colorScheme.error));
        actions = [IconButton(tooltip: 'Retry', icon: const Icon(Icons.refresh_rounded), onPressed: () => downloads.download(asset))];
      case DownloadStatus.notInstalled:
        final partial = st.received > 0;
        final size = asset.approxDownloadBytes > 0 ? formatBytes(asset.approxDownloadBytes) : 'size unknown';
        status = Text(partial ? 'Paused at ${formatBytes(st.received)} of $size' : size,
            style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant));
        actions = [
          IconButton.filledTonal(
            tooltip: partial ? 'Resume' : 'Download',
            icon: Icon(partial ? Icons.play_arrow_rounded : Icons.download_rounded),
            onPressed: () => downloads.download(asset),
          ),
        ];
    }
    return ListTile(
      onTap: onSelect,
      leading: selected == null
          ? Icon(switch (asset.kind) {
              ModelKind.speechToText => Icons.hearing_rounded,
              ModelKind.voiceActivity => Icons.graphic_eq_rounded,
              ModelKind.speakerVoiceprint => Icons.fingerprint_rounded,
              ModelKind.speakerTurns => Icons.forum_rounded,
              ModelKind.toneOfVoice => Icons.mood_rounded,
              ModelKind.languageModel => Icons.auto_awesome_rounded,
            })
          : Icon(selected! ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded,
              color: selected! ? t.colorScheme.primary : null),
      title: Row(
        children: [
          Flexible(child: Text(asset.title, style: const TextStyle(fontWeight: FontWeight.w600))),
          if (asset.supportsTools) ...[const SizedBox(width: 6), const Pill('Agent', icon: Icons.bolt_rounded, color: Color(0xFF9C36B5))],
        ],
      ),
      subtitle: Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(asset.description, maxLines: 2, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 6),
            status,
          ],
        ),
      ),
      trailing: Row(mainAxisSize: MainAxisSize.min, children: [
        if (onEdit != null) IconButton(tooltip: 'Edit', icon: const Icon(Icons.edit_rounded), onPressed: onEdit),
        ...actions,
      ]),
    );
  }
}

class _CustomModelDialog extends StatefulWidget {
  const _CustomModelDialog({required this.settings});

  final AppSettings settings;

  @override
  State<_CustomModelDialog> createState() => _CustomModelDialogState();
}

class _CustomModelDialogState extends State<_CustomModelDialog> {
  late final _url = TextEditingController(text: widget.settings.customLlmUrl);
  late final _name = TextEditingController(text: widget.settings.customLlmName);
  late String _type = widget.settings.customLlmType;
  late bool _tools = widget.settings.customLlmTools;
  late bool _token = widget.settings.customLlmNeedsToken;
  String? _error;

  static const _types = {
    'gemma4': 'Gemma 4',
    'gemmaIt': 'Gemma 3 / 3n',
    'qwen3': 'Qwen 3',
    'qwen': 'Qwen 2.5',
    'phi': 'Phi-4',
    'deepSeek': 'DeepSeek R1',
    'llama': 'Llama',
    'general': 'Other',
  };

  @override
  void dispose() {
    _url.dispose();
    _name.dispose();
    super.dispose();
  }

  void _save() {
    final u = Uri.tryParse(_url.text.trim());
    if (u == null || u.scheme != 'https' || u.host.isEmpty) {
      setState(() => _error = 'Enter an https:// link to a .litertlm file.');
      return;
    }
    Navigator.pop(
      context,
      widget.settings.copyWith(
        customLlmUrl: _url.text.trim(),
        customLlmName: _name.text.trim(),
        customLlmType: _type,
        customLlmTools: _tools,
        customLlmNeedsToken: _token,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Custom assistant model'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _url,
              keyboardType: TextInputType.url,
              decoration: InputDecoration(labelText: 'Download URL (.litertlm)', errorText: _error),
            ),
            const SizedBox(height: 12),
            TextField(controller: _name, decoration: const InputDecoration(labelText: 'Name (optional)')),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: _types.containsKey(_type) ? _type : 'general',
              decoration: const InputDecoration(labelText: 'Model family'),
              items: [for (final e in _types.entries) DropdownMenuItem(value: e.key, child: Text(e.value))],
              onChanged: (v) => setState(() => _type = v ?? 'general'),
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Supports tool calling (agent)'),
              value: _tools,
              onChanged: (v) => setState(() => _tools = v),
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Needs my Hugging Face token'),
              value: _token,
              onChanged: (v) => setState(() => _token = v),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
        FilledButton(onPressed: _save, child: const Text('Use this model')),
      ],
    );
  }
}
