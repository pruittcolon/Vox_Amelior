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

  /// Big downloads are confirmed first; small ones just start.
  Future<bool> _okToDownload(ModelAsset m) async {
    if (s.models.isInstalled(m) || m.approxDownloadBytes < 1000000000) return true;
    return confirm(
      context,
      'Download ${formatBytes(m.approxDownloadBytes)}?',
      '${m.title} is large. Wi-Fi recommended. What you use now keeps working until it is ready.',
      action: 'Download',
    );
  }

  Future<void> _deleteModel(ModelAsset m) async {
    if (!await confirm(context, 'Delete ${m.title}?', 'Frees ${formatBytes(m.approxDownloadBytes)}. You can download it again later.')) return;
    if (m.kind == ModelKind.speechToText) {
      await s.removeSpeechModel(m);
    } else {
      s.downloads.remove(m);
    }
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
          final installed = s.models.isInstalled;
          // What runs now: the chosen model once it is installed, until then another installed one.
          final asr = installed(st.asrAsset) ? st.asrAsset : ModelCatalog.recognizers.where(installed).firstOrNull;
          final search = s.searchModel;
          final custom = st.customLlmUrl.trim().isEmpty
              ? null
              : ModelCatalog.custom(
                  url: st.customLlmUrl,
                  name: st.customLlmName,
                  llmType: st.customLlmType,
                  supportsTools: st.customLlmTools,
                  requiresToken: st.customLlmNeedsToken,
                );
          return ListView(
            key: const ValueKey('models-list'),
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
            children: [
              Text(
                'Everything runs on this phone and nothing you say leaves it. Tap a choice to switch: a model that is '
                'not here yet downloads first (in the background, resuming if interrupted) and takes over when it is ready.',
                style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
              ),
              if (!s.downloads.speechReady) ...[
                const SizedBox(height: 16),
                FilledButton.icon(
                  onPressed: speechBusy ? null : s.downloads.downloadSpeechModels,
                  icon: const Icon(Icons.download_rounded),
                  label: Text('Download speech models (${formatBytes(speechBytes)})'),
                ),
              ],
              _section(context, 'Hearing speech', Icons.hearing_rounded, 'Turns speech into text. Larger models are a little more accurate and slower.', [
                for (final (m, label) in [
                  (ModelCatalog.parakeetFp16, 'Standard · fp16'),
                  (ModelCatalog.parakeet, 'Small · int8'),
                  (ModelCatalog.parakeetFp32, 'Full precision · fp32'),
                ])
                  _ChoiceRow(
                    asset: m,
                    label: label,
                    downloads: s.downloads,
                    chosen: st.asrAsset.id == m.id,
                    inUse: asr?.id == m.id,
                    onChoose: () async {
                      if (!await _okToDownload(m)) return;
                      await s.selectSpeechModel(ModelCatalog.recognizerName(m));
                      // At setup the detector and voiceprints come too.
                      for (final support in ModelCatalog.speechSupport) {
                        await s.downloads.download(support);
                      }
                    },
                    onDelete: asr?.id == m.id ? null : () => _deleteModel(m),
                  ),
                const Divider(indent: 16, endIndent: 16),
                for (final m in [ModelCatalog.voiceActivity, ModelCatalog.speakerVoiceprint, ModelCatalog.diarizer])
                  _SupportRow(asset: m, downloads: s.downloads),
              ]),
              _section(context, 'Tone of voice', Icons.mood_rounded, 'Hears how each line was said (happy, angry…) and sounds like laughter.', [
                _OffRow(label: 'Off', chosen: !st.hearTone, onChoose: () => s.updateSettings(st.copyWith(hearTone: false))),
                _ChoiceRow(
                  asset: ModelCatalog.toneModel,
                  label: 'On · SenseVoice Small',
                  downloads: s.downloads,
                  chosen: st.hearTone,
                  inUse: st.hearTone && installed(ModelCatalog.toneModel),
                  onChoose: () async {
                    await s.updateSettings(st.copyWith(hearTone: true));
                    await s.downloads.download(ModelCatalog.toneModel);
                  },
                  onDelete: st.hearTone ? null : () => _deleteModel(ModelCatalog.toneModel),
                ),
              ]),
              _section(context, 'Search by meaning', Icons.manage_search_rounded,
                  'Finds what was said in other words, and picks what Gemma reads to answer. Each size prepares every line once.', [
                _OffRow(label: 'Off · words only', chosen: !st.meaningSearch, onChoose: () => s.updateSettings(st.copyWith(meaningSearch: false))),
                for (final m in [ModelCatalog.textEmbedderSmall, ModelCatalog.textEmbedder, ModelCatalog.textEmbedderFull])
                  _ChoiceRow(
                    asset: m,
                    label: ModelCatalog.embedderLabel(m),
                    downloads: s.downloads,
                    chosen: st.meaningSearch && st.searchAsset.id == m.id,
                    inUse: st.meaningSearch && search?.id == m.id,
                    onChoose: () async {
                      if (await _okToDownload(m)) await s.selectSearchModel(ModelCatalog.embedderName(m));
                    },
                    onDelete: st.meaningSearch && search?.id == m.id ? null : () => _deleteModel(m),
                  ),
              ]),
              _section(context, 'Assistant', Icons.auto_awesome_rounded, 'Gemma answers questions and writes reviews.', [
                for (final m in [...ModelCatalog.assistants, ?custom])
                  _ChoiceRow(
                    asset: m,
                    label: m.title,
                    downloads: s.downloads,
                    chosen: st.llmId == m.id,
                    inUse: st.llmId == m.id && installed(m),
                    onChoose: () async {
                      if (!await _okToDownload(m)) return;
                      await s.updateSettings(st.copyWith(llmId: m.id));
                      await s.downloads.download(m);
                    },
                    onDelete: st.llmId == m.id ? null : () => _deleteModel(m),
                    onEdit: m.id == ModelCatalog.customLlmId ? _editCustom : null,
                  ),
                ListTile(
                  leading: const Icon(Icons.add_link_rounded),
                  title: Text(custom == null ? 'Use another model…' : 'Edit custom model…'),
                  subtitle: const Text('Any LiteRT-LM (.litertlm) model URL'),
                  onTap: _editCustom,
                ),
              ]),
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
                Text('You can leave this screen; downloads keep going.', textAlign: TextAlign.center, style: t.textTheme.bodySmall),
              ],
            ],
          );
        },
      ),
    );
  }

  /// One job (hearing, tone, search, assistant) and its choices.
  Widget _section(BuildContext context, String title, IconData icon, String about, List<Widget> rows) {
    final t = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 20),
      child: VoxCard(
        padding: const EdgeInsets.fromLTRB(0, 14, 0, 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Icon(icon, color: t.colorScheme.primary),
                  const SizedBox(width: 10),
                  Expanded(child: Text(title, style: t.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800))),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 6),
              child: Text(about, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant)),
            ),
            ...rows,
          ],
        ),
      ),
    );
  }
}

/// A radio-style choice without a model ("Off").
class _OffRow extends StatelessWidget {
  const _OffRow({required this.label, required this.chosen, required this.onChoose});

  final String label;
  final bool chosen;
  final VoidCallback onChoose;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return ListTile(
      onTap: chosen ? null : onChoose,
      leading: Icon(chosen ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded, color: chosen ? t.colorScheme.primary : null),
      title: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
    );
  }
}

/// One model to choose: tap to use it (downloading it first if needed).
/// Shows whether it is in use, waiting, downloading or here, and its size.
class _ChoiceRow extends StatelessWidget {
  const _ChoiceRow({
    required this.asset,
    required this.label,
    required this.downloads,
    required this.chosen,
    required this.inUse,
    required this.onChoose,
    this.onDelete,
    this.onEdit,
  });

  final ModelAsset asset;
  final String label;
  final ModelDownloads downloads;

  /// Picked in settings (it may still be downloading).
  final bool chosen;

  /// Running now.
  final bool inUse;
  final Future<void> Function() onChoose;
  final VoidCallback? onDelete;
  final VoidCallback? onEdit;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final st = downloads.stateOf(asset);
    final size = asset.approxDownloadBytes > 0 ? formatBytes(asset.approxDownloadBytes) : 'size unknown';
    const green = Color(0xFF0E9F6E);
    final Widget status = switch (st.status) {
      DownloadStatus.installed => inUse
          ? const Pill('In use', icon: Icons.check_circle_rounded, color: green)
          : Pill(chosen ? 'Starting…' : 'Downloaded · $size', icon: Icons.download_done_rounded, color: t.colorScheme.onSurfaceVariant),
      DownloadStatus.queued => const Pill('Waiting to download', icon: Icons.schedule_rounded),
      DownloadStatus.downloading || DownloadStatus.unpacking => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(borderRadius: BorderRadius.circular(8), child: LinearProgressIndicator(value: st.progress, minHeight: 6)),
            const SizedBox(height: 4),
            Text(
              '${describeDownload(st)}${chosen ? ' · switches when ready' : ''}',
              style: t.textTheme.bodySmall,
            ),
          ],
        ),
      DownloadStatus.failed => Text(st.error ?? 'Download failed', maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(color: t.colorScheme.error)),
      DownloadStatus.notInstalled => Text(
          st.received > 0 ? 'Paused at ${formatBytes(st.received)} of $size' : size,
          style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.onSurfaceVariant),
        ),
    };
    final busy = st.status == DownloadStatus.downloading || st.status == DownloadStatus.queued;
    return ListTile(
      onTap: chosen && st.status != DownloadStatus.failed && st.status != DownloadStatus.notInstalled ? null : onChoose,
      leading: Icon(chosen ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded, color: chosen ? t.colorScheme.primary : null),
      title: Row(
        children: [
          Flexible(child: Text(label, style: const TextStyle(fontWeight: FontWeight.w600))),
          if (asset.supportsTools) ...[const SizedBox(width: 6), const Pill('Agent', icon: Icons.bolt_rounded, color: Color(0xFF9C36B5))],
        ],
      ),
      subtitle: Padding(
        padding: const EdgeInsets.only(top: 2),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(asset.description, maxLines: 2, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 6),
            status,
          ],
        ),
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (onEdit != null) IconButton(tooltip: 'Edit', icon: const Icon(Icons.edit_rounded), onPressed: onEdit),
          if (busy)
            IconButton(tooltip: 'Pause', icon: const Icon(Icons.pause_rounded), onPressed: () => downloads.cancel(asset))
          else if (st.status == DownloadStatus.failed)
            IconButton(tooltip: 'Retry', icon: const Icon(Icons.refresh_rounded), onPressed: () => downloads.download(asset))
          else if (st.status == DownloadStatus.installed && onDelete != null)
            IconButton(tooltip: 'Delete to free space', icon: const Icon(Icons.delete_outline_rounded), onPressed: onDelete),
        ],
      ),
    );
  }
}

/// A model that works alongside the speech recognizer (no choice to make).
class _SupportRow extends StatelessWidget {
  const _SupportRow({required this.asset, required this.downloads});

  final ModelAsset asset;
  final ModelDownloads downloads;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final st = downloads.stateOf(asset);
    final (IconData icon, String text, Color? color) = switch (st.status) {
      DownloadStatus.installed => (Icons.check_circle_rounded, 'Ready', const Color(0xFF0E9F6E)),
      DownloadStatus.queued => (Icons.schedule_rounded, 'Waiting to download', null),
      DownloadStatus.downloading || DownloadStatus.unpacking => (Icons.downloading_rounded, describeDownload(st), null),
      DownloadStatus.failed => (Icons.error_outline_rounded, st.error ?? 'Download failed', t.colorScheme.error),
      DownloadStatus.notInstalled => (Icons.download_rounded, formatBytes(asset.approxDownloadBytes), null),
    };
    return ListTile(
      dense: true,
      leading: Icon(switch (asset.kind) {
        ModelKind.voiceActivity => Icons.graphic_eq_rounded,
        ModelKind.speakerVoiceprint => Icons.fingerprint_rounded,
        _ => Icons.forum_rounded,
      }),
      title: Text(asset.title, style: const TextStyle(fontWeight: FontWeight.w600)),
      subtitle: Row(
        children: [
          Icon(icon, size: 14, color: color ?? t.colorScheme.onSurfaceVariant),
          const SizedBox(width: 4),
          Flexible(child: Text(text, maxLines: 1, overflow: TextOverflow.ellipsis)),
        ],
      ),
      trailing: st.status == DownloadStatus.notInstalled || st.status == DownloadStatus.failed
          ? IconButton(tooltip: 'Download', icon: const Icon(Icons.download_rounded), onPressed: () => downloads.download(asset))
          : null,
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
