import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:vox_amelior_mobile/app/app_services.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';
import 'package:vox_amelior_mobile/models/model_catalog.dart';
import 'package:vox_amelior_mobile/ui/format.dart';

/// First-run setup: download the on-device models and (optionally) connect
/// Hugging Face for Gemma.
class SetupScreen extends StatefulWidget {
  const SetupScreen({super.key, required this.services, this.onDone});

  final AppServices services;

  /// Shown as a "Continue" button once the speech models are installed.
  final VoidCallback? onDone;

  @override
  State<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends State<SetupScreen> {
  final _token = TextEditingController();
  bool _hasToken = false;
  bool _showToken = false;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    s.tokens.huggingFace().then((t) {
      if (!mounted) return;
      setState(() => _hasToken = t != null);
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
    setState(() => _hasToken = true);
    if (mounted) showMessage(context, 'Token saved securely on this phone.');
  }

  Future<void> _clearToken() async {
    await s.tokens.clearHuggingFace();
    setState(() => _hasToken = false);
  }

  Future<void> _open(String url) async {
    if (!await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication) && mounted) {
      showMessage(context, 'Could not open $url');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Set up Vox')),
      body: ListenableBuilder(
        listenable: Listenable.merge([s.downloads, s.settings]),
        builder: (context, _) {
          final speech = ModelCatalog.all.where((m) => m.essential).toList();
          final speechBytes = speech.fold<int>(0, (a, m) => a + m.approxDownloadBytes);
          final gemma = s.settings.value.gemmaAsset;
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                'Everything runs on this phone. Nothing you say leaves it. '
                'First, download the speech models once (Wi-Fi recommended, keep Vox open until done).',
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 16),
              _Section(
                title: '1. Speech models (required)',
                subtitle: 'About ${formatBytes(speechBytes)}. Needed to listen, transcribe and recognise voices.',
                children: [
                  for (final m in speech) _ModelTile(asset: m, downloads: s.downloads),
                  const SizedBox(height: 8),
                  if (!s.downloads.speechReady)
                    FilledButton.icon(
                      onPressed: speech.any((m) => s.downloads.stateOf(m).isBusy) ? null : s.downloads.downloadSpeechModels,
                      icon: const Icon(Icons.download),
                      label: const Text('Download speech models'),
                    ),
                ],
              ),
              const SizedBox(height: 16),
              _Section(
                title: '2. Assistant (optional)',
                subtitle: 'Gemma answers questions like "What did Sam say about the plumber?". '
                    'Google requires you to accept its licence on Hugging Face first.',
                children: [
                  _Step(n: 1, text: 'Create a free Hugging Face account and open the model page. Tap "Agree and access repository".'),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: () => _open(gemma.licenseUrl ?? 'https://huggingface.co'),
                      icon: const Icon(Icons.open_in_new),
                      label: const Text('Open Gemma model page'),
                    ),
                  ),
                  const _Step(n: 2, text: 'Create an access token (type "Read") and paste it below.'),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: () => _open('https://huggingface.co/settings/tokens'),
                      icon: const Icon(Icons.key),
                      label: const Text('Open token settings'),
                    ),
                  ),
                  if (_hasToken)
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: const Icon(Icons.verified_user, color: Colors.green),
                      title: const Text('Hugging Face token saved'),
                      trailing: TextButton(onPressed: _clearToken, child: const Text('Remove')),
                    )
                  else
                    TextField(
                      controller: _token,
                      obscureText: !_showToken,
                      autocorrect: false,
                      enableSuggestions: false,
                      decoration: InputDecoration(
                        labelText: 'Hugging Face token',
                        hintText: 'hf_...',
                        border: const OutlineInputBorder(),
                        suffixIcon: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              tooltip: _showToken ? 'Hide' : 'Show',
                              icon: Icon(_showToken ? Icons.visibility_off : Icons.visibility),
                              onPressed: () => setState(() => _showToken = !_showToken),
                            ),
                            IconButton(tooltip: 'Save', icon: const Icon(Icons.save), onPressed: _saveToken),
                          ],
                        ),
                      ),
                      onSubmitted: (_) => _saveToken(),
                    ),
                  const SizedBox(height: 8),
                  SegmentedButton<String>(
                    segments: const [
                      ButtonSegment(value: 'gemma-3n-e4b-it-int4', label: Text('E4B (best)')),
                      ButtonSegment(value: 'gemma-3n-e2b-it-int4', label: Text('E2B (lighter)')),
                    ],
                    selected: {s.settings.value.gemmaModelId},
                    onSelectionChanged: (v) => s.updateSettings(s.settings.value.copyWith(gemmaModelId: v.first)),
                  ),
                  _ModelTile(asset: gemma, downloads: s.downloads, tokenSaved: _hasToken),
                ],
              ),
              const SizedBox(height: 24),
              if (widget.onDone != null)
                FilledButton(
                  onPressed: s.downloads.speechReady ? widget.onDone : null,
                  child: Text(s.downloads.speechReady ? 'Continue' : 'Download the speech models to continue'),
                ),
            ],
          );
        },
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.subtitle, required this.children});

  final String title;
  final String subtitle;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(subtitle, style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              ...children,
            ],
          ),
        ),
      );
}

class _Step extends StatelessWidget {
  const _Step({required this.n, required this.text});

  final int n;
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CircleAvatar(radius: 10, child: Text('$n', style: const TextStyle(fontSize: 12))),
            const SizedBox(width: 8),
            Expanded(child: Text(text)),
          ],
        ),
      );
}

class _ModelTile extends StatelessWidget {
  const _ModelTile({required this.asset, required this.downloads, this.tokenSaved = true});

  final ModelAsset asset;
  final ModelDownloads downloads;
  final bool tokenSaved;

  @override
  Widget build(BuildContext context) {
    final st = downloads.stateOf(asset);
    final theme = Theme.of(context);
    Widget status;
    Widget? action;
    switch (st.status) {
      case DownloadStatus.installed:
        status = const Text('Installed', style: TextStyle(color: Colors.green));
        action = IconButton(
          tooltip: 'Delete to free space',
          icon: const Icon(Icons.delete_outline),
          onPressed: () async {
            if (await confirm(context, 'Delete ${asset.title}?', 'You can download it again later.')) {
              downloads.remove(asset);
            }
          },
        );
      case DownloadStatus.downloading:
      case DownloadStatus.unpacking:
        final unpacking = st.status == DownloadStatus.unpacking;
        status = Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            LinearProgressIndicator(value: unpacking ? null : st.progress),
            const SizedBox(height: 4),
            Text(unpacking
                ? 'Unpacking… (about a minute)'
                : '${formatBytes(st.received)} of ${formatBytes(st.total)}'),
          ],
        );
        action = unpacking
            ? null
            : IconButton(tooltip: 'Pause', icon: const Icon(Icons.pause), onPressed: () => downloads.cancel(asset));
      case DownloadStatus.failed:
        status = Text(st.error ?? 'Failed', style: TextStyle(color: theme.colorScheme.error));
        action = IconButton(tooltip: 'Retry', icon: const Icon(Icons.refresh), onPressed: () => downloads.download(asset));
      case DownloadStatus.notInstalled:
        final resumable = st.received > 0;
        status = Text(resumable
            ? 'Paused at ${formatBytes(st.received)} of ${formatBytes(asset.approxDownloadBytes)}'
            : '${formatBytes(asset.approxDownloadBytes)} download');
        final blocked = asset.requiresToken && !tokenSaved;
        action = IconButton(
          tooltip: blocked ? 'Save a Hugging Face token first' : (resumable ? 'Resume' : 'Download'),
          icon: Icon(resumable ? Icons.play_arrow : Icons.download),
          onPressed: blocked ? null : () => downloads.download(asset),
        );
    }
    return ListTile(
      contentPadding: EdgeInsets.zero,
      title: Text(asset.title),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [Text(asset.description), const SizedBox(height: 4), status],
      ),
      isThreeLine: true,
      trailing: action,
    );
  }
}
