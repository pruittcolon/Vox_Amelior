import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:webview_flutter/webview_flutter.dart';

class HtmlDashboardsPage extends StatefulWidget {
  const HtmlDashboardsPage({super.key});

  @override
  State<HtmlDashboardsPage> createState() => _HtmlDashboardsPageState();
}

class HtmlResource {
  const HtmlResource({
    required this.label,
    required this.path,
    this.description,
  });

  final String label;
  final String path;
  final String? description;
}

class _HtmlDashboardsPageState extends State<HtmlDashboardsPage> {
  late final WebViewController _controller;
  late final TextEditingController _baseUrlController;
  HtmlResource? _selectedResource;
  double _progress = 0;
  String? _errorMessage;
  bool _autoLaunchExternal = true;

  static const _primaryColor = Color(0xFF0D1B2A);
  static const _cardColor = Color(0xFF1B263B);
  static const _accentColor = Color(0xFF33A1F2);

  static const String _uiPrefix = '/ui';

  static const List<HtmlResource> _resources = [
    HtmlResource(
      label: 'Login Portal',
      path: '$_uiPrefix/login.html',
      description: 'Authenticate with the Nemo stack.',
    ),
    HtmlResource(
      label: 'Control Center',
      path: '$_uiPrefix/index.html',
      description: 'Primary dashboard entry point.',
    ),
    HtmlResource(
      label: 'Gemma Chat',
      path: '$_uiPrefix/gemma.html',
      description: 'Gemma chat UI and tooling.',
    ),
    HtmlResource(
      label: 'Email Analyzer',
      path: '$_uiPrefix/email.html',
      description: 'Email triage and insights.',
    ),
    HtmlResource(
      label: 'Email Debug',
      path: '$_uiPrefix/test-email-gemma-debug.html',
      description: 'Debug view for Gemma email flows.',
    ),
    HtmlResource(
      label: 'Emotions Dashboard',
      path: '$_uiPrefix/emotions.html',
      description: 'Emotion analysis interface.',
    ),
    HtmlResource(
      label: 'Emotions Debug',
      path: '$_uiPrefix/test-emotions-debug.html',
      description: 'Focused debugging harness.',
    ),
    HtmlResource(
      label: 'Memories',
      path: '$_uiPrefix/memories.html',
      description: 'Long-term memory explorer.',
    ),
    HtmlResource(
      label: 'Speakers',
      path: '$_uiPrefix/speakers.html',
      description: 'Speaker profile tooling.',
    ),
    HtmlResource(
      label: 'Transcripts',
      path: '$_uiPrefix/transcripts.html',
      description: 'Recent transcription history.',
    ),
    HtmlResource(
      label: 'Analysis',
      path: '$_uiPrefix/analysis.html',
      description: 'Vision / multi-modal analysis view.',
    ),
    HtmlResource(
      label: 'Patterns',
      path: '$_uiPrefix/patterns.html',
    ),
    HtmlResource(
      label: 'Settings',
      path: '$_uiPrefix/settings.html',
    ),
    HtmlResource(
      label: 'Search',
      path: '$_uiPrefix/search.html',
    ),
  ];

  @override
  void initState() {
    super.initState();
    final base = _resolveBaseUrl();
    _baseUrlController = TextEditingController(text: base);
    _selectedResource = _resources.first;
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(_primaryColor)
      ..setNavigationDelegate(
        NavigationDelegate(
          onProgress: (int progress) {
            setState(() => _progress = progress / 100);
          },
          onPageStarted: (_) {
            setState(() {
              _errorMessage = null;
            });
          },
          onPageFinished: (_) {
            setState(() => _progress = 1);
          },
          onWebResourceError: (error) {
            setState(() {
              _errorMessage = 'Unable to load page (${error.errorCode})';
              _progress = 0;
            });
          },
        ),
      );
    _loadSelectedResource();
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    super.dispose();
  }

  String _resolveBaseUrl() {
    final htmlBase = dotenv.env['HTML_SERVER_BASE_URL']?.trim();
    if (htmlBase != null && htmlBase.isNotEmpty) {
      return htmlBase;
    }
    final memoryBase = dotenv.env['MEMORY_SERVER_BASE']?.trim();
    if (memoryBase != null && memoryBase.isNotEmpty) {
      return memoryBase;
    }
    return 'http://127.0.0.1:8000';
  }

  void _loadSelectedResource() {
    final url = _currentResourceUrl();
    if (url == null) {
      return;
    }

    try {
      _controller.loadRequest(Uri.parse(url));
      setState(() {
        _errorMessage = null;
        _progress = 0.05;
      });
      if (_autoLaunchExternal) {
        Future.microtask(() => _launchExternal(url, silent: true));
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Invalid URL: $url';
      });
    }
  }

  String _combineBaseAndPath(String base, String path) {
    if (path.startsWith('http')) {
      return path;
    }
    final normalizedBase =
        base.endsWith('/') ? base.substring(0, base.length - 1) : base;
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return '$normalizedBase$normalizedPath';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _primaryColor,
      appBar: AppBar(
        backgroundColor: _primaryColor,
        title: const Text(
          'HTML Dashboards',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: _buildControls(),
          ),
          if (_progress > 0 && _progress < 1)
            LinearProgressIndicator(
              minHeight: 2,
              value: _progress,
              backgroundColor: Colors.white10,
              valueColor: const AlwaysStoppedAnimation<Color>(_accentColor),
            ),
          Expanded(
            child: Stack(
              children: [
                WebViewWidget(
                  controller: _controller,
                ),
                if (_errorMessage != null)
                  Align(
                    alignment: Alignment.topCenter,
                    child: Container(
                      margin: const EdgeInsets.all(16),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.9),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.warning, color: Colors.white),
                          const SizedBox(width: 8),
                          Flexible(
                            child: Text(
                              _errorMessage!,
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                          IconButton(
                            onPressed: () =>
                                setState(() => _errorMessage = null),
                            icon: const Icon(Icons.close, color: Colors.white),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Card(
      color: _cardColor,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Server Base URL',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _baseUrlController,
              keyboardType: TextInputType.url,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                hintText: 'http://192.168.x.x:8000',
                hintStyle: const TextStyle(color: Colors.white54),
                filled: true,
                fillColor: Colors.white12,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
              onSubmitted: (_) => _loadSelectedResource(),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<HtmlResource>(
              value: _selectedResource,
              dropdownColor: _cardColor,
              iconEnabledColor: Colors.white,
              decoration: InputDecoration(
                labelText: 'HTML Page',
                labelStyle: const TextStyle(color: Colors.white70),
                filled: true,
                fillColor: Colors.white12,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              items: _resources
                  .map(
                    (resource) => DropdownMenuItem<HtmlResource>(
                      value: resource,
                      child: Text(
                        resource.label,
                        style: const TextStyle(color: Colors.white),
                      ),
                    ),
                  )
                  .toList(),
              onChanged: (value) {
                setState(() {
                  _selectedResource = value;
                });
                _loadSelectedResource();
              },
            ),
            if (_selectedResource?.description != null) ...[
              const SizedBox(height: 8),
              Text(
                _selectedResource!.description!,
                style: const TextStyle(color: Colors.white70),
              ),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.refresh),
                  label: const Text('Reload'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _accentColor,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: _loadSelectedResource,
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  icon: const Icon(Icons.open_in_browser),
                  label: const Text('Open in browser'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white12,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: _openExternally,
                ),
              ],
            ),
            SwitchListTile.adaptive(
              value: _autoLaunchExternal,
              onChanged: (value) {
                setState(() {
                  _autoLaunchExternal = value;
                });
              },
              title: const Text(
                'Also open in Android browser',
                style: TextStyle(color: Colors.white),
              ),
              subtitle: const Text(
                'Launches Chrome (or default) each time you change pages.',
                style: TextStyle(color: Colors.white54, fontSize: 12),
              ),
              contentPadding: EdgeInsets.zero,
              activeColor: _accentColor,
            ),
          ],
        ),
      ),
    );
  }

  String? _currentResourceUrl() {
    final resource = _selectedResource;
    final baseUrl = _baseUrlController.text.trim();
    if (resource == null) {
      _showSnack('Select an HTML page to load.');
      return null;
    }
    if (baseUrl.isEmpty) {
      _showSnack('Enter the server address (e.g., http://192.168.0.7:8000).');
      return null;
    }
    return _combineBaseAndPath(baseUrl, resource.path);
  }

  Future<void> _openExternally() async {
    final url = _currentResourceUrl();
    if (url == null) {
      return;
    }
    await _launchExternal(url);
  }

  Future<void> _launchExternal(String url, {bool silent = false}) async {
    final uri = Uri.tryParse(url);
    if (uri == null) {
      if (!silent) {
        _showSnack('Invalid URL: $url');
      }
      return;
    }

    try {
      final launched =
          await launchUrl(uri, mode: LaunchMode.externalApplication);
      if (!launched && !silent) {
        Clipboard.setData(ClipboardData(text: url));
        _showSnack('Could not open browser. Link copied to clipboard.');
      }
    } catch (_) {
      Clipboard.setData(ClipboardData(text: url));
      if (!silent) {
        _showSnack('Browser launch failed. Link copied for manual use.');
      }
    }
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: Colors.black87,
        content: Text(message),
      ),
    );
  }
}
