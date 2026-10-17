import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../core/library/library.dart';
import '../core/script/script_parser.dart';
import 'reader_screen.dart';
import 'scope.dart';

/// Paste a new script (e.g. Claude's reply) or edit a document's text.
class EditTextScreen extends StatefulWidget {
  const EditTextScreen({super.key, this.doc});

  /// Null: a new document.
  final Doc? doc;

  @override
  State<EditTextScreen> createState() => _EditTextScreenState();
}

class _EditTextScreenState extends State<EditTextScreen> {
  final _text = TextEditingController();
  final _title = TextEditingController();
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final doc = widget.doc;
    if (doc != null) {
      _title.text = doc.title;
      _text.text = await context.app.library.readText(doc);
    } else {
      final clip = await Clipboard.getData(Clipboard.kTextPlain);
      final pasted = clip?.text?.trim() ?? '';
      if (pasted.length > 40) {
        _text.text = pasted;
        _title.text = parseScript(pasted, fallbackTitle: '').title;
      }
    }
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _paste() async {
    final clip = await Clipboard.getData(Clipboard.kTextPlain);
    final pasted = clip?.text ?? '';
    if (pasted.isEmpty) {
      if (mounted) context.toast('The clipboard is empty. Copy Claude\'s reply first.');
      return;
    }
    setState(() {
      _text.text = pasted;
      if (_title.text.trim().isEmpty) _title.text = parseScript(pasted, fallbackTitle: '').title;
    });
  }

  Future<void> _save() async {
    final text = _text.text;
    if (text.trim().isEmpty) {
      context.toast('Nothing to save yet.');
      return;
    }
    final lib = context.app.library;
    final nav = Navigator.of(context);
    final title = _title.text.trim().isEmpty ? parseScript(text, fallbackTitle: 'Untitled').title : _title.text.trim();
    final doc = widget.doc;
    if (doc != null) {
      doc.title = title;
      await lib.updateText(doc, text);
      nav.pop();
    } else {
      final created = await lib.add(title: title, kind: 'pasted', text: text);
      await nav.pushReplacement(MaterialPageRoute<void>(builder: (_) => ReaderScreen(doc: created)));
    }
  }

  @override
  void dispose() {
    _text.dispose();
    _title.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.doc == null ? 'New script' : 'Edit text'),
        actions: [
          if (widget.doc == null)
            IconButton(tooltip: 'Paste', icon: const Icon(Icons.content_paste), onPressed: _paste),
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: FilledButton(onPressed: _loading ? null : _save, child: const Text('Save')),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
              child: Column(
                children: [
                  TextField(
                    controller: _title,
                    decoration: const InputDecoration(labelText: 'Title'),
                    textCapitalization: TextCapitalization.sentences,
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: TextField(
                      controller: _text,
                      expands: true,
                      maxLines: null,
                      textAlignVertical: TextAlignVertical.top,
                      keyboardType: TextInputType.multiline,
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 14, height: 1.4),
                      decoration: const InputDecoration(
                        hintText: 'Paste a script from Claude, or any text.\n\n'
                            'HOST: Welcome to the show!\n'
                            'GUEST: Thanks for having me.\n\n'
                            '## Part 1\n'
                            '[pause 2]',
                        alignLabelWithHint: true,
                      ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}
