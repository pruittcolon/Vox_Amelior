/// Reads .txt / .md / .pdf files into plain text the app can narrate.
library;

import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:pdfrx/pdfrx.dart';

import '../script/pdf_cleanup.dart';

class Imported {
  const Imported(this.title, this.kind, this.text);
  final String title;
  final String kind;
  final String text;
}

class ImportError implements Exception {
  ImportError(this.message);
  final String message;
  @override
  String toString() => message;
}

const supportedExtensions = ['txt', 'md', 'markdown', 'text', 'pdf'];

Future<Imported> importFile(String path, {String? displayName}) async {
  final name = displayName ?? p.basename(path);
  final ext = p.extension(name).toLowerCase().replaceFirst('.', '');
  final title = _titleFrom(name);
  final file = File(path);
  if (!file.existsSync()) throw ImportError('Could not open $name.');
  if (ext == 'pdf' || await _looksLikePdf(file)) {
    return Imported(title, 'pdf', await readPdfText(path));
  }
  final bytes = await file.readAsBytes();
  if (bytes.length > 20 * 1024 * 1024) throw ImportError('$name is over 20 MB of text. Split it into parts.');
  return Imported(title, ext == 'md' || ext == 'markdown' ? 'md' : 'txt', decodeText(bytes));
}

/// UTF-8 (with or without BOM), UTF-16 with BOM, else Latin-1.
String decodeText(List<int> bytes) {
  if (bytes.length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE) {
    return String.fromCharCodes(_utf16(bytes.sublist(2), little: true));
  }
  if (bytes.length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF) {
    return String.fromCharCodes(_utf16(bytes.sublist(2), little: false));
  }
  try {
    final s = utf8.decode(bytes);
    return s.startsWith('﻿') ? s.substring(1) : s;
  } on FormatException {
    return latin1.decode(bytes);
  }
}

List<int> _utf16(List<int> b, {required bool little}) => [
      for (var i = 0; i + 1 < b.length; i += 2) little ? b[i] | b[i + 1] << 8 : b[i] << 8 | b[i + 1],
    ];

Future<bool> _looksLikePdf(File f) async {
  final raf = await f.open();
  try {
    final head = await raf.read(5);
    return String.fromCharCodes(head) == '%PDF-';
  } finally {
    await raf.close();
  }
}

Future<String> readPdfText(String path) async {
  await pdfrxFlutterInitialize();
  final PdfDocument doc;
  try {
    doc = await PdfDocument.openFile(path);
  } catch (e) {
    throw ImportError('This PDF could not be opened (it may be damaged or password protected).');
  }
  try {
    final pages = <String>[];
    for (final page in doc.pages) {
      final text = await page.loadStructuredText();
      pages.add(text.fullText);
    }
    final cleaned = cleanPdfPages(pages);
    if (cleaned.replaceAll(RegExp(r'\s'), '').length < 20) {
      throw ImportError(
        'This PDF has no readable text (it is probably scanned images). '
        'Ask Claude to transcribe it, then paste the text in.',
      );
    }
    return cleaned;
  } finally {
    await doc.dispose();
  }
}

String _titleFrom(String fileName) {
  var t = p.basenameWithoutExtension(fileName);
  t = t.replaceFirst(RegExp(r'^\d{10,}-'), ''); // our shared-file prefix
  t = t.replaceAll(RegExp(r'[_]+'), ' ').replaceAll(RegExp(r'\s+'), ' ').trim();
  return t.isEmpty ? 'Untitled' : t;
}
