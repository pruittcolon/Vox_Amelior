import 'dart:convert';

final RegExp _placeholder = RegExp(r'\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\|\s*([a-z]+)\s*)?\}\}');

/// Fills `{{name}}` placeholders from [values].
///
/// Filters: `{{text|json}}` escapes for use inside a JSON string,
/// `{{text|url}}` percent-encodes. Unknown names render as empty text.
String renderTemplate(String template, Map<String, String> values) {
  return template.replaceAllMapped(_placeholder, (m) {
    final value = values[m.group(1)!] ?? '';
    switch (m.group(2)) {
      case 'json':
        final encoded = jsonEncode(value);
        return encoded.substring(1, encoded.length - 1);
      case 'url':
        return Uri.encodeComponent(value);
      default:
        return value;
    }
  });
}
