import 'package:flutter/material.dart';
import 'package:vox_amelior_mobile/app/model_downloads.dart';

String formatBytes(int bytes) {
  if (bytes >= 1000000000) return '${(bytes / 1e9).toStringAsFixed(1)} GB';
  if (bytes >= 1000000) return '${(bytes / 1e6).toStringAsFixed(0)} MB';
  if (bytes >= 1000) return '${(bytes / 1e3).toStringAsFixed(0)} KB';
  return '$bytes B';
}

String two(int n) => n.toString().padLeft(2, '0');

String formatTime(DateTime d) => '${two(d.hour)}:${two(d.minute)}';

const List<String> _weekdayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

/// "Today", "Yesterday", "Monday", or "3 Sep 2026" for older days.
String formatDayName(DateTime d, {DateTime? now}) {
  final today = now ?? DateTime.now();
  final diff = DateTime(today.year, today.month, today.day).difference(DateTime(d.year, d.month, d.day)).inDays;
  if (diff >= 2 && diff < 7) return _weekdayNames[d.weekday - 1];
  return formatDay(d, now: now);
}

/// 1,284 · 12.9K · 3.4M
String formatCount(int n) {
  if (n >= 1000000) return '${(n / 1e6).toStringAsFixed(1)}M';
  if (n >= 10000) return '${(n / 1e3).toStringAsFixed(1)}K';
  final s = n.toString();
  return s.length > 3 ? '${s.substring(0, s.length - 3)},${s.substring(s.length - 3)}' : s;
}

/// Short talk time: "45 s", "12 min", "3 h 20 min", "41 h".
String formatTalk(Duration d) {
  if (d.inMinutes < 1) return '${d.inSeconds} s';
  if (d.inHours < 1) return '${d.inMinutes} min';
  if (d.inHours >= 10) return '${d.inHours} h';
  final m = d.inMinutes % 60;
  return m == 0 ? '${d.inHours} h' : '${d.inHours} h $m min';
}

/// For use mid-sentence: "today 20:31", "yesterday 08:02", "Monday 19:40",
/// "3 Sep 2026 19:40".
String formatWhen(DateTime d, {DateTime? now}) {
  final day = formatDayName(d, now: now);
  return '${day == 'Today' || day == 'Yesterday' ? day.toLowerCase() : day} ${formatTime(d)}';
}

String formatDuration(Duration d) {
  if (d.inMinutes < 1) return '<1 min';
  if (d.inHours < 1) return '${d.inMinutes} min';
  final m = d.inMinutes % 60;
  return m == 0 ? '${d.inHours} h' : '${d.inHours} h $m min';
}

String formatDay(DateTime d, {DateTime? now}) {
  final today = now ?? DateTime.now();
  final a = DateTime(d.year, d.month, d.day);
  final b = DateTime(today.year, today.month, today.day);
  final diff = b.difference(a).inDays;
  if (diff == 0) return 'Today';
  if (diff == 1) return 'Yesterday';
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return '${d.day} ${months[d.month - 1]} ${d.year}';
}

/// Stable colour per speaker label so people are easy to tell apart.
Color speakerColor(String label, {bool known = true}) {
  if (!known) return const Color(0xFF868E96);
  const palette = [Color(0xFF0E9F6E), Color(0xFF4F46E5), Color(0xFFE8590C), Color(0xFF9C36B5), Color(0xFF1C7ED6), Color(0xFFD6336C), Color(0xFF087F5B), Color(0xFFB08800)];
  var h = 0;
  for (final c in label.codeUnits) {
    h = (h * 31 + c) & 0x7fffffff;
  }
  return palette[h % palette.length];
}

void showMessage(BuildContext context, String text) {
  ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(SnackBar(content: Text(text)));
}

Future<bool> confirm(BuildContext context, String title, String body, {String action = 'Delete'}) async {
  final ok = await showDialog<bool>(
    context: context,
    builder: (c) => AlertDialog(
      title: Text(title),
      content: Text(body),
      actions: [
        TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(c, true), child: Text(action)),
      ],
    ),
  );
  return ok ?? false;
}

Future<String?> askText(BuildContext context, String title, {String initial = '', String hint = ''}) {
  final controller = TextEditingController(text: initial);
  return showDialog<String>(
    context: context,
    builder: (c) => AlertDialog(
      title: Text(title),
      content: TextField(
        controller: controller,
        autofocus: true,
        decoration: InputDecoration(hintText: hint),
        textCapitalization: TextCapitalization.words,
        onSubmitted: (v) => Navigator.pop(c, v.trim()),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(c), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(c, controller.text.trim()), child: const Text('OK')),
      ],
    ),
  );
}

/// One line describing a download in progress, e.g. "320 MB of 1.1 GB · 29% · about 4 min left"
/// or "Step 3 of 4: Unpacking the model · 45% · about 2 min left".
String describeDownload(DownloadState st) {
  final parts = <String>[
    if (st.status == DownloadStatus.queued)
      'Waiting to download'
    else if (st.status == DownloadStatus.unpacking)
      st.step > 0 && st.steps > 0 ? 'Step ${st.step} of ${st.steps}: ${st.stage ?? 'Finishing'}' : '${st.stage ?? 'Finishing'}…'
    else
      '${formatBytes(st.received)} of ${formatBytes(st.total)}',
    if (st.status != DownloadStatus.queued && st.progress != null) '${(st.progress! * 100).toStringAsFixed(0)}%',
    if (st.remaining != null) 'about ${formatDuration(st.remaining!)} left',
  ];
  return parts.join(' · ');
}
