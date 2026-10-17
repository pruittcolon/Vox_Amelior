/// Turns written text into something a voice can read well.
///
/// [displayText] removes markdown so the screen shows clean prose;
/// [speechText] then expands what voices stumble over (e.g. -> for example,
/// 1905–1910 -> 1905 to 1910, URLs dropped).
library;

final _boldItalic = RegExp(r'(\*\*\*|\*\*|__)(.+?)\1');
final _italicStar = RegExp(r'(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])');
final _italicUnderscore = RegExp(r'(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])');
final _code = RegExp(r'`([^`]*)`');
final _link = RegExp(r'!?\[([^\]]*)\]\([^)]*\)');
final _footnote = RegExp(r'\[\^?\d{1,3}\]|\^\d{1,3}\b');
final _html = RegExp(r'<[^>]{1,80}>');
final _spaces = RegExp(r'\s+');

String displayText(String raw) {
  var s = raw;
  s = s.replaceAllMapped(_link, (m) => m[1]!);
  s = s.replaceAllMapped(_boldItalic, (m) => m[2]!);
  s = s.replaceAllMapped(_italicStar, (m) => m[1]!);
  s = s.replaceAllMapped(_italicUnderscore, (m) => m[1]!);
  s = s.replaceAllMapped(_code, (m) => m[1]!);
  s = s.replaceAll(_footnote, '');
  s = s.replaceAll(_html, '');
  s = s.replaceAll('­', ''); // soft hyphen
  s = s.replaceAll(_spaces, ' ');
  return s.trim();
}

final _url = RegExp(r'(https?://|www\.)\S+', caseSensitive: false);
final _email = RegExp(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b');
final _rangeDash = RegExp(r'(\d)\s*[–—-]\s*(\d)');
final _emDash = RegExp(r'\s*[—–]\s*|\s+-{1,2}\s+');
final _percent = RegExp(r'(\d)\s*%');
final _degreesC = RegExp(r'(\d)\s*°\s*C\b');
final _degreesF = RegExp(r'(\d)\s*°\s*F\b');
final _degrees = RegExp(r'(\d)\s*°');
final _thousands = RegExp(r'(\d),(\d{3})\b');
final _bracketTag = RegExp(r'\[[^\]]{0,40}\]');
final _emoji = RegExp(
  r'[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}\u{200D}]',
  unicode: true,
);
final _ellipsis = RegExp(r'\.{3,}|…');
final _repeatPunct = RegExp(r'([!?])[!?]+');
final _bullet = RegExp(r'^[•·▪◦‣]\s*');

/// Spoken forms for abbreviations; matched as whole words, case-sensitive.
const _abbreviations = <String, String>{
  'e.g.': 'for example',
  'E.g.': 'For example',
  'i.e.': 'that is',
  'I.e.': 'That is',
  'etc.': 'et cetera',
  'vs.': 'versus',
  'vs': 'versus',
  'Dr.': 'Doctor',
  'Mr.': 'Mister',
  'Mrs.': 'Missus',
  'Ms.': 'Miz',
  'Prof.': 'Professor',
  'Jr.': 'Junior',
  'Sr.': 'Senior',
  'approx.': 'approximately',
  'Fig.': 'Figure',
  'fig.': 'figure',
  'Vol.': 'Volume',
  'Ch.': 'Chapter',
  'cf.': 'compare',
  'a.m.': 'A M',
  'p.m.': 'P M',
  'w/': 'with',
  'w/o': 'without',
};

final _abbrevPattern = RegExp(
  '(?<![\\w.])(${_abbreviations.keys.map(RegExp.escape).join('|')})(?![\\w])',
);
final _numberSign = RegExp(r'\bNo\.\s*(\d)');

String speechText(String display) {
  var s = display;
  s = s.replaceAll(_url, '');
  s = s.replaceAll(_email, '');
  s = s.replaceAll(_bracketTag, ' ');
  s = s.replaceAll(_emoji, '');
  s = s.replaceAll(_bullet, '');
  s = s.replaceAllMapped(_numberSign, (m) => 'number ${m[1]}');
  s = s.replaceAllMapped(_abbrevPattern, (m) => _abbreviations[m[1]]!);
  s = s.replaceAllMapped(_thousands, (m) => '${m[1]}${m[2]}');
  s = s.replaceAllMapped(_rangeDash, (m) => '${m[1]} to ${m[2]}');
  s = s.replaceAll(_emDash, ', ');
  s = s.replaceAllMapped(_percent, (m) => '${m[1]} percent');
  s = s.replaceAllMapped(_degreesC, (m) => '${m[1]} degrees Celsius');
  s = s.replaceAllMapped(_degreesF, (m) => '${m[1]} degrees Fahrenheit');
  s = s.replaceAllMapped(_degrees, (m) => '${m[1]} degrees');
  s = s.replaceAll('&', ' and ');
  s = s.replaceAll(RegExp(r'[“”„«»]'), '"');
  s = s.replaceAll(RegExp(r'[‘’‚]'), "'");
  s = s.replaceAll(_ellipsis, '...');
  s = s.replaceAllMapped(_repeatPunct, (m) => m[1]!);
  s = s.replaceAll(RegExp(r'[*_#~|<>{}=+\\^]'), ' ');
  s = s.replaceAll(_spaces, ' ').trim();
  s = s.replaceAllMapped(RegExp(r'\s+([,.!?;:])'), (m) => m[1]!);
  s = s.replaceAll(RegExp(r',\s*,'), ',');
  s = s.replaceFirst(RegExp(r'^[,;:]\s*'), '');
  return s;
}

/// True when [text] has something a voice can actually say.
bool isSpeakable(String text) => RegExp(r'[A-Za-z0-9À-ɏ]').hasMatch(text);
