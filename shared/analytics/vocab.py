import collections
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TokenStats = Dict[str, Any]
SegmentRow = Dict[str, Any]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "at",
    "for",
    "to",
    "of",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
    "this",
    "that",
    "these",
    "those",
}


def _tokenize(text: str) -> List[str]:
    """Lightweight word tokenizer for transcripts."""
    if not text:
        return []
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _estimate_sentences(text: str) -> int:
    """Very rough sentence count based on punctuation."""
    if not text:
        return 0
    parts = re.split(r"[.!?]+", text)
    count = sum(1 for p in parts if p.strip())
    return count or 1


def empty_vocab_summary(
    *,
    fallback_reason: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an empty but structurally complete vocab payload."""
    payload: Dict[str, Any] = {
        "global": {
            "total_segments": 0,
            "total_words": 0,
            "unique_words": 0,
            "type_token_ratio": 0.0,
            "avg_words_per_segment": 0.0,
            "avg_words_per_sentence": 0.0,
            "days_covered": 0,
        },
        "per_speaker": {},
        "timeline": {
            "dates": [],
            "unique_words": [],
            "word_counts": [],
        },
        "top_words": {
            "global": [],
            "per_speaker": {},
        },
    }
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    if error:
        payload["error"] = error
    return payload


def _detect_record_date_column(conn: sqlite3.Connection) -> Optional[str]:
    """Mirror AnalyticsEngine date-column detection for transcript_records."""
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(transcript_records)")
        rows = cur.fetchall()
    except sqlite3.Error:
        return None
    columns = set()
    for row in rows:
        try:
            # sqlite3.Row or dict
            columns.add(row["name"])
        except Exception:
            columns.add(row[1])
    if "created_at" in columns:
        return "created_at"
    if "timestamp" in columns:
        return "timestamp"
    return None


def fetch_segments_for_vocab(
    conn: sqlite3.Connection,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    speakers: Optional[Sequence[str]] = None,
    max_rows: int = 10000,
) -> List[SegmentRow]:
    """
    Fetch minimal segment data for vocabulary analysis.

    Returns rows with: text, speaker, day (YYYY-MM-DD).
    """
    date_column = _detect_record_date_column(conn)
    where_clauses: List[str] = ["ts.text IS NOT NULL", "ts.text != ''"]
    params: List[Any] = []

    if date_column:
        if start_date:
            where_clauses.append(f"tr.{date_column} >= ?")
            params.append(f"{start_date}T00:00:00")
        if end_date:
            where_clauses.append(f"tr.{date_column} <= ?")
            params.append(f"{end_date}T23:59:59")

    if speakers:
        normalized = [s.strip().lower() for s in speakers if s and str(s).strip()]
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            where_clauses.append(f"LOWER(ts.speaker) IN ({placeholders})")
            params.extend(normalized)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    day_expr = f"date(tr.{date_column})" if date_column else "NULL"

    sql = (
        "SELECT ts.text AS text, ts.speaker AS speaker, "
        f"{day_expr} AS day "
        "FROM transcript_segments ts "
        "JOIN transcript_records tr ON ts.transcript_id = tr.id "
        f"WHERE {where_sql} "
        "ORDER BY tr.id DESC, ts.seq ASC "
        "LIMIT ?"
    )
    params.append(int(max_rows))

    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    rows: List[SegmentRow] = []
    for row in cur.fetchall():
        if isinstance(row, sqlite3.Row):
            rows.append(
                {
                    "text": row["text"],
                    "speaker": row["speaker"],
                    "day": row["day"],
                }
            )
        else:
            # Fallback: positional tuple
            text, speaker, day = row
            rows.append({"text": text, "speaker": speaker, "day": day})
    return rows


def compute_vocab_metrics(segments: Iterable[SegmentRow]) -> Dict[str, Any]:
    """Compute global, per-speaker, timeline, and top-word stats."""
    global_word_count = 0
    global_sentence_count = 0
    global_segments = 0
    global_vocab: set[str] = set()

    per_speaker_counts: Dict[str, int] = collections.defaultdict(int)
    per_speaker_segments: Dict[str, int] = collections.defaultdict(int)
    per_speaker_sentences: Dict[str, int] = collections.defaultdict(int)
    per_speaker_vocab: Dict[str, set[str]] = collections.defaultdict(set)

    per_speaker_tokens: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    global_tokens = collections.Counter()

    # Timeline: day -> {words, unique}
    day_word_counts: Dict[str, int] = collections.defaultdict(int)
    day_vocab: Dict[str, set[str]] = collections.defaultdict(set)

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = (seg.get("speaker") or "Unknown").strip() or "Unknown"
        day = seg.get("day") or None

        tokens = _tokenize(text)
        if not tokens:
            continue

        sentence_count = _estimate_sentences(text)

        token_set = set(tokens)
        word_count = len(tokens)

        global_segments += 1
        global_word_count += word_count
        global_sentence_count += sentence_count
        global_vocab.update(token_set)
        global_tokens.update(tokens)

        per_speaker_segments[speaker] += 1
        per_speaker_counts[speaker] += word_count
        per_speaker_sentences[speaker] += sentence_count
        per_speaker_vocab[speaker].update(token_set)
        per_speaker_tokens[speaker].update(tokens)

        if day:
            day_word_counts[day] += word_count
            day_vocab[day].update(token_set)

    if not global_segments:
        return empty_vocab_summary(fallback_reason="no_segments")

    # Global stats
    global_unique = len(global_vocab)
    type_token_ratio = float(global_unique) / float(global_word_count) if global_word_count else 0.0
    avg_words_per_segment = float(global_word_count) / float(global_segments) if global_segments else 0.0
    avg_words_per_sentence = (
        float(global_word_count) / float(global_sentence_count) if global_sentence_count else 0.0
    )

    # Per-speaker stats
    per_speaker: Dict[str, Any] = {}
    for speaker, words in per_speaker_counts.items():
        unique = len(per_speaker_vocab[speaker])
        segments = per_speaker_segments[speaker] or 1
        sentences = per_speaker_sentences[speaker] or segments
        per_speaker[speaker] = {
            "total_segments": segments,
            "total_words": words,
            "unique_words": unique,
            "type_token_ratio": float(unique) / float(words) if words else 0.0,
            "avg_words_per_segment": float(words) / float(segments) if segments else 0.0,
            "avg_words_per_sentence": float(words) / float(sentences) if sentences else 0.0,
            "share_of_global_words": float(words) / float(global_word_count) if global_word_count else 0.0,
        }

    # Timeline arrays sorted by date
    dates_sorted = sorted(day_word_counts.keys())
    timeline_unique = [len(day_vocab[d]) for d in dates_sorted]
    timeline_counts = [day_word_counts[d] for d in dates_sorted]

    # Notable shifts in unique vocabulary (simple top-k deltas)
    notable_shifts: List[Dict[str, Any]] = []
    if len(dates_sorted) > 1:
        deltas: List[Tuple[str, int]] = []
        prev_unique = timeline_unique[0]
        for idx in range(1, len(dates_sorted)):
            current_unique = timeline_unique[idx]
            delta = current_unique - prev_unique
            if delta != 0:
                deltas.append((dates_sorted[idx], delta))
            prev_unique = current_unique
        # Pick top positive and negative changes by absolute magnitude
        deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)
        for date, delta in deltas_sorted[:5]:
            notable_shifts.append(
                {
                    "date": date,
                    "delta_unique": int(delta),
                    "direction": "up" if delta > 0 else "down",
                }
            )

    # Top words
    def _top(counter: collections.Counter, n: int = 30) -> List[Dict[str, Any]]:
        return [{"word": w, "count": int(c)} for w, c in counter.most_common(n)]

    top_global = _top(global_tokens, 40)
    top_per_speaker: Dict[str, List[Dict[str, Any]]] = {
        speaker: _top(counter, 30) for speaker, counter in per_speaker_tokens.items()
    }

    # Speaker lexical fingerprints (distinctive words per speaker)
    distinctive_per_speaker: Dict[str, List[Dict[str, Any]]] = {}
    if global_word_count:
        for speaker, counter in per_speaker_tokens.items():
            total_words_speaker = per_speaker_counts.get(speaker) or 1
            scored: List[Tuple[str, float, int]] = []
            for word, count in counter.items():
                global_count = global_tokens.get(word) or 1
                # Relative frequency in speaker vs global
                p_speaker = float(count) / float(total_words_speaker)
                p_global = float(global_count) / float(global_word_count)
                if p_global <= 0.0:
                    continue
                score = p_speaker / p_global
                # Skip extremely rare words that only appear once globally
                if global_count < 2:
                    continue
                scored.append((word, score, count))
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            distinctive_per_speaker[speaker] = [
                {"word": w, "score": float(s), "count": int(c)} for w, s, c in scored[:10]
            ]

    # Speaker highlights for global section
    most_talkative_speaker = None
    richest_speaker = None
    if per_speaker:
        most_talkative_speaker = max(
            per_speaker.items(), key=lambda kv: kv[1].get("total_words", 0)
        )[0]
        richest_speaker = max(
            per_speaker.items(), key=lambda kv: kv[1].get("type_token_ratio", 0.0)
        )[0]

    return {
        "global": {
            "total_segments": global_segments,
            "total_words": global_word_count,
            "unique_words": global_unique,
            "type_token_ratio": type_token_ratio,
            "avg_words_per_segment": avg_words_per_segment,
            "avg_words_per_sentence": avg_words_per_sentence,
            "days_covered": len(dates_sorted),
            "most_talkative_speaker": most_talkative_speaker,
            "richest_speaker": richest_speaker,
        },
        "per_speaker": per_speaker,
        "timeline": {
            "dates": dates_sorted,
            "unique_words": timeline_unique,
            "word_counts": timeline_counts,
            "notable_shifts": notable_shifts,
        },
        "top_words": {
            "global": top_global,
            "per_speaker": top_per_speaker,
            "distinctive_per_speaker": distinctive_per_speaker,
        },
    }


def query_vocab_summary(
    conn: sqlite3.Connection,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    speakers: Optional[Sequence[str]] = None,
    max_rows: int = 10000,
) -> Dict[str, Any]:
    """
    High-level helper used by the Insights service.

    Opens no connections and performs no writes; caller is responsible for managing
    the connection lifecycle.
    """
    rows = fetch_segments_for_vocab(
        conn,
        start_date=start_date,
        end_date=end_date,
        speakers=list(speakers) if speakers else None,
        max_rows=max_rows,
    )
    return compute_vocab_metrics(rows)
