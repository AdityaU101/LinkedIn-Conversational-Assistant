"""Text utilities shared across ingestion, parsing and matching."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|section)[^>]*>", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9+#.]+")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'fÃ¼r' -> 'für').

    Only applied when the repair round-trips cleanly and reduces the number of
    typical mojibake markers, so correct text is never altered.
    """
    if not text or ("Ã" not in text and "Â" not in text and "â€" not in text):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    markers = lambda s: s.count("Ã") + s.count("Â") + s.count("â€")  # noqa: E731
    return repaired if markers(repaired) < markers(text) else text


def strip_html(text: str) -> str:
    """Convert an HTML fragment to plain text while keeping block boundaries."""
    if not text:
        return ""
    text = _BLOCK_TAG.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    lines = [collapse_ws(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_for_hash(text: str) -> str:
    """Aggressive normalisation used for content hashing / dedup."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return collapse_ws(text)


def content_hash(text: str) -> str:
    return sha1(normalize_for_hash(text))


def normalize_key(text: str | None) -> str:
    """Normalise short strings (titles, companies, skills) for matching."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _NON_ALNUM.sub(" ", text)
    return collapse_ws(text).strip(". ")


def slug(text: str | None, max_len: int = 80) -> str:
    key = normalize_key(text).replace("+", "plus").replace("#", "sharp")
    key = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    return key[:max_len] or "unknown"


def word_chunks(text: str, chunk_words: int = 350, overlap: float = 0.10) -> list[tuple[int, int, str]]:
    """Split text into word chunks with fractional overlap.

    Returns ``(start_char, end_char, chunk_text)`` so that offsets inside a
    chunk can be mapped back to the source document.
    """
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return []
    step = max(1, int(chunk_words * (1 - overlap)))
    chunks: list[tuple[int, int, str]] = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + chunk_words]
        start, end = window[0][0], window[-1][1]
        chunks.append((start, end, text[start:end]))
        if i + chunk_words >= len(tokens):
            break
    return chunks


def find_span(haystack: str, needle: str, start_hint: int = 0) -> tuple[int, int] | None:
    """Locate ``needle`` in ``haystack`` (exact, then case/whitespace-insensitive)."""
    if not needle:
        return None
    idx = haystack.find(needle, start_hint)
    if idx < 0:
        idx = haystack.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    words = [re.escape(w) for w in needle.split()]
    if not words:
        return None
    pattern = re.compile(r"\s+".join(words), re.I)
    m = pattern.search(haystack)
    return (m.start(), m.end()) if m else None
