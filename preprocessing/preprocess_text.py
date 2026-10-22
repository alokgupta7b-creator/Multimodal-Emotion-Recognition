"""
preprocessing.preprocess_text
==============================
Text normalisation utilities for multimodal emotion recognition.

Design principles
-----------------
* Operates independently of any specific dataset schema.
* Preserves emotion-bearing tokens: negation words (``not``, ``no``,
  ``never``), emotionally significant punctuation (``!``, ``?``), and
  common contractions.
* Does **not** remove stop-words or apply stemming/lemmatisation — those
  decisions belong to downstream feature-extraction code.
* Never modifies the ground-truth labels; text cleaning touches only the
  utterance string.
* Treats ``None``, empty, and whitespace-only strings as missing text
  rather than raising unhandled exceptions.

Public API
----------
    normalize_text(text, preserve_punctuation=True) -> str
    preprocess_text_record(record, text_field, normalize) -> dict
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Characters to remove: control characters and "curly" typographic quotes
# that carry no semantic weight and can confuse tokenisers.
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TYPOGRAPHIC_QUOTES_RE = re.compile(r"[\u2018\u2019\u201c\u201d\u2032\u2033]")

# Collapse any whitespace sequence (including NBSP, thin space, etc.) to a
# single ASCII space, then strip.
_WHITESPACE_RE = re.compile(r"\s+")

# URL pattern: http(s):// or www. prefixes
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# E-mail pattern
_EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w.\-]+\.\w+\b")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def normalize_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    lowercase: bool = False,
    preserve_punctuation: bool = True,
) -> str:
    """
    Apply a reproducible, order-preserving text normalisation pipeline.

    Steps applied in order:
    1. Unicode NFKC normalisation (converts full-width chars, ligatures, etc.)
    2. Replace typographic quote variants with ASCII apostrophes/quotes.
    3. Strip control characters (C0 and C1 except ``\\t``, ``\\n``, ``\\r``).
    4. Optionally remove URLs and e-mail addresses.
    5. Optionally lowercase.
    6. Optionally remove characters that are not letters, digits, spaces,
       apostrophes, hyphens, or sentence punctuation.
    7. Collapse whitespace and strip.

    Parameters
    ----------
    text:
        Input string.  Must be ``str``; ``None`` is not accepted here — use
        ``preprocess_text_record`` for safe handling.
    remove_urls:
        Replace HTTP/HTTPS and www URLs with a single space.
    remove_emails:
        Replace e-mail addresses with a single space.
    lowercase:
        Convert the result to lowercase.
    preserve_punctuation:
        When ``True`` (default), ``!``, ``?``, ``'``, ``-``, ``:``, ``,``,
        and ``.`` are kept.  These carry emotional signal in conversational
        text.  When ``False``, only alphanumeric characters and spaces remain.

    Returns
    -------
    str
        Cleaned, normalised string (may be empty).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"normalize_text expects a str, got {type(text).__name__!r}"
        )

    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Typographic quotes → ASCII
    text = _TYPOGRAPHIC_QUOTES_RE.sub("'", text)

    # 3. Control characters
    text = _CTRL_CHARS_RE.sub(" ", text)

    # 4. URLs and e-mails
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_emails:
        text = _EMAIL_RE.sub(" ", text)

    # 5. Case
    if lowercase:
        text = text.lower()

    # 6. Character filtering
    if preserve_punctuation:
        # Keep letters, digits, whitespace, and selected punctuation
        text = re.sub(r"[^\w\s!?'.,;:\-]", " ", text)
    else:
        # Keep only alphanumeric and whitespace
        text = re.sub(r"[^\w\s]", " ", text)

    # 7. Whitespace collapse and strip
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def preprocess_text_record(
    record: Dict[str, Any],
    text_field: str = "text",
    normalize: bool = True,
    lowercase: bool = False,
    preserve_punctuation: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
) -> Tuple[str, str, str]:
    """
    Extract, validate, and optionally normalise the text field from a
    manifest record dict.

    Does **not** modify the input ``record`` in place.

    Parameters
    ----------
    record:
        A manifest row (dict) from either the MELD or CMU-MOSEI manifest.
    text_field:
        Key to look up within ``record``.  For MELD manifests produced
        with ``include_transcript_text=True`` the key is ``"text"``.  For
        CMU-MOSEI manifests the key is ``"transcript"``.  Pass whichever
        is appropriate.
    normalize:
        Apply ``normalize_text`` when ``True`` (default).  When ``False``
        the raw value is returned as-is (after ``str`` coercion and strip).
    lowercase, preserve_punctuation, remove_urls, remove_emails:
        Forwarded to ``normalize_text`` when ``normalize=True``.

    Returns
    -------
    (raw_text, processed_text, status)
        raw_text       — original string value from the record (stripped).
        processed_text — normalised string (equals raw_text when normalize=False).
        status         — ``"ok"``, ``"missing_field"``, ``"empty"``,
                         ``"whitespace_only"``, or ``"null"``.
    """
    raw_value = record.get(text_field)

    # Null / None
    if raw_value is None:
        return "", "", "null"

    # Coerce to str and strip
    raw_str = str(raw_value).strip()

    if raw_str == "" or raw_str.lower() in ("none", "nan", "null"):
        return "", "", "empty"

    if not raw_str.replace(" ", ""):
        return raw_str, "", "whitespace_only"

    if normalize:
        processed = normalize_text(
            raw_str,
            remove_urls=remove_urls,
            remove_emails=remove_emails,
            lowercase=lowercase,
            preserve_punctuation=preserve_punctuation,
        )
    else:
        processed = raw_str

    return raw_str, processed, "ok"


def get_text_field_for_dataset(dataset: str) -> str:
    """
    Return the manifest column name that holds utterance text for a given
    dataset.

    Parameters
    ----------
    dataset:
        ``"meld"`` or ``"mosei"``.

    Returns
    -------
    str
        Column name: ``"text"`` for MELD (when ``include_transcript_text``
        was set) or ``"transcript"`` for CMU-MOSEI.
    """
    if dataset == "meld":
        return "text"
    if dataset == "mosei":
        return "transcript"
    raise ValueError(f"Unknown dataset: {dataset!r}")


def load_transcript_source(
    source_path: Union[Path, str],
    expected_split: Optional[str] = None,
) -> Dict[Tuple[str, str], str]:
    """
    Load external utterance transcripts from a CSV file or directory.

    Supported formats:
    1. Single CSV file (e.g. MELD ``train_sent_emo.csv``).
       Must contain dialogue ID, utterance ID, and utterance text columns.
    2. A directory containing split-specific CSVs (e.g. ``datasets/MELD/annotations``).
       Resolves ``{split}_sent_emo.csv`` or ``{split}.csv`` if ``expected_split``
       is supplied.

    Parameters
    ----------
    source_path:
        Path to a CSV file or directory of CSV files.
    expected_split:
        When set, filters records or selects the split-specific file.

    Returns
    -------
    dict: (dialogue_id_str, utterance_id_str) -> raw_utterance_text
    """
    import csv
    from pathlib import Path

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript source not found: {source_path}")

    files_to_read: List[Path] = []
    if source_path.is_dir():
        if expected_split:
            candidates = [
                source_path / f"{expected_split}_sent_emo.csv",
                source_path / f"{expected_split}.csv",
                source_path / f"{expected_split}_annotations.csv",
            ]
            matched = [c for c in candidates if c.exists()]
            if matched:
                files_to_read = [matched[0]]
            else:
                files_to_read = sorted(source_path.glob("*.csv"))
        else:
            files_to_read = sorted(source_path.glob("*.csv"))
    else:
        files_to_read = [source_path]

    if not files_to_read:
        raise ValueError(
            f"No valid transcript CSV files found in: {source_path}"
        )

    transcripts: Dict[Tuple[str, str], str] = {}

    for csv_file in files_to_read:
        with open(csv_file, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue

            # Identify columns case-insensitively
            col_map = {col.lower().strip(): col for col in reader.fieldnames}

            # Dialogue ID
            dia_col = None
            for cand in ("dialogue_id", "dialogueid", "dialogue"):
                if cand in col_map:
                    dia_col = col_map[cand]
                    break

            # Utterance ID
            utt_col = None
            for cand in ("utterance_id", "utteranceid", "utterance_no"):
                if cand in col_map:
                    utt_col = col_map[cand]
                    break

            # Text / Utterance string
            text_col = None
            for cand in ("utterance", "text", "transcript", "utterance_text"):
                if cand in col_map:
                    text_col = col_map[cand]
                    break

            split_col = col_map.get("split")

            if dia_col is None or utt_col is None or text_col is None:
                continue

            for row in reader:
                if (
                    split_col
                    and expected_split
                    and str(row.get(split_col, "")).strip().lower()
                    != expected_split.lower()
                ):
                    continue

                did = str(row[dia_col]).strip()
                uid = str(row[utt_col]).strip()
                txt = str(row[text_col]).strip()
                transcripts[(did, uid)] = txt

    return transcripts
