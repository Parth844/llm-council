"""Answer extraction helpers shared by the engine and the eval harness."""

from __future__ import annotations

import re

_FINAL_RE = re.compile(r"FINAL ANSWER\s*[:\-]?\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_answer(text: str) -> str | None:
    """Return the content of the last 'FINAL ANSWER:' line, if present."""
    matches = _FINAL_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip().strip("*_` ")
    return answer or None


def extract_number(text: str) -> float | None:
    """Extract a numeric answer: last number on the FINAL ANSWER line,
    falling back to the last number anywhere in the text (GSM8K-style)."""
    target = extract_final_answer(text) or text
    nums = _NUM_RE.findall(target)
    if not nums:
        nums = _NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None
