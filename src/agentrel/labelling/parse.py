"""Parse a usable answer out of the agent's free-text final answer.

Agents answer in prose ("The survival rate is about 0.38, or 38%."). To score
against a scripted value we need the number or the category label. This is part
of the "messy logs" cleaning the report calls out (section 3.3).
"""
from __future__ import annotations

import re

_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
_PCT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")


def parse_numeric(text: str | None) -> float | None:
    """Return the most likely numeric answer from free text.

    Preference order: an explicit percentage (converted to a fraction, since our
    rates are fractions) only if no bare fraction is present; otherwise the LAST
    number in the string, which is usually the concluding value."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    # a trailing "= 0.38" or bare number is the strongest signal
    nums = _NUM_RE.findall(text)
    cleaned = [n.replace(",", "") for n in nums]
    floats = []
    for c in cleaned:
        try:
            floats.append(float(c))
        except ValueError:
            continue

    # percentage handling: "38%" -> 0.38, but only when NO plausible bare
    # fraction (a value in [0,1] that isn't itself the percent number) is present
    pct = _PCT_RE.search(text)
    if pct:
        try:
            pct_val = float(pct.group(1))
        except ValueError:
            pct_val = None
        fractions = [f for f in floats if 0.0 <= f <= 1.0 and f != pct_val]
        if pct_val is not None:
            if fractions:
                return fractions[-1]       # a bare fraction wins over the percent
            return pct_val / 100.0

    return floats[-1] if floats else None


def parse_categorical(text: str | None) -> str | None:
    """Return a normalised category label from free text.

    Strips surrounding prose/quotes/punctuation and lowercases, so 'home' and
    '"Home."' compare equal. Keeps the last token-ish span, which is usually the
    answer."""
    if text is None:
        return None
    text = text.strip().strip(".\"'` ")
    if not text:
        return None
    # if it looks like "the answer is X", take the tail after a colon or 'is'
    for sep in (":", " is ", " = "):
        if sep in text.lower():
            text = re.split(sep, text, flags=re.IGNORECASE)[-1]
    return text.strip().strip(".\"'` ").lower()
