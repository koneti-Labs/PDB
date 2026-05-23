"""
audio/language_id.py

Language identification helpers.

Strategy
--------
Whisper detects ~99 languages and returns a full probability distribution.
We mask that distribution to our 5 supported codes and renormalize so that
the winner is always one of {hi, te, kn, en, ta}.

This prevents two common failure modes:
  • Hindi clips misclassified as Urdu (ur) or Punjabi (pa)
  • Telugu/Kannada clips misclassified as Malayalam (ml) or Marathi (mr)

After renormalization the confidence score reflects how dominant the winning
language is *within our 5-code constraint*, not globally.
"""
from __future__ import annotations

from config.languages import (
    LANGUAGE_DISPLAY,
    LANGUAGE_NATIVE,
    SUPPORTED_LANG_CODES,
)


def constrain_and_renormalize(
    all_probs: dict[str, float],
) -> tuple[str, float]:
    """
    Mask *all_probs* to SUPPORTED_LANG_CODES and renormalize to sum to 1.

    Parameters
    ----------
    all_probs:
        Whisper's full language probability dict, e.g.
        ``{"hi": 0.40, "ur": 0.30, "en": 0.15, ...}``

    Returns
    -------
    (lang_code, confidence)
        *lang_code* is always in SUPPORTED_LANG_CODES.
        *confidence* is the renormalized probability of the winning code, 0–1.

    Fallback
    --------
    If none of the 5 supported codes appear in *all_probs*, returns
    ``("en", 0.0)`` — English with zero confidence — so the caller can
    display a clear warning.
    """
    subset = {k: v for k, v in all_probs.items() if k in SUPPORTED_LANG_CODES}

    if not subset:
        return "en", 0.0

    total = sum(subset.values())
    if total <= 0.0:
        return "en", 0.0

    normalized = {k: v / total for k, v in subset.items()}
    best = max(normalized, key=normalized.__getitem__)
    return best, normalized[best]


def format_language_result(lang_code: str, confidence: float) -> str:
    """
    Build a human-readable language label for CLI output.

    Example output:
      ``Telugu / తెలుగు (te)  confidence: 0.94``
      ``Hindi / हिन्दी (hi)  confidence: 0.31  ⚠ low confidence``  ← caller adds warning

    The ⚠ suffix is NOT added here; the caller checks CONFIDENCE_THRESHOLD.
    """
    name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
    native = LANGUAGE_NATIVE.get(lang_code, "")

    # Avoid redundant duplication for English ("English / English")
    if native and native != name:
        label = f"{name} / {native} ({lang_code})"
    else:
        label = f"{name} ({lang_code})"

    return f"{label}   confidence: {confidence:.2f}"
