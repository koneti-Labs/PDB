"""
config/languages.py

Single source of truth for language codes, display names, and native labels.

Phase 2 upgrade path: if a Whisper transcription returns low confidence for
Kannada or Telugu, route through IndicConformer-600M for a second-pass
transcription (language code already known from Phase 1 detection).
"""
from __future__ import annotations

# Constrained to these 5 codes only.
# Whisper's full probability distribution is masked to this set and renormalized.
# This prevents misclassification into closely related languages
# (e.g. Hindi → Urdu, Telugu → some other Dravidian language).
SUPPORTED_LANG_CODES: frozenset[str] = frozenset({"hi", "te", "kn", "en", "ta"})

# Human-readable display names (Latin script)
LANGUAGE_DISPLAY: dict[str, str] = {
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "en": "English",
    "ta": "Tamil",
}

# Native script labels — shown alongside Latin name in CLI output
LANGUAGE_NATIVE: dict[str, str] = {
    "hi": "हिन्दी",
    "te": "తెలుగు",
    "kn": "ಕನ್ನಡ",
    "en": "English",
    "ta": "தமிழ்",
}

# Minimum renormalized confidence to consider a detection reliable.
# Below this threshold the CLI will show a ⚠ low-confidence warning.
CONFIDENCE_THRESHOLD: float = 0.40
