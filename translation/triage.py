"""
translation/triage.py

TriageService — extracts structured emergency triage data from patient speech.

Phase 3: uses GemmaEngine.emergency_triage() which routes to gemma4:e4b
(REASONING_EXTRACTION mode, think=True) for careful severity inference.

Design:
  1. TriageService.extract() calls engine.emergency_triage() → raw JSON string
  2. _parse_json() strips markdown fences, falls back to regex JSON extraction
  3. Returns a fully-typed TriageResult TypedDict

Privacy: TriageService never touches audio — it operates on transcribed text
only.  The audio deletion contract is already enforced in audio.handler.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from config.languages import LANGUAGE_DISPLAY
from core.engine import GemmaEngine


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TriageResult(TypedDict):
    """Structured output from emergency triage extraction."""
    chief_complaint: str          # One-sentence English summary
    severity: str                 # "mild" | "moderate" | "severe" | "critical"
    duration: str                 # e.g. "since morning, ~3 hours" or "not mentioned"
    symptoms: list[str]           # Every distinct symptom reported
    vitals_mentioned: list[str]   # Any vitals the patient named (empty list if none)
    needs_immediate_attention: bool
    language: str                 # ISO 639-1 code of the patient's language


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TriageService:
    """
    Wraps GemmaEngine.emergency_triage() with JSON parsing and validation.

    Parameters
    ----------
    engine:
        A GemmaEngine instance.  Injected for testability — tests substitute
        a MagicMock without patching module globals.
    """

    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, lang_code: str) -> TriageResult:
        """
        Extract structured triage information from transcribed patient speech.

        Parameters
        ----------
        text:
            Raw Whisper transcript in the patient's language.
        lang_code:
            ISO 639-1 code from the 5 supported codes (hi/te/kn/en/ta).

        Returns
        -------
        TriageResult TypedDict with all fields populated.

        Raises
        ------
        ValueError
            If Gemma 4 returns output that cannot be parsed as JSON.
        RuntimeError
            If the Ollama inference call fails (propagated from GemmaEngine).
        """
        raw = self._engine.emergency_triage(text, lang_code)
        data = _parse_json(raw)
        return TriageResult(
            chief_complaint=str(data.get("chief_complaint", "unknown")),
            severity=_normalise_severity(data.get("severity", "unknown")),
            duration=str(data.get("duration", "not mentioned")),
            symptoms=_to_str_list(data.get("symptoms", [])),
            vitals_mentioned=_to_str_list(data.get("vitals_mentioned", [])),
            needs_immediate_attention=bool(data.get("needs_immediate_attention", False)),
            language=lang_code,
        )

    # ------------------------------------------------------------------
    # Convenience: display name
    # ------------------------------------------------------------------

    @staticmethod
    def language_display(lang_code: str) -> str:
        return LANGUAGE_DISPLAY.get(lang_code, lang_code)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """
    Parse the JSON string returned by Gemma 4.

    Attempts four strategies in order:
    1. Direct json.loads() on the full (stripped) string.
    2. Strip markdown fences (```json ... ```) then json.loads().
    3. Regex extraction of the first ``{...}`` block.
    4. Structural repair for truncated JSON (num_predict cut-off).

    Raises ValueError if all four fail.
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```\s*$", "", fenced)
    try:
        return json.loads(fenced.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: structural repair of truncated JSON
    candidate = fenced.strip() if fenced.strip().startswith("{") else text
    try:
        return _repair_truncated_json(candidate)
    except ValueError:
        pass

    raise ValueError(
        f"Gemma 4 returned non-JSON triage output (first 300 chars): "
        f"{text[:300]!r}"
    )


def _repair_truncated_json(text: str) -> dict:
    """
    Close an incomplete JSON object that was cut off mid-stream.

    Walks the string tracking open brace/bracket depth (ignoring content inside
    strings), then appends the necessary closing characters and retries
    json.loads().  Strips trailing dangling commas, partial keys, and unclosed
    string values before closing.
    """
    s = text.strip()
    if not s.startswith("{"):
        raise ValueError("Not a JSON object")

    stack: list[str] = []
    in_str = False
    esc = False

    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack:
        raise ValueError("Stack balanced — structurally complete but still invalid")

    trimmed = s.rstrip()
    for bad_suffix in ('",', '",\n', ",", "{\n", "{"):
        if trimmed.endswith(bad_suffix):
            trimmed = trimmed[: -len(bad_suffix)]
            break
    if trimmed.count('"') % 2 != 0:
        last_quote = trimmed.rfind('"')
        trimmed = trimmed[:last_quote].rstrip().rstrip(",")

    closers = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    repaired = trimmed.rstrip(",").rstrip() + closers

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Repair failed: {exc}") from exc


def _normalise_severity(value: object) -> str:
    """Coerce severity to one of the four valid labels; default to 'unknown'."""
    valid = {"mild", "moderate", "severe", "critical"}
    s = str(value).lower().strip()
    return s if s in valid else "unknown"


def _to_str_list(value: object) -> list[str]:
    """Coerce value to list[str], handling None / non-list gracefully."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or value == "":
        return []
    return [str(value)]
