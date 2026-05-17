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

def _strip_think_tags(text: str) -> str:
    """
    Remove Gemma 4 extended-thinking blocks from raw model output.

    When think=True is used (TRIAGE_THINK_MODE), Gemma 4 prefixes its answer
    with a <think>...</think> reasoning trace.  These blocks must be stripped
    BEFORE any JSON parsing attempt because:
      - The think block may itself contain JSON-like { } syntax (examples,
        partial reasoning) that fools the greedy regex in Strategy 3.
      - Direct json.loads() fails on a string that starts with <think>.

    Also handles the common variant where the closing tag is missing
    (truncated by num_predict cap) — in that case everything from <think>
    to end-of-string is removed.
    """
    # Remove complete <think>...</think> blocks (non-greedy, case-insensitive)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed <think> blocks (truncated by token limit)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_json(raw: str) -> dict:
    """
    Parse the JSON string returned by Gemma 4.

    Attempts five strategies in order:
    0. Strip Gemma 4 <think>...</think> reasoning blocks first.
    1. Direct json.loads() on the full (stripped) string.
    2. Strip markdown fences (```json ... ```) then json.loads().
    3. Balanced-brace extraction of the LAST ``{...}`` block — iterates
       all opening braces from right to left and picks the first one that
       forms a valid, balanced JSON object.  This avoids the greedy regex
       bug where content inside think blocks (which may contain JSON-like
       snippets) would be matched instead of the real output JSON.
    4. Structural repair for truncated JSON (num_predict cut-off).

    Raises ValueError if all strategies fail.
    """
    # Strategy 0: strip think-mode reasoning traces before anything else.
    # This is the primary fix for TRIAGE_THINK_MODE=True: the <think> block
    # often contains JSON-like { } syntax that breaks the greedy regex below.
    text = _strip_think_tags(raw.strip())

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

    # Strategy 3: balanced-brace extraction (largest JSON object wins).
    # Walk all opening-brace positions left-to-right, scan forward counting
    # brace depth until balanced, then attempt json.loads on that candidate.
    # Collect every valid candidate and pick the largest one.  The outermost
    # wrapper object is always larger than any nested object inside it, so this
    # correctly extracts the root JSON even when the model wraps it in
    # preamble text or the JSON contains nested objects.
    matches = list(re.finditer(r"\{", text))
    best: dict | None = None
    best_len = 0
    for m in matches:
        candidate = text[m.start():]
        depth = 0
        end = -1
        in_str = False
        esc = False
        for idx, ch in enumerate(candidate):
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
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end != -1:
            try:
                parsed = json.loads(candidate[:end + 1])
                if isinstance(parsed, dict) and len(candidate) > best_len:
                    best = parsed
                    best_len = len(candidate)
            except json.JSONDecodeError:
                continue
    if best is not None:
        return best

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

    Walks the string tracking open brace/bracket depth (ignoring content
    inside strings), then appends the necessary closing characters.

    Raises ValueError if the result still cannot be parsed.
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if stack:
                stack.pop()

    if not stack and not in_string:
        raise ValueError("JSON is not truncated or cannot be repaired")

    # Trim trailing garbage (dangling commas, unclosed strings)
    trimmed = text.rstrip()
    if in_string:
        trimmed += '"'  # close the open string
    trimmed = trimmed.rstrip(",").rstrip()

    # Append closers in reverse stack order
    closers = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    repaired = trimmed + closers

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Structural repair failed: {exc}") from exc


_VALID_SEVERITIES = {"mild", "moderate", "severe", "critical"}


def _normalise_severity(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    val = raw.strip().lower()
    return val if val in _VALID_SEVERITIES else "unknown"


def _to_str_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw  # type: ignore[return-value]
    return [raw]  # type: ignore[list-item]
