"""
translation/prescription.py

PrescriptionService -- extracts structured medicine data from a prescription image.

Phase 4: uses GemmaEngine.transcribe_prescription() which routes to gemma4:e4b
(REASONING_EXTRACTION mode, multimodal vision).
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from core.engine import GemmaEngine


class MedicineItem(TypedDict):
    name: str
    dosage: str
    form: str
    frequency: str
    duration: str
    instructions: str


class PrescriptionResult(TypedDict):
    medicines: list[MedicineItem]
    doctor_name: str
    patient_name: str
    date: str
    notes: str


class PrescriptionService:
    """
    Wraps GemmaEngine.transcribe_prescription() with JSON parsing.

    Parameters
    ----------
    engine : GemmaEngine
        Injected for testability.
    """

    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    def extract(self, image_path: str) -> PrescriptionResult:
        """
        Extract structured prescription data from an image file.

        Parameters
        ----------
        image_path : str
            Path to prescription JPEG or PNG.

        Returns
        -------
        PrescriptionResult TypedDict.

        Raises
        ------
        ValueError
            If Gemma 4 returns non-JSON output.
        RuntimeError
            If Ollama inference fails.
        """
        raw = self._engine.transcribe_prescription(image_path)
        data = _parse_json(raw)
        return PrescriptionResult(
            medicines=_normalise_medicines(data.get("medicines", [])),
            doctor_name=str(data.get("doctor_name", "not visible")),
            patient_name=str(data.get("patient_name", "not visible")),
            date=str(data.get("date", "not visible")),
            notes=str(data.get("notes", "")),
        )


def _parse_json(raw: str) -> dict:
    """
    Parse JSON from Gemma 4 output using four strategies:
    1. Direct parse.
    2. Strip markdown fences then parse.
    3. Regex-extract first { ... } block.
    4. Attempt structural repair of truncated JSON (num_predict cut-off).
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```\s*$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first { ... } block (greedy)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: structural repair of truncated JSON
    # (happens when GEMMA_NUM_PREDICT cap cuts off the model mid-output)
    candidate = clean if clean.startswith("{") else text
    try:
        return _repair_truncated_json(candidate)
    except ValueError:
        pass

    raise ValueError(f"Gemma 4 returned non-JSON prescription output: {text[:200]!r}")


def _repair_truncated_json(text: str) -> dict:
    """
    Attempt to close an incomplete JSON object that was truncated mid-stream.

    Walks the string character-by-character tracking open braces/brackets,
    then appends the necessary closing characters and retries json.loads().
    Strips any trailing incomplete value (partial string, partial key, dangling
    comma) before closing.
    """
    s = text.strip()
    if not s.startswith("{"):
        raise ValueError("Not a JSON object")

    # Walk string tracking open structure depth (ignoring content inside strings)
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

    # If stack is empty the JSON was valid (parse error is semantic, not structural)
    if not stack:
        raise ValueError("Stack balanced — JSON is structurally complete but still invalid")

    # Trim trailing garbage: remove anything after the last complete value
    # (trailing comma, partial key/value, unclosed string)
    trimmed = s.rstrip()
    # Remove trailing open-string fragment (odd number of unescaped quotes at end)
    # and partial comma-separated entry
    for bad_suffix in ('",', '",\n', ",", "{\n", "{"):
        if trimmed.endswith(bad_suffix):
            trimmed = trimmed[: -len(bad_suffix)]
            break
    # Also strip a dangling un-closed string value at the very end
    if trimmed.count('"') % 2 != 0:
        last_quote = trimmed.rfind('"')
        trimmed = trimmed[:last_quote].rstrip().rstrip(",")

    # Close all open structures
    closers = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    repaired = trimmed.rstrip(",").rstrip() + closers

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Repair failed: {exc} — snippet: {repaired[-80:]!r}") from exc


def _normalise_medicines(raw: object) -> list[MedicineItem]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(MedicineItem(
            name=str(item.get("name", "unknown")),
            dosage=str(item.get("dosage", "not specified")),
            form=str(item.get("form", "other")),
            frequency=str(item.get("frequency", "not specified")),
            duration=str(item.get("duration", "not specified")),
            instructions=str(item.get("instructions", "none")),
        ))
    return result
