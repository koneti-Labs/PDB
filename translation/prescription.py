"""
translation/prescription.py

PrescriptionService -- extracts structured medicine data from a prescription image
and (Phase 4b) translates the result into the patient's native language.

Phase 4:  uses GemmaEngine.transcribe_prescription() -> gemma4:e4b (vision)
Phase 4b: uses GemmaEngine.translate_prescription_summary() -> gemma4:e2b (translation)
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
    Wraps GemmaEngine with prescription OCR (Phase 4) and translation (Phase 4b).

    Parameters
    ----------
    engine : GemmaEngine
        Injected for testability.
    """

    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    def translate_summary(self, result: PrescriptionResult, target_lang: str) -> str:
        """
        Produce a patient-friendly prescription explanation in the target language.

        Builds a plain-English summary from *result*, then calls
        GemmaEngine.translate_prescription_summary() (gemma4:e2b, FAST_TRANSLATION)
        to translate it into the patient's native language.

        Parameters
        ----------
        result : PrescriptionResult
            Structured data returned by extract().
        target_lang : str
            ISO 639-1 code (hi/te/kn/ta).  Returns the English summary unchanged
            when target_lang is "en".

        Returns
        -------
        str
            Translated, patient-friendly explanation.
        """
        english_summary = _build_english_summary(result)
        if target_lang == "en":
            return english_summary
        return self._engine.translate_prescription_summary(english_summary, target_lang)

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_english_summary(result: PrescriptionResult) -> str:
    """
    Build a plain-English, patient-readable summary from a PrescriptionResult.

    Used by PrescriptionService.translate_summary() as input to the translation
    prompt.  Keeps formatting minimal so the Gemma translation prompt is not
    cluttered with markdown or table syntax.
    """
    lines: list[str] = []

    if result["doctor_name"] not in ("not visible", ""):
        lines.append("Doctor: " + result["doctor_name"])
    if result["patient_name"] not in ("not visible", ""):
        lines.append("Patient: " + result["patient_name"])
    if result["date"] not in ("not visible", ""):
        lines.append("Date: " + result["date"])

    if lines:
        lines.append("")  # blank separator before medicines

    if result["medicines"]:
        lines.append("Medicines prescribed:")
        for i, med in enumerate(result["medicines"], 1):
            parts = [str(i) + ". " + med["name"]]
            if med["dosage"] not in ("not specified", ""):
                parts.append(med["dosage"])
            if med["form"] not in ("other", ""):
                parts.append("(" + med["form"] + ")")
            lines.append(" ".join(parts))

            if med["frequency"] not in ("not specified", ""):
                lines.append("   Take: " + med["frequency"])
            if med["duration"] not in ("not specified", ""):
                lines.append("   For: " + med["duration"])
            if med["instructions"] not in ("none", ""):
                lines.append("   Note: " + med["instructions"])
    else:
        lines.append("No medicines could be read from the prescription.")

    if result["notes"] not in ("", "none"):
        lines.append("\nAdditional instructions: " + result["notes"])

    return "\n".join(lines)


def _strip_think_tags(text: str) -> str:
    """
    Remove Gemma 4 extended-thinking blocks from raw model output.

    gemma4:e4b (REASONING_EXTRACTION) may emit <think>...</think> blocks even
    for vision/OCR tasks.  These must be stripped before JSON parsing because
    the think block can contain JSON-like { } syntax that breaks the greedy
    regex strategy.  Also handles truncated blocks (no closing tag).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_json(raw: str) -> dict:
    """
    Parse JSON from Gemma 4 output using five strategies:
    0. Strip <think>...</think> reasoning blocks (gemma4:e4b think mode).
    1. Direct parse.
    2. Strip markdown fences then parse.
    3. Find the LAST balanced { ... } block (avoids think-block JSON-like noise).
    4. Attempt structural repair of truncated JSON (num_predict cut-off).
    """
    # Strategy 0: strip think-mode reasoning traces so they don't pollute
    # downstream strategies with JSON-like content from the reasoning section.
    text = _strip_think_tags(raw.strip())

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

    # Strategy 3: find the LARGEST balanced { ... } block.
    # Collect every valid JSON candidate and pick the one with the most
    # characters.  The outermost wrapper object is always larger than any
    # nested object inside it, so this correctly extracts the root JSON even
    # when the model wraps it in preamble text.
    matches = list(re.finditer(r"\{", text))
    best: dict | None = None
    best_len = 0
    for m in matches:
        candidate = text[m.start():]
        depth = 0
        end = -1
        in_str = False
        esc = False
        for i, ch in enumerate(candidate):
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
                    end = i
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
    # (happens when GEMMA_NUM_PREDICT cap cuts off the model mid-output)
    candidate = clean if clean.startswith("{") else text
    try:
        return _repair_truncated_json(candidate)
    except ValueError:
        pass

    raise ValueError(
        "Gemma 4 returned non-JSON prescription output: " + repr(text[:300])
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
        raise ValueError("Structural repair failed: " + str(exc)) from exc


def _normalise_medicines(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": str(item.get("name", "unknown")),
                "dosage": str(item.get("dosage", "not specified")),
                "form": str(item.get("form", "other")),
                "frequency": str(item.get("frequency", "not specified")),
                "duration": str(item.get("duration", "not specified")),
                "instructions": str(item.get("instructions", "none")),
            }
        )
    return out
