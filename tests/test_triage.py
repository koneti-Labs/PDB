"""
tests/test_triage.py

Unit tests for Phase 3: Emergency Triage Extraction.

Coverage:
  - GemmaEngine.emergency_triage():
      model routing (REASONING_EXTRACTION → gemma4:e4b)
      think=True passed to client.generate()
      prompt contains patient text + language display name
      RuntimeError propagated on Ollama failure

  - TriageService.extract():
      happy-path JSON parsing → TriageResult
      markdown fence stripping (```json ... ```)
      JSON embedded in prose (regex fallback)
      garbage → ValueError
      all 5 language codes resolve correct display names
      TriageResult has all required TypedDict fields
      needs_immediate_attention flag
      severity normalisation
      symptoms and vitals_mentioned are lists

  - _parse_json() and helpers (tested via TriageService)
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Ensure stubs present (conftest loads them, but belt-and-suspenders here)
# ---------------------------------------------------------------------------
if "sounddevice" not in sys.modules:
    sd = types.ModuleType("sounddevice")
    sd.InputStream = MagicMock()
    sys.modules["sounddevice"] = sd

if "ollama" not in sys.modules:
    ol = types.ModuleType("ollama")
    class _StubClient:
        def __init__(self, **kw): pass
        def generate(self, **kw): return {"response": ""}
        def list(self): return {"models": []}
    ol.Client = _StubClient
    sys.modules["ollama"] = ol

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from core.engine import GemmaEngine, InferenceMode, MODELS
from translation.triage import TriageService, TriageResult, _parse_json, _normalise_severity, _to_str_list

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_TRIAGE_JSON = {
    "chief_complaint": "severe chest pain radiating to left arm",
    "severity": "severe",
    "duration": "since this morning, about 2 hours",
    "symptoms": ["chest pain", "shortness of breath", "sweating"],
    "vitals_mentioned": ["pulse racing"],
    "needs_immediate_attention": True,
}


def _engine_with_mock(raw_response: str = "") -> tuple[GemmaEngine, MagicMock]:
    """Return (engine, mock_client) with client.generate pre-configured."""
    engine = GemmaEngine()
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": raw_response}
    engine._client = mock_client
    # REASONING_EXTRACTION calls use _reasoning_client — wire the same mock
    engine._reasoning_client = mock_client
    return engine, mock_client


def _service_with_mock(raw_response: str) -> tuple[TriageService, MagicMock]:
    """Return (service, mock_client) with the engine pre-wired."""
    engine, mock_client = _engine_with_mock(raw_response)
    return TriageService(engine), mock_client


# ===========================================================================
# 1. GemmaEngine.emergency_triage() — model routing
# ===========================================================================

def test_engine_triage_uses_reasoning_extraction_model():
    """emergency_triage() must route to gemma4:e4b (REASONING_EXTRACTION)."""
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage("మీకు ఏమి జరిగింది?", "te")
    kwargs = mock_client.generate.call_args.kwargs
    assert kwargs["model"] == MODELS[InferenceMode.REASONING_EXTRACTION]
    assert kwargs["model"] == MODELS[InferenceMode.REASONING_EXTRACTION]


def test_engine_triage_model_is_gemma4():
    """All MODELS values must start with 'gemma4:' — competition rule."""
    assert all(v.startswith("gemma4:") for v in MODELS.values())


def test_engine_triage_passes_think_true():
    """emergency_triage() must pass think=True to client.generate()."""
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage("सीने में दर्द है", "hi")
    kwargs = mock_client.generate.call_args.kwargs
    assert kwargs.get("think") is True


def test_engine_triage_prompt_contains_text():
    """Prompt must embed the patient's transcript."""
    patient_text = "ನನಗೆ ತಲೆನೋವು ಇದೆ"
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage(patient_text, "kn")
    prompt = mock_client.generate.call_args.kwargs["prompt"]
    assert patient_text in prompt


def test_engine_triage_prompt_contains_language_name():
    """Prompt must contain the display name 'English' (not just bare code 'en')."""
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage("chest pain", "en")
    prompt = mock_client.generate.call_args.kwargs["prompt"]
    # Display name must appear in the patient statement line
    assert "English" in prompt
    assert "Patient statement (English)" in prompt


def test_engine_triage_prompt_language_tamil():
    """Tamil lang_code → 'Tamil' in prompt."""
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage("வலி உள்ளது", "ta")
    prompt = mock_client.generate.call_args.kwargs["prompt"]
    assert "Tamil" in prompt


def test_engine_triage_raises_on_ollama_failure():
    """RuntimeError from Ollama must propagate as-is."""
    engine, mock_client = _engine_with_mock()
    mock_client.generate.side_effect = ConnectionRefusedError("Ollama not running")
    with pytest.raises(RuntimeError, match="Ollama inference failed"):
        engine.emergency_triage("pain", "en")


def test_engine_triage_error_message_includes_model():
    """RuntimeError message must name the model."""
    engine, mock_client = _engine_with_mock()
    mock_client.generate.side_effect = Exception("timeout")
    with pytest.raises(RuntimeError, match=MODELS[InferenceMode.REASONING_EXTRACTION]):
        engine.emergency_triage("pain", "en")


# ===========================================================================
# 2. TriageService.extract() — JSON parsing
# ===========================================================================

def test_service_extract_happy_path():
    """Valid JSON response → fully-typed TriageResult."""
    service, _ = _service_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    result = service.extract("chest pain", "en")
    assert result["chief_complaint"] == _VALID_TRIAGE_JSON["chief_complaint"]
    assert result["severity"] == "severe"
    assert result["needs_immediate_attention"] is True
    assert isinstance(result["symptoms"], list)
    assert isinstance(result["vitals_mentioned"], list)


def test_service_extract_strips_markdown_fences():
    """Response wrapped in ```json ... ``` must be parsed correctly."""
    fenced = f"```json\n{json.dumps(_VALID_TRIAGE_JSON)}\n```"
    service, _ = _service_with_mock(fenced)
    result = service.extract("chest pain", "en")
    assert result["severity"] == "severe"


def test_service_extract_finds_json_in_prose():
    """JSON buried in reasoning prose must be extracted via regex fallback."""
    prose = f"Let me think...\n\nHere is the answer:\n{json.dumps(_VALID_TRIAGE_JSON)}\n\nDone."
    service, _ = _service_with_mock(prose)
    result = service.extract("chest pain", "en")
    assert result["chief_complaint"] == _VALID_TRIAGE_JSON["chief_complaint"]


def test_service_extract_raises_value_error_on_garbage():
    """Non-JSON output must raise ValueError."""
    service, _ = _service_with_mock("I cannot help with that.")
    with pytest.raises(ValueError, match="non-JSON"):
        service.extract("chest pain", "en")


def test_service_language_stored_in_result():
    """lang_code is always stored in TriageResult.language."""
    service, _ = _service_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    result = service.extract("some text", "te")
    assert result["language"] == "te"


# ===========================================================================
# 3. TriageResult TypedDict — fields and types
# ===========================================================================

def test_triage_result_has_all_required_fields():
    """TriageResult must expose all seven specified fields."""
    service, _ = _service_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    result = service.extract("chest pain", "en")
    required = {
        "chief_complaint", "severity", "duration",
        "symptoms", "vitals_mentioned", "needs_immediate_attention", "language",
    }
    assert required.issubset(result.keys())


def test_symptoms_is_list_of_strings():
    service, _ = _service_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    result = service.extract("text", "hi")
    assert isinstance(result["symptoms"], list)
    assert all(isinstance(s, str) for s in result["symptoms"])


def test_vitals_mentioned_is_list():
    service, _ = _service_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    result = service.extract("text", "hi")
    assert isinstance(result["vitals_mentioned"], list)


def test_needs_immediate_attention_false_for_mild():
    mild_data = {**_VALID_TRIAGE_JSON, "severity": "mild", "needs_immediate_attention": False}
    service, _ = _service_with_mock(json.dumps(mild_data))
    result = service.extract("mild headache", "en")
    assert result["needs_immediate_attention"] is False


def test_empty_vitals_returns_empty_list():
    data = {**_VALID_TRIAGE_JSON, "vitals_mentioned": []}
    service, _ = _service_with_mock(json.dumps(data))
    result = service.extract("text", "en")
    assert result["vitals_mentioned"] == []


# ===========================================================================
# 4. All 5 language codes → correct display name in prompt
# ===========================================================================

@pytest.mark.parametrize("code,display", [
    ("hi", "Hindi"),
    ("te", "Telugu"),
    ("kn", "Kannada"),
    ("en", "English"),
    ("ta", "Tamil"),
])
def test_all_language_codes_route_display_name(code, display):
    engine, mock_client = _engine_with_mock(json.dumps(_VALID_TRIAGE_JSON))
    engine.emergency_triage("text", code)
    prompt = mock_client.generate.call_args.kwargs["prompt"]
    assert display in prompt


# ===========================================================================
# 5. _normalise_severity helper
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    ("mild", "mild"),
    ("MODERATE", "moderate"),
    ("Severe", "severe"),
    ("critical", "critical"),
    ("life-threatening", "unknown"),
    ("", "unknown"),
])
def test_normalise_severity(raw, expected):
    assert _normalise_severity(raw) == expected


# ===========================================================================
# 6. _to_str_list helper
# ===========================================================================

def test_to_str_list_from_list():
    assert _to_str_list(["a", "b"]) == ["a", "b"]


def test_to_str_list_from_none():
    assert _to_str_list(None) == []


def test_to_str_list_from_scalar():
    assert _to_str_list("chest pain") == ["chest pain"]
