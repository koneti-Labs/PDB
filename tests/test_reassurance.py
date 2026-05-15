"""
tests/test_reassurance.py

Unit tests for translation/reassurance.py — ReassuranceService.
All GemmaEngine calls are mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from translation.reassurance import (
    REASSURANCE_PHRASES,
    ReassuranceService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_returning(text: str) -> MagicMock:
    eng = MagicMock()
    eng.emergency_reassurance.return_value = text
    return eng


# ---------------------------------------------------------------------------
# REASSURANCE_PHRASES bank tests
# ---------------------------------------------------------------------------

class TestPhrasesBank:
    def test_has_at_least_ten_phrases(self):
        assert len(REASSURANCE_PHRASES) >= 10

    def test_all_are_tuples_of_two_strings(self):
        for cat, phrase in REASSURANCE_PHRASES:
            assert isinstance(cat, str)
            assert isinstance(phrase, str)
            assert len(phrase) > 0

    def test_all_categories_are_valid(self):
        valid = {"URGENT", "MEDICAL", "COMFORT", "INFO"}
        for cat, _ in REASSURANCE_PHRASES:
            assert cat in valid, f"Unknown category: {cat}"

    def test_urgent_phrases_exist(self):
        urgent = [p for cat, p in REASSURANCE_PHRASES if cat == "URGENT"]
        assert len(urgent) >= 2

    def test_comfort_phrases_exist(self):
        comfort = [p for cat, p in REASSURANCE_PHRASES if cat == "COMFORT"]
        assert len(comfort) >= 1

    def test_no_empty_phrases(self):
        for _, phrase in REASSURANCE_PHRASES:
            assert phrase.strip(), "Found empty phrase"

    def test_phrases_are_in_english(self):
        # Very basic sanity: all phrases use ASCII-range characters
        for _, phrase in REASSURANCE_PHRASES:
            assert all(ord(c) < 256 for c in phrase), f"Non-ASCII in: {phrase}"

    def test_static_phrases_method_returns_same(self):
        assert ReassuranceService.phrases() == REASSURANCE_PHRASES


# ---------------------------------------------------------------------------
# ReassuranceService.translate() tests
# ---------------------------------------------------------------------------

class TestReassuranceServiceTranslate:
    def test_english_passthrough_no_engine_call(self):
        """If target_lang='en', return phrase unchanged without calling engine."""
        eng = MagicMock()
        svc = ReassuranceService(eng)
        phrase = "Help is coming. You are safe."
        result = svc.translate(phrase, "en")
        assert result == phrase
        eng.emergency_reassurance.assert_not_called()

    def test_hindi_calls_engine(self):
        eng = _engine_returning("मदद आ रही है। आप सुरक्षित हैं।")
        svc = ReassuranceService(eng)
        result = svc.translate("Help is coming. You are safe.", "hi")
        assert result == "मदद आ रही है। आप सुरक्षित हैं।"
        eng.emergency_reassurance.assert_called_once_with("Help is coming. You are safe.", "hi")

    def test_telugu_calls_engine(self):
        eng = _engine_returning("సహాయం వస్తోంది.")
        svc = ReassuranceService(eng)
        result = svc.translate("Help is coming. You are safe.", "te")
        assert "సహాయం" in result

    def test_kannada_calls_engine(self):
        eng = _engine_returning("ಸಹಾಯ ಬರುತ್ತಿದೆ.")
        svc = ReassuranceService(eng)
        result = svc.translate("Help is coming. You are safe.", "kn")
        eng.emergency_reassurance.assert_called_once()

    def test_tamil_calls_engine(self):
        eng = _engine_returning("உதவி வருகிறது.")
        svc = ReassuranceService(eng)
        result = svc.translate("Help is coming. You are safe.", "ta")
        eng.emergency_reassurance.assert_called_once()

    def test_engine_receives_correct_phrase_and_lang(self):
        eng = _engine_returning("test")
        svc = ReassuranceService(eng)
        svc.translate("We are calling an ambulance right now.", "te")
        eng.emergency_reassurance.assert_called_once_with(
            "We are calling an ambulance right now.", "te"
        )

    def test_runtime_error_propagates(self):
        eng = MagicMock()
        eng.emergency_reassurance.side_effect = RuntimeError("Ollama timeout")
        svc = ReassuranceService(eng)
        with pytest.raises(RuntimeError, match="Ollama timeout"):
            svc.translate("The doctor is on their way.", "hi")

    def test_empty_phrase_passthrough_english(self):
        eng = MagicMock()
        svc = ReassuranceService(eng)
        result = svc.translate("", "en")
        assert result == ""
        eng.emergency_reassurance.assert_not_called()

    def test_each_urgent_phrase_can_be_translated(self):
        """Smoke test: all URGENT phrases trigger engine call for non-English."""
        for cat, phrase in REASSURANCE_PHRASES:
            if cat == "URGENT":
                eng = _engine_returning(f"[translated] {phrase}")
                svc = ReassuranceService(eng)
                result = svc.translate(phrase, "hi")
                assert result.startswith("[translated]")

    def test_long_custom_phrase(self):
        long_phrase = "Please remain calm. " * 20
        eng = _engine_returning("कृपया शांत रहें।")
        svc = ReassuranceService(eng)
        result = svc.translate(long_phrase, "hi")
        assert result == "कृपया शांत रहें।"


# ---------------------------------------------------------------------------
# ReassuranceService.language_display() tests
# ---------------------------------------------------------------------------

class TestLanguageDisplay:
    def test_hindi(self):
        assert ReassuranceService.language_display("hi") == "Hindi"

    def test_telugu(self):
        assert ReassuranceService.language_display("te") == "Telugu"

    def test_kannada(self):
        assert ReassuranceService.language_display("kn") == "Kannada"

    def test_tamil(self):
        assert ReassuranceService.language_display("ta") == "Tamil"

    def test_english(self):
        assert ReassuranceService.language_display("en") == "English"

    def test_unknown_code_returns_code_itself(self):
        assert ReassuranceService.language_display("zz") == "zz"
