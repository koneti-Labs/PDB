"""
tests/test_translation.py

Unit tests for translation/service.py.

GemmaEngine is mocked so tests are pure prompt-construction verification.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.languages import LANGUAGE_DISPLAY
from translation.prompts import DOCTOR_TO_PATIENT_PROMPT, PATIENT_TO_DOCTOR_PROMPT
from translation.service import TranslationService


def _service_with_mock(return_text: str = "translated") -> tuple[TranslationService, MagicMock]:
    mock_engine = MagicMock()
    mock_engine.generate.return_value = return_text
    return TranslationService(mock_engine), mock_engine


class TestPatientToDoctor:

    def test_returns_engine_output(self) -> None:
        service, _ = _service_with_mock("Patient has a headache and fever.")
        result = service.patient_to_doctor("నాకు తల నొప్పి", "te")
        assert result == "Patient has a headache and fever."

    def test_prompt_contains_source_text(self) -> None:
        service, mock_engine = _service_with_mock()
        service.patient_to_doctor("నాకు జ్వరం", "te")
        prompt = mock_engine.generate.call_args.args[0]
        assert "నాకు జ్వరం" in prompt

    def test_prompt_contains_language_display_name(self) -> None:
        service, mock_engine = _service_with_mock()
        service.patient_to_doctor("some text", "kn")
        prompt = mock_engine.generate.call_args.args[0]
        assert LANGUAGE_DISPLAY["kn"] in prompt   # "Kannada"

    def test_uses_fast_translation_mode(self) -> None:
        from core.engine import InferenceMode
        service, mock_engine = _service_with_mock()
        service.patient_to_doctor("text", "hi")
        mode = mock_engine.generate.call_args.kwargs["mode"]
        assert mode == InferenceMode.FAST_TRANSLATION

    @pytest.mark.parametrize("lang_code", ["hi", "te", "kn", "en", "ta"])
    def test_all_supported_languages_resolve_display_name(self, lang_code: str) -> None:
        service, mock_engine = _service_with_mock()
        service.patient_to_doctor("test", lang_code)
        prompt = mock_engine.generate.call_args.args[0]
        assert LANGUAGE_DISPLAY[lang_code] in prompt


class TestDoctorToPatient:

    def test_returns_engine_output(self) -> None:
        service, _ = _service_with_mock("రోజుకు రెండుసార్లు మాత్ర వేసుకోండి")
        result = service.doctor_to_patient("Take tablet twice a day.", "te")
        assert "రోజుకు" in result

    def test_prompt_contains_doctor_text(self) -> None:
        service, mock_engine = _service_with_mock()
        service.doctor_to_patient("Rest and drink fluids.", "hi")
        prompt = mock_engine.generate.call_args.args[0]
        assert "Rest and drink fluids." in prompt

    def test_prompt_contains_target_language_name(self) -> None:
        service, mock_engine = _service_with_mock()
        service.doctor_to_patient("You have a fever.", "ta")
        prompt = mock_engine.generate.call_args.args[0]
        assert LANGUAGE_DISPLAY["ta"] in prompt   # "Tamil"

    def test_uses_fast_translation_mode(self) -> None:
        from core.engine import InferenceMode
        service, mock_engine = _service_with_mock()
        service.doctor_to_patient("Take rest.", "te")
        mode = mock_engine.generate.call_args.kwargs["mode"]
        assert mode == InferenceMode.FAST_TRANSLATION

    @pytest.mark.parametrize("lang_code", ["hi", "te", "kn", "ta"])
    def test_indic_languages_get_correct_display_name(self, lang_code: str) -> None:
        service, mock_engine = _service_with_mock()
        service.doctor_to_patient("Take medicine.", lang_code)
        prompt = mock_engine.generate.call_args.args[0]
        assert LANGUAGE_DISPLAY[lang_code] in prompt


class TestPromptTemplates:

    def test_patient_prompt_has_required_placeholders(self) -> None:
        filled = PATIENT_TO_DOCTOR_PROMPT.format(language="Hindi", text="hello")
        assert "Hindi" in filled
        assert "hello" in filled

    def test_doctor_prompt_has_required_placeholders(self) -> None:
        filled = DOCTOR_TO_PATIENT_PROMPT.format(language="Telugu", text="rest now")
        assert "Telugu" in filled
        assert "rest now" in filled

    def test_patient_prompt_instructs_no_diagnosis(self) -> None:
        assert "diagnos" in PATIENT_TO_DOCTOR_PROMPT.lower()

    def test_doctor_prompt_instructs_preserve_dosage(self) -> None:
        assert "dosage" in DOCTOR_TO_PATIENT_PROMPT.lower() or \
               "dose" in DOCTOR_TO_PATIENT_PROMPT.lower() or \
               "timing" in DOCTOR_TO_PATIENT_PROMPT.lower()
