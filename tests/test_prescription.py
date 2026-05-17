"""
tests/test_prescription.py

Unit tests for translation/prescription.py — PrescriptionService.
All Ollama / GemmaEngine calls are mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from translation.prescription import (
    PrescriptionService,
    _normalise_medicines,
    _parse_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_returning(text: str) -> MagicMock:
    eng = MagicMock()
    eng.transcribe_prescription.return_value = text
    return eng


VALID_JSON = json.dumps({
    "medicines": [
        {
            "name": "Paracetamol",
            "dosage": "500mg",
            "form": "tablet",
            "frequency": "twice daily",
            "duration": "5 days",
            "instructions": "after meals",
        },
        {
            "name": "Amoxicillin",
            "dosage": "250mg",
            "form": "capsule",
            "frequency": "three times daily",
            "duration": "7 days",
            "instructions": "with water",
        },
    ],
    "doctor_name": "Dr. Sharma",
    "patient_name": "Ravi Kumar",
    "date": "2025-05-10",
    "notes": "Rest and drink fluids",
})


# ---------------------------------------------------------------------------
# _parse_json tests
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_clean_json(self):
        obj = _parse_json(VALID_JSON)
        assert obj["doctor_name"] == "Dr. Sharma"

    def test_json_with_markdown_fence(self):
        fenced = "```json\n" + VALID_JSON + "\n```"
        obj = _parse_json(fenced)
        assert obj["patient_name"] == "Ravi Kumar"

    def test_json_embedded_in_text(self):
        raw = "Here is the result:\n" + VALID_JSON + "\n\nDone."
        obj = _parse_json(raw)
        assert "medicines" in obj

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_json("Not JSON at all.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_json("")

    def test_partial_json_fallback(self):
        partial = '{"doctor_name": "Dr. X", "medicines": []}'
        obj = _parse_json(partial)
        assert obj["doctor_name"] == "Dr. X"


# ---------------------------------------------------------------------------
# _normalise_medicines tests
# ---------------------------------------------------------------------------

class TestNormaliseMedicines:
    def test_valid_list(self):
        raw = [{"name": "Aspirin", "dosage": "100mg", "form": "tablet",
                "frequency": "once daily", "duration": "30 days", "instructions": "after meals"}]
        result = _normalise_medicines(raw)
        assert len(result) == 1
        assert result[0]["name"] == "Aspirin"

    def test_missing_fields_get_defaults(self):
        raw = [{"name": "Ibuprofen"}]
        result = _normalise_medicines(raw)
        assert result[0]["dosage"] == "not specified"
        assert result[0]["form"] == "other"
        assert result[0]["frequency"] == "not specified"
        assert result[0]["duration"] == "not specified"
        assert result[0]["instructions"] == "none"

    def test_not_a_list_returns_empty(self):
        assert _normalise_medicines(None) == []
        assert _normalise_medicines("bad") == []
        assert _normalise_medicines(42) == []

    def test_non_dict_items_skipped(self):
        raw = [{"name": "A", "dosage": "10mg", "form": "tablet",
                "frequency": "x", "duration": "x", "instructions": "x"},
               "not a dict",
               42]
        result = _normalise_medicines(raw)
        assert len(result) == 1
        assert result[0]["name"] == "A"

    def test_empty_list(self):
        assert _normalise_medicines([]) == []

    def test_multiple_medicines(self):
        raw = [
            {"name": f"Med{i}", "dosage": f"{i*100}mg", "form": "tablet",
             "frequency": "once", "duration": "5d", "instructions": "none"}
            for i in range(5)
        ]
        result = _normalise_medicines(raw)
        assert len(result) == 5
        assert result[3]["name"] == "Med3"


# ---------------------------------------------------------------------------
# PrescriptionService.extract() tests
# ---------------------------------------------------------------------------

class TestPrescriptionService:
    def test_extract_returns_prescription_result(self):
        eng = _engine_returning(VALID_JSON)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/rx.jpg")

        assert result["doctor_name"] == "Dr. Sharma"
        assert result["patient_name"] == "Ravi Kumar"
        assert result["date"] == "2025-05-10"
        assert result["notes"] == "Rest and drink fluids"
        assert len(result["medicines"]) == 2

    def test_first_medicine_fields(self):
        eng = _engine_returning(VALID_JSON)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/rx.jpg")
        med = result["medicines"][0]
        assert med["name"] == "Paracetamol"
        assert med["dosage"] == "500mg"
        assert med["form"] == "tablet"
        assert med["frequency"] == "twice daily"
        assert med["duration"] == "5 days"
        assert med["instructions"] == "after meals"

    def test_engine_called_with_image_path(self):
        eng = _engine_returning(VALID_JSON)
        svc = PrescriptionService(eng)
        svc.extract("/some/path/image.png")
        eng.transcribe_prescription.assert_called_once_with("/some/path/image.png")

    def test_invalid_json_raises_value_error(self):
        eng = _engine_returning("Sorry, I cannot read this.")
        svc = PrescriptionService(eng)
        with pytest.raises(ValueError):
            svc.extract("/fake/rx.jpg")

    def test_runtime_error_propagates(self):
        eng = MagicMock()
        eng.transcribe_prescription.side_effect = RuntimeError("Ollama down")
        svc = PrescriptionService(eng)
        with pytest.raises(RuntimeError, match="Ollama down"):
            svc.extract("/fake/rx.jpg")

    def test_empty_medicines_list(self):
        payload = json.dumps({
            "medicines": [],
            "doctor_name": "Dr. X",
            "patient_name": "not visible",
            "date": "not visible",
            "notes": "",
        })
        eng = _engine_returning(payload)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/rx.jpg")
        assert result["medicines"] == []
        assert result["doctor_name"] == "Dr. X"

    def test_missing_top_level_fields_get_defaults(self):
        payload = json.dumps({"medicines": []})
        eng = _engine_returning(payload)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/rx.jpg")
        assert result["doctor_name"] == "not visible"
        assert result["patient_name"] == "not visible"
        assert result["date"] == "not visible"
        assert result["notes"] == ""

    def test_markdown_fenced_response(self):
        fenced = "```json\n" + VALID_JSON + "\n```"
        eng = _engine_returning(fenced)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/rx.jpg")
        assert result["doctor_name"] == "Dr. Sharma"

    def test_six_medicines(self):
        meds = [
            {"name": f"Drug{i}", "dosage": f"{i*50}mg", "form": "tablet",
             "frequency": "once daily", "duration": "10 days", "instructions": "none"}
            for i in range(6)
        ]
        payload = json.dumps({
            "medicines": meds,
            "doctor_name": "Dr. Multi",
            "patient_name": "Patient X",
            "date": "2025-01-01",
            "notes": "",
        })
        eng = _engine_returning(payload)
        svc = PrescriptionService(eng)
        result = svc.extract("/fake/complex.jpg")
        assert len(result["medicines"]) == 6
        assert result["medicines"][5]["name"] == "Drug5"
