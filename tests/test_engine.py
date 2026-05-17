"""
tests/test_engine.py

Unit tests for core/engine.py -- GemmaEngine.

Ollama is mocked at the call site so no Ollama daemon is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.engine import MODELS, GemmaEngine, InferenceMode


def _engine_with_mock(response_text: str = "test response") -> tuple[GemmaEngine, MagicMock]:
    engine = GemmaEngine()
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": f"  {response_text}  "}
    engine._client = mock_client
    # Also wire the reasoning client so REASONING_EXTRACTION calls are captured
    engine._reasoning_client = mock_client
    return engine, mock_client


class TestModelRouting:

    def test_fast_translation_uses_e2b(self) -> None:
        engine, mock_client = _engine_with_mock()
        engine.generate("hello", mode=InferenceMode.FAST_TRANSLATION)
        assert mock_client.generate.call_args.kwargs["model"] == "gemma4:e2b"

    def test_reasoning_extraction_uses_4b(self) -> None:
        engine, mock_client = _engine_with_mock()
        engine.generate("hello", mode=InferenceMode.REASONING_EXTRACTION)
        expected = MODELS[InferenceMode.REASONING_EXTRACTION]
        assert mock_client.generate.call_args.kwargs["model"] == expected

    def test_models_dict_only_contains_gemma4(self) -> None:
        for tag in MODELS.values():
            assert tag.startswith("gemma4:"), f"Non-Gemma model tag: {tag}"

    def test_default_mode_is_fast_translation(self) -> None:
        engine, mock_client = _engine_with_mock()
        engine.generate("hello")
        expected = MODELS[InferenceMode.FAST_TRANSLATION]
        assert mock_client.generate.call_args.kwargs["model"] == expected


class TestResponseHandling:

    def test_response_is_stripped(self) -> None:
        engine, _ = _engine_with_mock("  trimmed  ")
        assert engine.generate("prompt") == "trimmed"

    def test_ollama_exception_raises_runtime_error(self) -> None:
        engine, mock_client = _engine_with_mock()
        mock_client.generate.side_effect = Exception("model failed")
        with pytest.raises(RuntimeError, match="Ollama inference failed"):
            engine.generate("prompt")

    def test_error_message_includes_model_name(self) -> None:
        engine, mock_client = _engine_with_mock()
        mock_client.generate.side_effect = Exception("boom")
        with pytest.raises(RuntimeError) as exc_info:
            engine.generate("prompt", mode=InferenceMode.FAST_TRANSLATION)
        assert "gemma4:e2b" in str(exc_info.value)

    def test_temperature_override_passed_through(self) -> None:
        engine, mock_client = _engine_with_mock()
        engine.generate("prompt", temperature=0.9)
        assert mock_client.generate.call_args.kwargs["options"]["temperature"] == 0.9

    def test_num_ctx_override_passed_through(self) -> None:
        engine, mock_client = _engine_with_mock()
        engine.generate("prompt", num_ctx=8192)
        assert mock_client.generate.call_args.kwargs["options"]["num_ctx"] == 8192


class TestPhaseStubs:

    def test_emergency_triage_implemented_phase3(self) -> None:
        """emergency_triage() is implemented in Phase 3."""
        import json
        engine, mock_client = _engine_with_mock()
        mock_client.generate.return_value = {"response": json.dumps({
            "chief_complaint": "chest pain",
            "severity": "severe",
            "duration": "2 hours",
            "symptoms": ["chest pain"],
            "vitals_mentioned": [],
            "needs_immediate_attention": True,
        })}
        result = engine.emergency_triage("chest pain", "en")
        assert isinstance(result, str)
        assert "chest pain" in result

    def test_transcribe_prescription_implemented_phase4(self) -> None:
        """transcribe_prescription() is implemented in Phase 4 via multimodal vision."""
        import os
        import tempfile
        # Create a tiny 1-pixel PNG to avoid file-not-found
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        fd, tmp = tempfile.mkstemp(suffix=".png")
        try:
            os.write(fd, tiny_png)
            os.close(fd)
            engine, mock_client = _engine_with_mock()
            import json
            mock_client.generate.return_value = {"response": json.dumps({
                "medicines": [],
                "doctor_name": "Dr. X",
                "patient_name": "not visible",
                "date": "not visible",
                "notes": "",
            })}
            result = engine.transcribe_prescription(tmp)
            assert isinstance(result, str)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_emergency_reassurance_implemented_phase5(self) -> None:
        """emergency_reassurance() is implemented in Phase 5."""
        engine, mock_client = _engine_with_mock("मदद आ रही है।")
        result = engine.emergency_reassurance("Help is coming.", "hi")
        assert isinstance(result, str)
        assert len(result) > 0


class TestConnectivityCheck:

    def test_returns_dict_keyed_by_model_tag(self) -> None:
        engine, _ = _engine_with_mock()
        engine._client.list.return_value = {
            "models": [
                {"model": "gemma4:e2b"},
                {"model": MODELS[InferenceMode.REASONING_EXTRACTION]},
            ]
        }
        result = engine.check_connectivity()
        assert set(result.keys()) == set(MODELS.values())

    def test_all_false_when_ollama_unreachable(self) -> None:
        engine, _ = _engine_with_mock()
        engine._client.list.side_effect = Exception("refused")
        result = engine.check_connectivity()
        assert isinstance(result, dict)
        assert all(v is False for v in result.values())
