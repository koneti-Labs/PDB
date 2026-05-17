"""
tests/test_handler.py

Unit tests for audio/handler.py — specifically the privacy / deletion contract.

The golden rule: after transcribe() returns OR raises, the audio file MUST be gone.
These tests enforce that guarantee without touching a real Whisper model.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io.wavfile import write as wav_write

from audio.handler import AudioHandler
from config.settings import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_wav(duration_s: float = 3.0) -> Path:
    """Write a synthetic sine-wave WAV to a temp file and return its Path."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, SAMPLE_RATE, audio)
    tmp.close()
    return Path(tmp.name)


def _make_handler_with_mock_model(
    monkeypatch, probs: dict, transcript: str = "test",
) -> AudioHandler:
    """
    Build an AudioHandler whose _model is fully mocked so no Whisper
    download or inference occurs.
    """
    handler = AudioHandler()

    # Patch _ensure_model so it doesn't trigger a download
    monkeypatch.setattr(handler, "_ensure_model", lambda: None)

    # Patch _detect_language_probs to return a controlled distribution
    monkeypatch.setattr(
        handler,
        "_detect_language_probs",
        lambda path: probs,
    )

    # Build a minimal fake model whose transcribe() returns an iterable of segments
    class FakeSegment:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def transcribe(self, path: str, **kwargs):
            return [FakeSegment(transcript)], None

    handler._model = FakeModel()
    return handler


# ---------------------------------------------------------------------------
# Deletion guarantee tests  ← the core of the privacy contract
# ---------------------------------------------------------------------------

class TestDeletionContract:

    def test_file_deleted_after_successful_transcribe(self, monkeypatch) -> None:
        """The audio file is gone after a successful transcription."""
        handler = _make_handler_with_mock_model(
            monkeypatch,
            probs={"hi": 0.8, "en": 0.1, "te": 0.05, "kn": 0.03, "ta": 0.02},
            transcript="नमस्ते",
        )
        path = _write_temp_wav()

        assert path.exists(), "Precondition: file must exist before transcribe()"
        handler.transcribe(path)
        assert not path.exists(), "Audio file MUST be deleted after successful transcribe()"

    def test_file_deleted_even_on_transcription_error(self, monkeypatch) -> None:
        """The audio file is deleted even when transcription raises an exception."""
        handler = AudioHandler()
        monkeypatch.setattr(handler, "_ensure_model", lambda: None)
        monkeypatch.setattr(
            handler,
            "_detect_language_probs",
            lambda path: {"hi": 0.9},
        )

        class BrokenModel:
            def transcribe(self, *args, **kwargs):
                raise RuntimeError("simulated transcription failure")

        handler._model = BrokenModel()
        path = _write_temp_wav()

        assert path.exists()
        with pytest.raises(RuntimeError, match="simulated transcription failure"):
            handler.transcribe(path)
        assert not path.exists(), "Audio file MUST be deleted even when transcription raises"

    def test_file_deleted_when_language_detection_errors(self, monkeypatch) -> None:
        """The audio file is deleted even if _detect_language_probs() raises."""
        handler = AudioHandler()
        monkeypatch.setattr(handler, "_ensure_model", lambda: None)
        monkeypatch.setattr(
            handler,
            "_detect_language_probs",
            lambda path: (_ for _ in ()).throw(RuntimeError("lang detection exploded")),
        )
        handler._model = MagicMock()
        path = _write_temp_wav()

        assert path.exists()
        with pytest.raises(RuntimeError, match="lang detection exploded"):
            handler.transcribe(path)
        assert not path.exists(), "Audio file MUST be deleted even when language detection raises"


# ---------------------------------------------------------------------------
# Pre-condition check
# ---------------------------------------------------------------------------

class TestPreConditions:

    def test_transcribe_raises_file_not_found_if_path_missing(self, monkeypatch) -> None:
        """FileNotFoundError is raised if the audio path does not exist."""
        handler = AudioHandler()
        # Attach a dummy model so _ensure_model won't download anything
        handler._model = MagicMock()
        monkeypatch.setattr(handler, "_ensure_model", lambda: None)

        ghost_path = Path(tempfile.gettempdir()) / "pdb_nonexistent_test_file.wav"
        ghost_path.unlink(missing_ok=True)

        with pytest.raises(FileNotFoundError):
            handler.transcribe(ghost_path)


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------

class TestReturnValue:

    def test_result_contains_expected_keys(self, monkeypatch) -> None:
        """TranscriptResult has language, confidence, and text."""
        handler = _make_handler_with_mock_model(
            monkeypatch,
            probs={"ta": 0.9},
            transcript="வணக்கம்",
        )
        path = _write_temp_wav()
        result = handler.transcribe(path)

        assert "language" in result
        assert "confidence" in result
        assert "text" in result

    def test_result_language_is_in_supported_set(self, monkeypatch) -> None:
        from config.languages import SUPPORTED_LANG_CODES

        handler = _make_handler_with_mock_model(
            monkeypatch,
            # Simulate Whisper detecting Urdu — constraint must pull it to Hindi
            probs={"ur": 0.7, "hi": 0.2, "en": 0.1},
            transcript="नमस्ते",
        )
        path = _write_temp_wav()
        result = handler.transcribe(path)

        assert result["language"] in SUPPORTED_LANG_CODES

    def test_empty_transcription_returns_placeholder(self, monkeypatch) -> None:
        """Empty transcript is replaced with '[no speech detected]'."""
        handler = AudioHandler()
        monkeypatch.setattr(handler, "_ensure_model", lambda: None)
        monkeypatch.setattr(handler, "_detect_language_probs", lambda p: {"en": 1.0})

        class SilentModel:
            def transcribe(self, *args, **kwargs):
                return [], None   # no segments

        handler._model = SilentModel()
        path = _write_temp_wav(0.5)    # very short silence

        # Need to manually write enough duration for the file to exist
        path = _write_temp_wav(3.0)
        result = handler.transcribe(path)
        assert result["text"] == "[no speech detected]"
