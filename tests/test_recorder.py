"""
tests/test_recorder.py

Unit tests for audio/recorder.py -- synthetic audio, no real microphone.

sounddevice is stubbed globally by tests/conftest.py before this module
is imported, so collection always succeeds without PortAudio.
Audio is fed via a background thread (after a small delay) to match real
sounddevice async behaviour and ensure recording_active is set first.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from audio.recorder import Recorder
from config.settings import MIN_RECORDING_SECONDS, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(duration_s: float, freq: float = 440.0) -> np.ndarray:
    """Return a synthetic sine wave as int16 mono (n_samples, 1)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16).reshape(-1, 1)


def _make_fake_stream(callback_holder: list, audio: np.ndarray, chunk_size: int = 1024):
    """
    Factory returning a context-manager fake InputStream.

    Audio is fed in a background thread after a short delay so that
    recording_active.set() in recorder.py is called before any
    data arrives -- matching real sounddevice async behaviour.
    """

    class FakeStream:
        def __init__(self, **kwargs):
            callback_holder.append(kwargs.get("callback"))

        def __enter__(self):
            cb = callback_holder[-1]

            def _feed() -> None:
                time.sleep(0.15)   # let recording_active.set() happen first
                if cb:
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i : i + chunk_size]
                        if len(chunk):
                            cb(chunk, len(chunk), None, None)

            threading.Thread(target=_feed, daemon=True).start()
            return self

        def __exit__(self, *_):
            pass

    return FakeStream


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecorder:

    def test_record_returns_path_on_adequate_duration(self, tmp_path, monkeypatch) -> None:
        """A recording of adequate length returns a Path to an existing WAV file."""
        recorder = Recorder()
        audio = _sine_wave(3.0)
        cb_holder: list = []

        FakeStream = _make_fake_stream(cb_holder, audio)
        monkeypatch.setattr("audio.recorder.sd.InputStream", FakeStream)
        monkeypatch.setattr("audio.recorder.AUDIO_TEMP_DIR", tmp_path)

        enter_count = [0]

        def fake_input():
            enter_count[0] += 1
            if enter_count[0] == 2:
                time.sleep(2.5)   # simulate user speaking for 2.5 s then pressing Enter

        monkeypatch.setattr("builtins.input", fake_input)

        result = recorder.record()

        assert result is not None, "Expected a valid Path from a 3-second recording"
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".wav"
        result.unlink(missing_ok=True)

    def test_file_is_in_temp_dir(self, tmp_path, monkeypatch) -> None:
        """The returned WAV file lives inside AUDIO_TEMP_DIR."""
        recorder = Recorder()
        audio = _sine_wave(3.0)
        cb_holder: list = []

        FakeStream = _make_fake_stream(cb_holder, audio)
        monkeypatch.setattr("audio.recorder.sd.InputStream", FakeStream)
        monkeypatch.setattr("audio.recorder.AUDIO_TEMP_DIR", tmp_path)

        enter_count = [0]

        def fake_input():
            enter_count[0] += 1
            if enter_count[0] == 2:
                time.sleep(2.5)

        monkeypatch.setattr("builtins.input", fake_input)

        result = recorder.record()
        assert result is not None
        assert str(result).startswith(str(tmp_path))
        result.unlink(missing_ok=True)

    def test_short_recording_returns_none(self, tmp_path, monkeypatch) -> None:
        """Clips shorter than MIN_RECORDING_SECONDS return None."""
        recorder = Recorder()
        audio = _sine_wave(0.1)
        cb_holder: list = []

        FakeStream = _make_fake_stream(cb_holder, audio)
        monkeypatch.setattr("audio.recorder.sd.InputStream", FakeStream)
        monkeypatch.setattr("audio.recorder.AUDIO_TEMP_DIR", tmp_path)

        enter_count = [0]

        def fake_input():
            enter_count[0] += 1
            # Both presses return immediately -> elapsed ~ 0 s

        monkeypatch.setattr("builtins.input", fake_input)

        result = recorder.record()
        assert result is None, f"Expected None for a sub-{MIN_RECORDING_SECONDS}s recording"

    def test_empty_audio_buffer_returns_none(self, tmp_path, monkeypatch) -> None:
        """record() returns None when the audio callback is never invoked."""
        recorder = Recorder()

        class EmptyStream:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self   # never feeds any data

            def __exit__(self, *_):
                pass

        monkeypatch.setattr("audio.recorder.sd.InputStream", EmptyStream)
        monkeypatch.setattr("audio.recorder.AUDIO_TEMP_DIR", tmp_path)

        enter_count = [0]

        def fake_input():
            enter_count[0] += 1
            if enter_count[0] == 2:
                time.sleep(2.5)

        monkeypatch.setattr("builtins.input", fake_input)

        result = recorder.record()
        assert result is None
