"""
tests/conftest.py

Pytest runs this file before any test module is imported.

Stubs two native dependencies that cannot be loaded in sandboxed CI:

1. sounddevice  — requires libportaudio (native .so)
2. ollama       — may fail in sandboxed networks with SOCKS proxy configured

Per-test monkeypatching can replace these stubs with finer-grained mocks.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub_sounddevice() -> None:
    if "sounddevice" in sys.modules:
        return
    sd = types.ModuleType("sounddevice")
    sd.InputStream = MagicMock(name="sounddevice.InputStream")
    sd.query_devices = MagicMock(return_value="(sounddevice stubbed)")
    sd.CallbackFlags = type("CallbackFlags", (), {"__bool__": lambda self: False})()
    sys.modules["sounddevice"] = sd


def _stub_ollama() -> None:
    """
    Stub ollama at collection time.

    In sandboxed environments the real ollama package tries to create an
    httpx.Client that reads the system proxy, which may fail with a SOCKS
    import error.  We substitute a minimal stub so test modules that import
    core.engine can be collected without a live Ollama daemon.

    Individual tests inject a MagicMock client via engine._client so the
    actual generate() logic is still exercised.
    """
    if "ollama" in sys.modules:
        # If the real package loaded successfully, leave it alone.
        try:
            import ollama as _real
            _real.Client  # verify it's usable
            return
        except Exception:
            pass

    ol = types.ModuleType("ollama")

    class _StubClient:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): return {"response": ""}
        def list(self): return {"models": []}

    ol.Client = _StubClient
    sys.modules["ollama"] = ol


_stub_sounddevice()
_stub_ollama()
