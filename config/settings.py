"""
config/settings.py

All tunable knobs in one place.  Import from here — never hard-code values
in audio/, cli/, or core/.

Pi 5 migration notes are inline so they're never lost.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Whisper ASR model
# ---------------------------------------------------------------------------
# "small" (244 M params, ~1 GB RAM) is the minimum for reliable Indic accuracy.
#   • Downgrade to "base" for Pi 5 if TTFT > 500 ms.
#   • Upgrade to "medium" if Telugu / Kannada accuracy is poor in Phase 1 eval.
# WHISPER_MODEL_SIZE: str = "small"
WHISPER_MODEL_SIZE: str = "base"

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000   # Hz -- Whisper's native expected format, 16 kHz mono
CHANNELS: int = 1           # Mono

MAX_RECORDING_SECONDS: int = 30   # Hard upper ceiling; CLI enforces this
MIN_RECORDING_SECONDS: float = 2.0  # Reject shorter clips -- language ID unreliable

# ---------------------------------------------------------------------------
# Whisper inference knobs (speed vs. accuracy trade-off)
# ---------------------------------------------------------------------------
# beam_size=1 gives greedy decoding -- ~4-5x faster than beam_size=5 on CPU.
# For GPU or Pi 5 + NPU, increase to 3-5 for better accuracy.
WHISPER_BEAM_SIZE: int = 1

# VAD pre-filters silence before decoding -- saves time on recordings with
# lots of silence; adds ~100ms overhead on dense speech. Set False to disable.
WHISPER_VAD_FILTER: bool = True

# Conditioning on previous segment text adds minor latency with little benefit
# for single-sentence clinical utterances.
WHISPER_CONDITION_ON_PREVIOUS: bool = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Temp audio files go here — always inside the OS temp dir, never user-visible.
AUDIO_TEMP_DIR: Path = Path(tempfile.gettempdir())

# Whisper model download cache — persists across runs so first-run is the only
# slow download (~500 MB for "small").
MODEL_CACHE_DIR: Path = Path.home() / ".cache" / "pdb" / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama (Phase 2+) — local inference, nothing leaves the device
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = "http://localhost:11434"
OLLAMA_TIMEOUT: int = 180            # seconds -- fast translation (gemma4:e2b)
OLLAMA_TIMEOUT_REASONING: int = 900  # seconds -- triage/OCR (vision is slow on CPU)

# Web server (Phase 6 UI)
WEB_HOST: str = "127.0.0.1"
WEB_PORT: int = 5000

# ---------------------------------------------------------------------------
# Gemma generation defaults (override per call if needed)
# ---------------------------------------------------------------------------
GEMMA_TEMPERATURE: float = 0.2     # low = more deterministic translations
GEMMA_TOP_P: float = 0.9
# full context window for reasoning/OCR (raised to fit image + JSON output)
GEMMA_NUM_CTX: int = 8192
GEMMA_NUM_CTX_FAST: int = 2048     # context window for translation/reassure.
                                   # Root-cause fix: 512 was too small for gemma4:e2b — the model
                                   # was truncating context and returning empty output.  2048 gives
                                   # comfortable headroom for all prompt + response combinations.

# Output length caps -- prevents the model from running off and producing
# multi-paragraph trailing chatter that adds seconds per request.
# raised from 256; accommodates longer Indic translations
GEMMA_NUM_PREDICT_FAST: int = 512
# triage JSON + prescription OCR (raised from 1024 to prevent truncation)
GEMMA_NUM_PREDICT_REASONING: int = 4096

# Hold the model in (V)RAM between calls.  Without this Ollama unloads
# after ~5 minutes idle (its default), and the next request pays a ~20-60s
# cold-load penalty.  Set to None to use Ollama's default, or "0" to unload
# immediately after each call.
GEMMA_KEEP_ALIVE: str = "30m"

# ---------------------------------------------------------------------------
# Triage extraction reasoning mode
# ---------------------------------------------------------------------------
# True  -> Gemma 4 emits an internal <think>...</think> reasoning trace before
#          the JSON.  More accurate severity assessment but adds 10-30s on CPU.
# False -> Direct JSON, no thinking step.  ~3-5x faster end-to-end; suitable
#          when TTFT matters more than the absolute best severity call (dev,
#          Pi 5 deployment, or when the prompt itself is already strong).
#
# Imported by core/engine.py::emergency_triage().  Flip to False for the
# fastest path; keep True for the most defensible triage decision.
TRIAGE_THINK_MODE: bool = True

# ---------------------------------------------------------------------------
# Concurrency / thread-pool settings
# ---------------------------------------------------------------------------

# Maximum number of OS threads dedicated to blocking inference calls
# (Whisper, Ollama HTTP, PyTorch TTS).  All three release the GIL during
# their C-extension work, so multiple threads DO run in true parallel on
# multi-core hardware.
#
# Rule of thumb:
#   • Laptop / competition (1 user): 2  — one for Whisper, one for Ollama
#   • Clinic (3-5 users):            4  — keeps Ollama queue short
#   • Pi 5 (memory-limited):         2  — RAM is the ceiling, not CPU
#
# Used by the module-level _inference_pool in web/server.py.
INFERENCE_POOL_SIZE: int = 3

# Languages to eagerly preload into TTSService._cache at server startup.
# Each model is ~80 MB RAM and takes 5-30 s to download on first run
# (cached to disk afterwards).
#
# Options:
#   []                        — lazy load on first request (default; saves RAM)
#   ["hi", "te", "kn", "ta"] — preload all Indic languages in parallel at startup
#   ["hi", "te", "kn", "ta", "en"] — preload everything (max RAM, zero first-use lag)
#
# Preloading runs in a ThreadPoolExecutor so all languages load simultaneously.
TTS_PREWARM_LANGS: list[str] = []  # change to ["hi","te","kn","ta"] for eager load


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_settings() -> list[str]:
    """Check that settings are reasonable; return a list of warnings (empty = OK)."""
    warnings: list[str] = []

    if OLLAMA_TIMEOUT <= 0:
        warnings.append(f"OLLAMA_TIMEOUT must be positive, got {OLLAMA_TIMEOUT}")
    if OLLAMA_TIMEOUT_REASONING <= 0:
        warnings.append(
            f"OLLAMA_TIMEOUT_REASONING must be positive, got {OLLAMA_TIMEOUT_REASONING}"
        )
    if OLLAMA_TIMEOUT_REASONING < OLLAMA_TIMEOUT:
        warnings.append(
            f"OLLAMA_TIMEOUT_REASONING ({OLLAMA_TIMEOUT_REASONING}) is less than "
            f"OLLAMA_TIMEOUT ({OLLAMA_TIMEOUT}); reasoning tasks need more time"
        )

    # Validate model names match the required gemma4:* pattern
    # (import here to avoid circular dependency at module level)
    try:
        from core.engine import MODELS
        for mode, model_tag in MODELS.items():
            if not model_tag.startswith("gemma4:"):
                warnings.append(
                    f"Model for {mode.value} is '{model_tag}' — "
                    f"only gemma4:* tags are permitted"
                )
    except ImportError:
        pass  # core.engine not available yet during early init

    return warnings


# Run validation at import time so misconfigurations are caught early.
_settings_warnings = validate_settings()
if _settings_warnings:
    import logging as _logging
    _settings_logger = _logging.getLogger(__name__)
    for _w in _settings_warnings:
        _settings_logger.warning("config/settings.py: %s", _w)
