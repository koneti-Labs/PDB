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
OLLAMA_TIMEOUT: int = 120            # seconds -- fast translation (gemma4:e2b)
OLLAMA_TIMEOUT_REASONING: int = 600  # seconds -- triage/OCR (vision is slow on CPU)

# Web server (Phase 6 UI)
WEB_HOST: str = "127.0.0.1"
WEB_PORT: int = 5000

# ---------------------------------------------------------------------------
# Gemma generation defaults (override per call if needed)
# ---------------------------------------------------------------------------
GEMMA_TEMPERATURE: float = 0.2     # low = more deterministic translations
GEMMA_TOP_P: float = 0.9
GEMMA_NUM_CTX: int = 8192          # full context window for reasoning/OCR (raised to fit image + JSON output)
GEMMA_NUM_CTX_FAST: int = 2048     # small window for translation/reassure (raised for longer Indic sentences)

# Output length caps -- prevents the model from running off and producing
# multi-paragraph trailing chatter that adds seconds per request.
GEMMA_NUM_PREDICT_FAST: int = 512        # patient/doctor utterances (raised from 256 for longer sentences)
GEMMA_NUM_PREDICT_REASONING: int = 4096  # triage JSON + prescription OCR (raised from 1024 to prevent truncation)

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
