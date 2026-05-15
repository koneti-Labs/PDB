# MANDATORY Constraints for PatientDoctorBridge

## Model Rule — GEMMA 4 ONLY

**NEVER select, reference, or suggest any model other than GEMMA 4.** This is a hard competition rule for the Kaggle Gemma 4 Impact Challenge. All LLM inference in this project MUST use Gemma 4 variants via Ollama:

- **`gemma4:e2b`** — Fast translation mode (`InferenceMode.FAST_TRANSLATION`). Used for:
  - Patient→Doctor translation
  - Doctor→Patient translation
  - Emergency reassurance translation
  - Prescription summary translation
- **`gemma4:e4b`** — Reasoning / extraction / vision mode (`InferenceMode.REASONING_EXTRACTION`). Used for:
  - Emergency triage extraction (structured JSON)
  - Prescription OCR via multimodal vision endpoint (with `images` param)
  - Any task requiring `think=True` (extended reasoning)

The model mapping is defined in `core/engine.py` as:
```python
MODELS = {
    InferenceMode.FAST_TRANSLATION: 'gemma4:e2b',
    InferenceMode.REASONING_EXTRACTION: 'gemma4:e4b',
}
```

**Do NOT:**
- Replace these with Llama, Mistral, GPT, Claude, or ANY other model
- Add alternative models as fallbacks
- Suggest benchmarking against non-Gemma models
- Change the model tags to anything other than `gemma4:e2b` and `gemma4:e4b`

## Python Version — 3.11 ONLY

**Use Python 3.11, NOT 3.12.** The TTS dependency (Coqui XTTS v2, `TTS==0.22.0`) only supports Python 3.11. All venvs, CI, and deployment scripts must target 3.11:

- Virtual env: `python3.11 -m venv .venv`
- CI (`.github/workflows/ci.yml`): `python-version: '3.11'`
- Pi 5 setup: `sudo apt install python3.11 python3.11-venv python3.11-dev`
- `pyproject.toml` already has `target-version = 'py311'` for ruff

## Kaggle Gemma 4 Impact Challenge Rules

- All inference must run **locally** — nothing leaves the device
- Must use **Gemma 4** model family exclusively
- Privacy-first: no cloud API calls for inference
- Audio is deleted immediately after transcription (`os.unlink` in `AudioHandler.transcribe`)
- Session data is memory-only, wiped on `end_session()`
- The project targets both laptop (Phase 1–6) and Raspberry Pi 5 + Hailo-10H NPU (Phase 7)
- TTFT target on Pi 5: < 500ms (benchmarked via `scripts/benchmark_ttft.py`)

## Performance Knobs (config/settings.py)

All performance-related knobs live in one place. Tune here, not in call sites.

Ollama / Gemma 4:
- `OLLAMA_TIMEOUT` — fast translation HTTP timeout (seconds)
- `OLLAMA_TIMEOUT_REASONING` — triage / OCR HTTP timeout
- `GEMMA_KEEP_ALIVE` — how long Ollama keeps the model resident (e.g. `"30m"`)
- `GEMMA_NUM_CTX` — context window for reasoning / OCR
- `GEMMA_NUM_CTX_FAST` — smaller context window for translation
- `GEMMA_NUM_PREDICT_FAST` — output token cap for translation
- `GEMMA_NUM_PREDICT_REASONING` — output token cap for triage / OCR
- `GEMMA_TEMPERATURE`, `GEMMA_TOP_P` — sampling defaults

Whisper:
- `WHISPER_MODEL_SIZE` — `tiny` / `base` / `small` / …
- `WHISPER_BEAM_SIZE` — 1 = greedy (fastest)
- `WHISPER_VAD_FILTER` — pre-trim silence
- `WHISPER_CONDITION_ON_PREVIOUS` — keep `False` for short clinical utterances

## Server Startup Contract

`pdb server` (in `web/server.py::run_server`) MUST warm up both Whisper and both Gemma 4 models before `app.run()` so the first user request is not a cold start. Do not remove the warmup.
