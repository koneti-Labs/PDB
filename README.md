# PatientDoctorBridge

**Kaggle Gemma 4 Impact Challenge entry** — a privacy-first, fully local voice bridge for Indian clinic consultations.

A patient speaks in Hindi, Telugu, Kannada, or Tamil. The doctor hears clean English. The doctor replies in English. The patient hears their own language. Everything runs on-device: no cloud, no API keys, no data ever leaves the room.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pulling the Gemma 4 Models](#pulling-the-gemma-4-models)
- [CLI Usage](#cli-usage)
  - [Phase 1 — Listen & Detect](#phase-1--listen--detect)
  - [Phase 2 — Bridge (Two-way Translation)](#phase-2--bridge-two-way-translation)
  - [Phase 3 — Emergency Triage](#phase-3--emergency-triage)
  - [Phase 4 — Prescription OCR](#phase-4--prescription-ocr)
  - [Phase 5 — Emergency Reassurance](#phase-5--emergency-reassurance)
  - [Phase 6 — Web UI](#phase-6--web-ui)
- [Running the Tests](#running-the-tests)
- [Project Structure](#project-structure)
- [Privacy Contract](#privacy-contract)
- [Competition Constraints](#competition-constraints)

---

## Overview

| Phase | Feature | CLI command | LLM (Gemma 4) | ASR / TTS support models |
|-------|---------|-------------|---------------|--------------------------|
| 1 | Voice capture + language detection | `pdb listen` | — | Whisper or IndicConformer (optional) |
| 2 | Two-way patient ↔ doctor translation | `pdb bridge` | `gemma4:e2b` | Whisper or IndicConformer; MMS-TTS for read-aloud |
| 3 | Emergency triage extraction | `pdb triage` | `gemma4:e4b` | Whisper or IndicConformer |
| 4 | Prescription OCR (multimodal vision) | `pdb prescription --image rx.jpg` | `gemma4:e4b` | — |
| 5 | Emergency reassurance phrases | `pdb reassure` | `gemma4:e2b` | MMS-TTS for read-aloud |
| 6 | Single-page web UI | `pdb server` | both | Per-request choice of ASR backend |

**Supported languages:** Hindi (hi), Telugu (te), Kannada (kn), Tamil (ta), English (en)

**ASR backends available** (selectable per request in the web UI):

- **Whisper** (`faster-whisper`, default) — fast (~140 MB "base" model), built-in language detection, ships with the base install. The Bridge and Triage tabs default to this.
- **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`, optional opt-in) — substantially better on Indian languages, produces native-script output (Devanagari / Telugu / Kannada / Tamil) on the first pass without script-normalisation patches. ~600 MB model, ~2-3× slower per request on CPU. Cannot auto-detect by itself; if the user leaves the language dropdown on **Auto-detect**, the server runs Whisper for cheap language detection and then re-transcribes with IndicConformer locked to the detected language (hybrid auto-detect).

**TTS** uses Facebook MMS-TTS via the local Transformers VITS model — patient-language read-aloud for Doctor→Patient translations, reassurance phrases, and prescription summaries. No OS voice pack or cloud TTS API required.

> **Competition note.** All *LLM* inference is Gemma 4 exclusively. Whisper, IndicConformer, and MMS-TTS are acoustic models (speech-to-text and text-to-speech), not LLMs, and they all run entirely locally — see the [Competition Constraints](#competition-constraints) section.

---

## Architecture

```
Browser / CLI
     │
     ▼
audio/recorder.py        ←  records via sounddevice or browser MediaRecorder
     │
web/server.py            ←  _dispatch_asr() — single source of truth for ASR routing
     │                      based on (asr_backend, locked_lang):
     │                        (whisper,         locked) → AudioHandler.transcribe_locked
     │                        (whisper,         auto)   → AudioHandler.transcribe   (constrained)
     │                        (indic_conformer, locked) → IndicConformerHandler.transcribe
     │                        (indic_conformer, auto)   → Whisper-detect → IndicConformer-transcribe
     │
audio/handler.py         ←  Whisper ASR (auto-detect + constrained renormalisation,
     │                      Hindi-script normaliser, repeat-token hallucination guard)
     │
audio/indic_conformer.py ←  AI4Bharat IndicConformer-600M ASR (optional opt-in)
     │
audio/preprocessor.py    ←  SNR / silence analysis + light noise gate
audio/script_normalizer.py ←  Devanagari ↔ Urdu-script + romanised-Latin recovery
     │                      audio file unconditionally deleted in a finally block
     ▼
config/languages.py      ←  constrains + renormalises to 5 language codes
     │
     ▼
core/engine.py           ←  GemmaEngine — routes to two Ollama clients:
     │                        FAST_TRANSLATION      → gemma4:e2b  (600 s timeout)
     │                        REASONING_EXTRACTION  → gemma4:e4b  (900 s timeout)
     │                      with empty-response retry + raw-output trace logging
     │
translation/
  service.py             ←  patient↔doctor text translation
  triage.py              ←  JSON extraction + TriageResult TypedDict
  prescription.py        ←  medicine list extraction from images
  reassurance.py         ←  comfort phrase bank + translation
     │
audio/tts.py             ←  Facebook MMS-TTS — patient-language read-aloud
     │
cli/                     ←  Rich terminal UI for each phase
web/server.py            ←  Flask API + single-page HTML/JS UI (Bridge / Triage / Rx / Reassure)
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | **3.11 exactly** | Required so the optional TTS dependency (Coqui XTTS v2) compiles; the active MMS-TTS path works under newer Pythons but the project pins 3.11 for reproducibility |
| [Ollama](https://ollama.com) | ≥ 0.4.7 | For local Gemma 4 inference |
| PortAudio | system library | Required by sounddevice for mic access |
| FFmpeg | any recent | Required by faster-whisper for audio decoding |
| Disk space | ≥ 2 GB free | Whisper-base ~140 MB, Gemma 4 weights ~6 GB total, MMS-TTS ~80 MB per language, IndicConformer ~600 MB if you opt into it |

### Install system dependencies

**Ubuntu / Debian:**
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev \
                 portaudio19-dev ffmpeg
```

**macOS (Homebrew):**
```bash
brew install portaudio ffmpeg
```

**Windows:**
Install PortAudio via the `sounddevice` wheel (usually bundled). Install FFmpeg and add it to PATH.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/patient-doctor-bridge.git
cd patient-doctor-bridge

# 2. Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the base package + dev tools
pip install -e ".[dev]"

# 4. (Optional) Opt into the AI4Bharat IndicConformer-600M ASR backend
pip install -e ".[indic-asr]"

# 5. Verify the CLI is available
pdb --help
```

### What the base install includes

The base install (`pip install -e .`) brings in everything needed for the default pipeline:

- `faster-whisper` for ASR
- `ollama` Python client for Gemma 4 calls
- `flask` for the web UI
- `torch` + `transformers` for the **Facebook MMS-TTS** read-aloud feature — these are required for the 🔊 buttons in the Bridge / Reassure / Prescription tabs to work; without them the `/api/tts` endpoint returns a 503 with an install hint and the UI shows a one-time toast.

### What the `[indic-asr]` extra adds

`torchaudio>=2.2,<3` — needed only by the AI4Bharat IndicConformer-600M checkpoint's `trust_remote_code=True` loader. If you never pick the IndicConformer backend in the UI, you can skip this extra.

### Windows note

The MMS-TTS HuggingFace cache uses symlinks by default. On Windows that prints a one-time warning unless you either run Python as administrator or enable [Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development). It's a warning, not an error — files still cache, just with extra disk usage.

---

## Pulling the Gemma 4 Models

Both models must be pulled once before any inference command will work:

```bash
# Fast translation model (~2B params, used for Bridge and Reassurance)
ollama pull gemma4:e2b

# Reasoning + vision model (~4B params, used for Triage and Prescription OCR)
ollama pull gemma4:e4b

# Verify both are available
ollama list
```

Make sure Ollama is running before using any `pdb` command that calls Gemma 4:

```bash
ollama serve
```

---

## CLI Usage

### Phase 1 — Listen & Detect

Records from the microphone, transcribes with Whisper, and detects the language. Audio is deleted immediately after transcription.

```bash
# Start recording (press CTRL+C to stop and transcribe)
pdb listen

# List available audio input devices
pdb listen --device list

# Use a specific device by index
pdb listen --device 2

# Auto-stop after N seconds (useful for testing)
pdb listen --duration 5
```

**Example output:**
```
──────────── PatientDoctorBridge — Phase 1 ────────────
Recording... press CTRL+C to stop.

Transcribing…
──────────────────────────────────────────────────────
Detected language:  Telugu (తెలుగు) [0.87]
Transcript:  నాకు తలనొప్పి మరియు జ్వరం వస్తోంది
──────────────────────────────────────────────────────

Record again? [y/N]
```

---

### Phase 2 — Bridge (Two-way Translation)

Interactive loop that alternates between recording the patient (speech → doctor English) and recording the doctor (English speech → patient language).

```bash
pdb bridge

# Use a specific mic device
pdb bridge --device 2
```

**Workflow:**
1. Press ENTER to record the patient speaking in their language.
2. Gemma 4 (`gemma4:e2b`) translates to clinical English for the doctor.
3. Press ENTER to record the doctor's English reply.
4. Gemma 4 translates back to the patient's detected language.
5. Repeat until done.

---

### Phase 3 — Emergency Triage

Records patient speech and uses Gemma 4 (`gemma4:e4b` with extended reasoning) to extract a structured triage card: chief complaint, severity, duration, symptoms, vitals, and an immediate-attention flag.

```bash
pdb triage

# Use a specific device
pdb triage --device 2
```

**Example triage card:**

```
╔══════════════════════════════════════════╗
║   ⚠  IMMEDIATE ATTENTION REQUIRED  ⚠   ║
╚══════════════════════════════════════════╝

┌─ Triage Card — URGENT ───────────────────┐
│ Chief Complaint    Severe chest pain,    │
│                    radiating to left arm │
│ Severity           SEVERE                │
│ Duration           Since this morning    │
│ Symptoms           • chest pain          │
│                    • shortness of breath │
│                    • sweating            │
│ Vitals Mentioned   • pulse racing        │
│ Patient Language   Telugu                │
└──────────────────────────────────────────┘
```

> **Note:** Triage uses `gemma4:e4b` with `think=True` for extended reasoning. On CPU this may take 60–120 seconds. The 300-second timeout in `config/settings.py` covers this.

---

### Phase 4 — Prescription OCR

Reads a JPEG or PNG photo of a handwritten or printed prescription using Gemma 4's multimodal vision, and extracts a structured table of medicines.

```bash
# Process a prescription image
pdb prescription --image /path/to/prescription.jpg

pdb prescription --image ~/Desktop/rx_scan.png
```

**Example output:**
```
┌─ Prescription Details ──────────────────────────────┐
│ File       rx_scan.png                              │
│ Doctor     Dr. R. Sharma                            │
│ Patient    Arun Kumar                               │
│ Date       12-May-2025                              │
│ Notes      Rest, plenty of fluids                   │
└─────────────────────────────────────────────────────┘

┌─ Medicines (2) ─────────────────────────────────────────────────────────────┐
│ #  Medicine       Dosage  Form     Frequency      Duration  Instructions   │
│ 1  Paracetamol    500mg   tablet   twice daily    5 days    after meals    │
│ 2  Amoxicillin    250mg   capsule  3 times daily  7 days    with water     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 5 — Emergency Reassurance

Displays a numbered menu of 12 built-in comfort phrases (URGENT / MEDICAL / COMFORT / INFO categories) and translates the selected phrase into the patient's language instantly via `gemma4:e2b`.

```bash
pdb reassure
```

**Workflow:**
1. A menu of phrases is shown (e.g. `1. URGENT — Help is coming. You are safe.`).
2. Enter the number (or `C` for a custom phrase).
3. Enter the patient's language code: `hi`, `te`, `kn`, `ta`, or `en`.
4. The translated phrase is displayed prominently.
5. Translate another or press `Q` to quit.

**Language codes:**

| Code | Language |
|------|----------|
| `hi` | Hindi |
| `te` | Telugu |
| `kn` | Kannada |
| `ta` | Tamil |
| `en` | English (no translation) |

---

### Phase 6 — Web UI

Starts a local Flask server and serves a single-page web app covering all four patient-facing workflows in browser tabs. Audio is recorded directly in the browser via the MediaRecorder API.

```bash
# Start on default host/port (127.0.0.1:5000)
pdb server

# Custom port
pdb server --port 8080

# Bind to all interfaces (e.g. for LAN access on Pi 5)
pdb server --host 0.0.0.0 --port 5000

# Flask debug mode (auto-reload on code changes)
pdb server --debug
```

Then open **http://localhost:5000** in any modern browser.

**Tabs available in the UI:**

| Tab | What it does |
|-----|-------------|
| **Bridge** | Record patient → get doctor English; Record doctor → get patient translation. Doctor-side translation is auto-spoken in the patient's language. |
| **Triage** | Record patient → structured triage card |
| **Prescription** | Upload prescription image → medicines table + optional patient-language summary (with 🔊) |
| **Reassure** | Pick a comfort phrase → instant translation in patient language, auto-spoken |

### Per-request ASR controls (Bridge & Triage)

Both audio tabs expose two dropdowns above the Record button:

- **Patient language** — defaults to **Auto-detect**. Pick a specific language (Hindi / Telugu / Kannada / Tamil / English) only as an override when Auto-detect keeps choosing the wrong one. After a successful transcription, the detected language is shown in a small badge next to the dropdown.
- **Speech model** — defaults to **Whisper (fast)**. Switch to **IndicConformer-600M (AI4Bharat)** for Indian languages when transcript quality matters more than latency. First IndicConformer call downloads ~600 MB to `~/.cache/pdb/models/indic-conformer/`; subsequent calls are fast.

Combinations:

| Patient language | Speech model | Behaviour |
|------------------|--------------|-----------|
| Auto-detect      | Whisper       | Whisper auto-detects + transcribes in one pass |
| Specific (e.g. Telugu) | Whisper | Whisper locked to the chosen language |
| Auto-detect      | IndicConformer | Whisper runs **only to detect** the language → IndicConformer re-transcribes locked to that language (hybrid auto-detect) |
| Specific (e.g. Telugu) | IndicConformer | IndicConformer transcribes directly with the chosen language |

The Doctor turn always uses Whisper-locked-English because Whisper is already excellent on English and the IndicConformer load tax adds no value there.

> **Tip:** The status bar at the bottom of the UI shows Ollama connectivity and any server-side warnings (e.g. "No clear speech was detected", "Gemma 4 returned an empty translation"). If you see a `Detected: Telugu` badge appear with Telugu speech but the translation comes back empty, restart Flask — `core/engine.py` is not auto-reloaded.

---

## Running the Tests

The test suite runs entirely without Ollama or a microphone — all external calls are mocked.

```bash
# Run all 137 tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_prescription.py -v
pytest tests/test_reassurance.py -v
pytest tests/test_triage.py -v
pytest tests/test_engine.py -v
pytest tests/test_translation.py -v
pytest tests/test_handler.py -v
pytest tests/test_language_id.py -v
pytest tests/test_recorder.py -v

# Run a single test by name
pytest tests/test_triage.py::test_engine_triage_passes_think_true -v

# Run with coverage (install pytest-cov first)
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

**Expected result:** `137 passed`

### Test organisation

| File | Tests | What is covered |
|------|-------|----------------|
| `test_engine.py` | 14 | Model routing, response handling, Ollama error propagation, think=True, phase stubs |
| `test_translation.py` | 13 | patient→doctor and doctor→patient prompts, language display names |
| `test_triage.py` | 32 | Triage model routing, prompt content, JSON parsing, TriageResult fields, all 5 languages |
| `test_prescription.py` | 21 | JSON parsing strategies (clean/fenced/embedded), medicine normalisation, PrescriptionService |
| `test_reassurance.py` | 16 | Phrase bank integrity, English passthrough, engine call routing, language display |
| `test_handler.py` | 7 | Audio deletion contract, transcription return values |
| `test_language_id.py` | 11 | Language constraint/renormalisation, edge cases |
| `test_recorder.py` | 4 | Recording duration checks |

---

## Project Structure

```
patient-doctor-bridge/
├── audio/
│   ├── handler.py          # Whisper transcription + language detection + hallucination guard
│   ├── indic_conformer.py  # OPTIONAL: AI4Bharat IndicConformer-600M ASR backend
│   ├── language_id.py      # Constrain/renormalise Whisper language probs to 5 codes
│   ├── preprocessor.py     # SNR + silence analysis, light noise gate
│   ├── recorder.py         # sounddevice mic capture → temp WAV file
│   ├── script_normalizer.py# Detect & repair Urdu / romanised script in Indic transcripts
│   └── tts.py              # Facebook MMS-TTS — patient-language read-aloud
│
├── cli/
│   ├── main.py             # pdb entry point — routes subcommands
│   ├── bridge.py           # pdb bridge — two-way translation loop
│   ├── triage.py           # pdb triage — emergency triage extraction
│   ├── prescription.py     # pdb prescription — OCR a prescription image
│   └── reassure.py         # pdb reassure — comfort phrase translation menu
│
├── config/
│   ├── languages.py        # Supported codes, display names, confidence threshold
│   └── settings.py         # All tunable knobs (Whisper size, timeouts, ports…)
│
├── core/
│   ├── engine.py           # GemmaEngine — Ollama client, model routing, generate()
│   └── session.py          # In-memory session store; wipes on end_session()
│
├── translation/
│   ├── prompts.py          # All Gemma 4 prompt templates
│   ├── service.py          # TranslationService (patient↔doctor)
│   ├── triage.py           # TriageService + TriageResult TypedDict
│   ├── prescription.py     # PrescriptionService + MedicineItem TypedDict
│   └── reassurance.py      # ReassuranceService + REASSURANCE_PHRASES bank
│
├── web/
│   ├── server.py           # Flask app factory + API endpoints
│   └── static/
│       └── index.html      # Single-page UI (HTML + CSS + JS, no build step)
│
├── tests/
│   ├── conftest.py         # Stubs for sounddevice and ollama (no real hardware needed)
│   ├── test_engine.py
│   ├── test_translation.py
│   ├── test_triage.py
│   ├── test_prescription.py
│   ├── test_reassurance.py
│   ├── test_handler.py
│   ├── test_language_id.py
│   └── test_recorder.py
│
├── docs/
│   ├── phase1_eval.md
│   ├── phase2_eval.md
│   └── phase3_eval.md
│
├── pyproject.toml
└── README.md
```

---

## Privacy Contract

PatientDoctorBridge is designed for use in clinics where patient data is highly sensitive:

- **Audio files** are written to the OS temp directory and deleted with `os.unlink()` immediately after Whisper transcription finishes — even if transcription throws an error (the delete happens in a `finally` block).
- **Session data** (transcripts, triage results) is stored in memory only, in a `dict` inside `core/session.py`. `end_session()` clears it completely; nothing is written to disk.
- **Inference** runs entirely locally via Ollama. No text, audio, or images are sent to any external server.
- **Prescription images** are read once from disk, base64-encoded, sent to the local Ollama process, and the temp file is deleted in a `finally` block.

---

## Competition Constraints

This project was built for the **Kaggle Gemma 4 Impact Challenge**. The following rules are hard-coded into `core/engine.py` and must not be changed:

```python
MODELS = {
    InferenceMode.FAST_TRANSLATION:     "gemma4:e2b",
    InferenceMode.REASONING_EXTRACTION: "gemma4:e4b",
}
```

- **Only `gemma4:*` model tags are permitted for LLM inference.** No GPT, Llama, Mistral, or other LLMs for any of: translation, triage extraction, prescription OCR (multimodal vision), or reassurance phrase translation.
- **All inference must run locally.** No cloud API calls. Every model — LLM and non-LLM — runs on the device via Ollama (`http://localhost:11434`) or local PyTorch / CTranslate2.
- **Python 3.11 only.** The TTS dependency (Coqui XTTS v2, `TTS==0.22.0`) does not support 3.12+.
- TTFT target on Raspberry Pi 5 + Hailo-10H NPU: **< 500 ms** (Phase 7).

### Models used and why each is compliant

| Component | Model | Role | LLM? | Why it's compliant |
|-----------|-------|------|------|--------------------|
| Translation, triage, OCR, reassurance | `gemma4:e2b` / `gemma4:e4b` (Ollama) | All language understanding & generation | **Yes** | Both are Gemma 4 variants — the only models permitted by the challenge rules. |
| Speech-to-text (default) | `faster-whisper` (CTranslate2) | Acoustic model: waveform → text glyphs | No | ASR is a signal-processing model, not LLM inference. Runs locally. |
| Speech-to-text (optional) | `ai4bharat/indic-conformer-600m-multilingual` | Acoustic model trained specifically for Indian languages | No | Same category as Whisper — Conformer-CTC/RNNT, not an LLM. Runs locally via Transformers + Torch. Opt-in only. |
| Text-to-speech | `facebook/mms-tts-*` (VITS) | Vocoder + duration predictor: text glyphs → waveform | No | TTS is the inverse of ASR; not LLM inference. Runs locally via Transformers + Torch. |

The challenge rules' "Gemma 4 only" requirement applies to LLM inference. ASR and TTS are auxiliary subsystems that produce or consume audio; the original project already used Whisper (also non-Gemma) for ASR, so the precedent for non-Gemma signal models is built into the project from day one. IndicConformer is purely additive — Whisper remains the default and the user opts in.

---

## Tuning & Configuration

All knobs are in `config/settings.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `ASR_BACKEND_DEFAULT` | `"whisper"` | Backend used when a request doesn't specify one. Set to `"indic_conformer"` to make IndicConformer the default on requests that omit `asr_backend`. The UI always sends an explicit value. |
| `ASR_BACKENDS_AVAILABLE` | `("whisper", "indic_conformer")` | Whitelist of accepted backends. Requests with any other value silently fall back to `ASR_BACKEND_DEFAULT`. |
| `INDIC_CONFORMER_MODEL_ID` | `"ai4bharat/indic-conformer-600m-multilingual"` | HuggingFace model id loaded by `IndicConformerHandler`. Pin to a specific revision if you need reproducibility. |
| `INDIC_CONFORMER_DECODER` | `"rnnt"` | `"rnnt"` (more accurate) or `"ctc"` (slightly faster, marginally less accurate). |
| `WHISPER_MODEL_SIZE` | `"base"` | Upgrade to `"small"` for better Indic accuracy; downgrade for Pi 5 speed. |
| `WHISPER_BEAM_SIZE` | `1` | Greedy decoding (~4–5× faster than beam=5 on CPU). |
| `WHISPER_VAD_FILTER` | `True` | Pre-filters silence; saves time on recordings with long pauses. |
| `OLLAMA_TIMEOUT` | `600` | Seconds for fast-translation calls. Sized to cover Gemma 4's first cold-load (up to several minutes on a laptop CPU); subsequent calls return in seconds thanks to `keep_alive`. |
| `OLLAMA_TIMEOUT_REASONING` | `900` | Seconds for triage/OCR with `think=True`. Used for warmup too so the first Gemma load doesn't fail. |
| `GEMMA_TEMPERATURE` | `0.2` | Low = more deterministic translations. The engine auto-bumps temperature to 0.5 on a retry when the first call returns empty. |
| `GEMMA_NUM_CTX` | `8192` | Context window for reasoning / OCR. |
| `GEMMA_NUM_CTX_FAST` | `2048` | Context window for translation / reassurance. |
| `GEMMA_KEEP_ALIVE` | `"30m"` | How long Ollama pins the model in (V)RAM between calls. |
| `TTS_PREWARM_LANGS` | `[]` | Languages to eagerly preload at server startup. Empty = lazy load on first 🔊 click. Set to `["hi","te","kn","ta"]` for zero-latency first-use. |
| `WEB_HOST` | `127.0.0.1` | Change to `0.0.0.0` for LAN access. |
| `WEB_PORT` | `5000` | Web UI port. |

