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

| Phase | Feature | CLI command | Model |
|-------|---------|-------------|-------|
| 1 | Voice capture + language detection | `pdb listen` | Whisper (local) |
| 2 | Two-way patient ↔ doctor translation | `pdb bridge` | `gemma4:e2b` |
| 3 | Emergency triage extraction | `pdb triage` | `gemma4:e4b` |
| 4 | Prescription OCR (multimodal vision) | `pdb prescription --image rx.jpg` | `gemma4:e4b` |
| 5 | Emergency reassurance phrases | `pdb reassure` | `gemma4:e2b` |
| 6 | Single-page web UI | `pdb server` | both |

**Supported languages:** Hindi (hi), Telugu (te), Kannada (kn), Tamil (ta), English (en)

---

## Architecture

```
Browser / CLI
     │
     ▼
audio/recorder.py   ←  records via sounddevice (CTRL+C to stop)
     │
audio/handler.py    ←  Whisper ASR (one-pass transcription + language detection)
     │               ←  audio file deleted immediately after transcription
     ▼
config/languages.py ←  constrains + renormalises to 5 language codes
     │
     ▼
core/engine.py      ←  GemmaEngine — routes to two Ollama clients:
     │                   FAST_TRANSLATION      → gemma4:e2b  (120 s timeout)
     │                   REASONING_EXTRACTION  → gemma4:e4b  (300 s timeout)
     │
translation/
  service.py        ←  patient↔doctor text translation
  triage.py         ←  JSON extraction + TriageResult TypedDict
  prescription.py   ←  medicine list extraction from images
  reassurance.py    ←  comfort phrase bank + translation
     │
cli/                ←  Rich terminal UI for each phase
web/server.py       ←  Flask API + single-page HTML/JS UI
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | **3.11 exactly** | TTS dependency (Coqui XTTS v2) requires 3.11 |
| [Ollama](https://ollama.com) | ≥ 0.4.7 | For local Gemma 4 inference |
| PortAudio | system library | Required by sounddevice for mic access |
| FFmpeg | any recent | Required by faster-whisper for audio decoding |

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

# 3. Install the package and all dependencies
pip install -e ".[dev]"

# 4. Verify the CLI is available
pdb --help
```

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
| **Bridge** | Record patient → get doctor English; Record doctor → get patient translation |
| **Triage** | Record patient → structured triage card |
| **Prescription** | Upload prescription image → medicines table |
| **Reassure** | Pick a comfort phrase → instant translation in patient language |

> **Tip:** The status bar at the bottom of the UI shows Ollama connectivity. If it shows a warning, run `ollama serve` and refresh the page.

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
│   ├── handler.py          # Whisper transcription + language detection (one-pass)
│   ├── language_id.py      # Constrain/renormalise Whisper language probs to 5 codes
│   └── recorder.py         # sounddevice mic capture → temp WAV file
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

- **Only `gemma4:*` model tags are permitted.** No GPT, Llama, Mistral, or other models.
- **All inference must run locally.** No cloud API calls.
- **Python 3.11 only.** The TTS dependency (Coqui XTTS v2, `TTS==0.22.0`) does not support 3.12+.
- TTFT target on Raspberry Pi 5 + Hailo-10H NPU: **< 500 ms** (Phase 7).

---

## Tuning & Configuration

All knobs are in `config/settings.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `WHISPER_MODEL_SIZE` | `"base"` | Upgrade to `"small"` for better Indic accuracy; downgrade for Pi 5 speed |
| `WHISPER_BEAM_SIZE` | `1` | Greedy decoding (~4–5× faster than beam=5 on CPU) |
| `WHISPER_VAD_FILTER` | `True` | Pre-filters silence; saves time on recordings with long pauses |
| `OLLAMA_TIMEOUT` | `120` | Seconds for fast translation calls |
| `OLLAMA_TIMEOUT_REASONING` | `300` | Seconds for triage/OCR with `think=True` |
| `GEMMA_TEMPERATURE` | `0.2` | Low = more deterministic translations |
| `GEMMA_NUM_CTX` | `4096` | Context window size |
| `WEB_HOST` | `127.0.0.1` | Change to `0.0.0.0` for LAN access |
| `WEB_PORT` | `5000` | Web UI port |

