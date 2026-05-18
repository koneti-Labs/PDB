# PatientDoctorBridge - Deep Technical Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Inference Engine](#core-inference-engine)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Translation Services](#translation-services)
- [CLI Layer](#cli-layer)
- [Web UI Architecture](#web-ui-architecture)
- [Configuration Management](#configuration-management)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Privacy Contract](#privacy-contract)
- [Performance Optimization](#performance-optimization)
- [Testing Strategy](#testing-strategy)

---

## Architecture Overview

PatientDoctorBridge is a privacy-first, fully local voice translation system for Indian clinics. The architecture follows a layered design with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ CLI (Rich)   │  │ Web UI       │  │ Future: Mobile App   │ │
│  │ argparse/TUI │  │ Flask/JS     │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Translation  │  │ Triage       │  │ Prescription OCR     │ │
│  │ Service      │  │ Service      │  │ Service              │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Reassurance  │  │ Session      │                             │
│  │ Service      │  │ Manager      │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Engine Layer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ GemmaEngine - Ollama Client Wrapper                       │   │
│  │ • Model routing (gemma4:e2b / gemma4:e4b)                 │   │
│  │ • Dual timeout clients (120s / 600s)                      │   │
│  │ • Warmup & keep_alive for model pinning                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Audio Processing Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Recorder     │  │ AudioHandler │  │ Language ID          │ │
│  │ sounddevice  │  │ Whisper      │  │ Constraint &         │ │
│  │              │  │ faster-      │  │ Renormalization      │ │
│  │              │  │ whisper      │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ IndicConfor- │  │ Script       │  │ Preprocessor         │ │
│  │ merHandler   │  │ Normalizer   │  │ (SNR / silence /     │ │
│  │ (optional)   │  │ (Devanagari/ │  │  light noise gate)   │ │
│  │ AI4Bharat    │  │  Urdu/Latin) │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TTSService (audio/tts.py) — Facebook MMS-TTS             │  │
│  │ Patient-language read-aloud for translations & phrases   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                         │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Languages    │  │ Settings     │                             │
│  │ (5 codes)    │  │ (tunable     │                             │
│  │              │  │  knobs)      │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Privacy-First**: All data processed in-memory, audio deleted immediately, no cloud API calls
2. **Local-Only Inference**: All LLM calls via Ollama running locally
3. **Dependency Injection**: Services accept engine instances for testability
4. **Singleton Pattern**: Class-level model caching to avoid repeated loading
5. **Warmup Strategy**: Preload models at server startup to eliminate cold-start latency

---

## Core Inference Engine

### GemmaEngine (`core/engine.py`)

The `GemmaEngine` class is the single point of truth for all Gemma 4 inference via Ollama.

#### Model Routing

```python
MODELS: dict[InferenceMode, str] = {
    InferenceMode.FAST_TRANSLATION: "gemma4:e2b",
    InferenceMode.REASONING_EXTRACTION: "gemma4:e4b",
}
```

**Competition Rule**: Only `gemma4:*` model tags are permitted. This is a hard constraint from the Kaggle Gemma 4 Impact Challenge.

- **gemma4:e2b** (~2B params): Used for fast translation tasks
  - Patient → Doctor translation
  - Doctor → Patient translation
  - Emergency reassurance translation
  - Prescription summary translation (patient-facing)
  - Context window: 2048 tokens (`GEMMA_NUM_CTX_FAST`)
  - Output cap: 512 tokens (`GEMMA_NUM_PREDICT_FAST`)
  - Timeout: 600s (`OLLAMA_TIMEOUT`) — sized to cover the first cold-load on a laptop CPU; subsequent calls are sub-second thanks to `keep_alive="30m"`

- **gemma4:e4b** (~4B params): Used for reasoning and vision tasks
  - Emergency triage extraction (with `think=True`)
  - Prescription OCR (multimodal vision)
  - Context window: 8192 tokens (`GEMMA_NUM_CTX`)
  - Output cap: 4096 tokens (`GEMMA_NUM_PREDICT_REASONING`)
  - Timeout: 900s (`OLLAMA_TIMEOUT_REASONING`) — also used for warmup so the very first model load doesn't fail under the shorter fast-translation timeout

#### Empty-Response Recovery

`GemmaEngine.generate()` includes a one-shot retry path for two failure modes that produced silent empty UI boxes in production:

1. **Think-block only.** `gemma4:e2b` occasionally emits only a `<think>...</think>` reasoning trace (often truncated by `num_predict`). After `_strip_think_tags()` runs, nothing is left.
2. **Whitespace-only output.** Low temperatures (≤0.3) plus certain Indic→English prompts can lock the sampler into producing nothing.

When either is detected (`cleaned == ""` after `_clean_response`), the engine retries once with `think=False` explicitly, `num_predict` doubled, and temperature nudged up to 0.5. Every call also prints a one-line trace at `dim` level:

```
Gemma gemma4:e2b raw=412ch clean=398ch first120='Sir, the patient reports…'
```

This trace appears for both the initial call and the retry, making the failure mode immediately visible in the server log.

#### Dual Client Pattern

```python
def __init__(self) -> None:
    self._client = ollama.Client(
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT,  # 120s for fast calls
    )
    self._reasoning_client = ollama.Client(
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT_REASONING,  # 600s for slow calls
    )
```

The dual client pattern prevents fast translation calls from timing out while waiting for slow reasoning calls to complete.

#### Keep-Alive Strategy

```python
generate_kwargs: dict[str, Any] = {
    "model": model,
    "prompt": prompt,
    "options": options,
    "keep_alive": GEMMA_KEEP_ALIVE,  # "30m" by default
}
```

Without `keep_alive`, Ollama unloads models after ~5 minutes of inactivity, causing a 20-60 second cold-load penalty on the next request. The 30-minute keep-alive pins models in (V)RAM between calls.

#### Warmup Mechanism

```python
def warmup(self) -> dict[str, bool]:
    """Send a 1-token generate to every configured model so Ollama loads
    them into (V)RAM up-front."""
    for mode, model in MODELS.items():
        client = (
            self._reasoning_client
            if mode == InferenceMode.REASONING_EXTRACTION
            else self._client
        )
        try:
            client.generate(
                model=model,
                prompt="ok",
                options={"num_predict": 1, "temperature": 0.0},
                keep_alive=GEMMA_KEEP_ALIVE,
            )
            results[model] = True
        except Exception:
            results[model] = False
    return results
```

**Server Startup Contract** (from `web/server.py::run_server`):
```python
_preload_audio_handler()  # Load Whisper
_get_engine().warmup()    # Pin both Gemma models
app.run(...)              # Now serving with warm models
```

This ensures the first user request doesn't pay a cold-start penalty.

#### Method Signatures

```python
# Core generation
def generate(
    self,
    prompt: str,
    mode: InferenceMode = InferenceMode.FAST_TRANSLATION,
    temperature: float | None = None,
    num_ctx: int | None = None,
    think: bool = False,
) -> str

# Phase 2: Translation
def translate_patient_to_doctor(self, text: str, source_lang: str) -> str
def translate_doctor_to_patient(self, text: str, target_lang: str) -> str

# Phase 3: Triage
def emergency_triage(self, text: str, lang_code: str = "en") -> str

# Phase 4: Prescription OCR
def transcribe_prescription(self, image_path: str) -> str

# Phase 5: Reassurance
def emergency_reassurance(self, phrase: str, target_lang: str) -> str

# Health check
def check_connectivity(self) -> dict[str, bool]
```

### Session Management (`core/session.py`)

The `Session` class provides in-memory session storage with guaranteed cleanup.

```python
class Session:
    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.started_at: datetime = datetime.now(tz=timezone.utc)
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None
    def get(self, key: str, default: Any = None) -> Any
    def append(self, key: str, value: Any) -> None
    def end_session(self) -> None  # Wipes all data
```

**Privacy Contract**: `end_session()` is called in a `finally` block in all CLI commands, ensuring session data is wiped even on KeyboardInterrupt.

---

## Audio Processing Pipeline

### Recorder (`audio/recorder.py`)

The `Recorder` class implements push-to-talk audio capture via sounddevice.

#### Recording Flow

```
User presses ENTER
    ↓
Open InputStream (sounddevice)
    ↓
Start callback thread (audio_chunks.append)
    ↓
User presses ENTER to stop
    ↓
Validate duration (2-30s)
    ↓
Save to tempfile.NamedTemporaryFile
    ↓
Return Path object
```

#### Key Parameters

```python
SAMPLE_RATE: int = 16_000   # Whisper's native format
CHANNELS: int = 1           # Mono
MAX_RECORDING_SECONDS: int = 30
MIN_RECORDING_SECONDS: float = 2.0
```

#### Privacy

Audio is written to `AUDIO_TEMP_DIR` (OS temp directory via `tempfile.gettempdir()`). No audio is written outside this directory. The caller (`AudioHandler.transcribe`) is responsible for deletion.

### AudioHandler (`audio/handler.py`)

The `AudioHandler` class is the single authoritative transcription entry point.

#### Class-Level Model Singleton

```python
class AudioHandler:
    _model = None  # Class-level, shared across all instances

    @classmethod
    def _ensure_model(cls) -> None:
        if cls._model is None:
            cls._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_CACHE_DIR),
            )
```

**Why Class-Level?** In the Flask web server, a new `AudioHandler` is created per request. Without the class-level singleton, Whisper would be re-loaded on every request (visible as repeated "Loading Whisper..." logs). The singleton ensures one model per Python process, shared across all requests.

#### Transcription Flow

```
AudioHandler.transcribe(audio_path)
    ↓
_ensure_model()  # Load Whisper once per process
    ↓
Whisper transcribe (language=None, beam_size=1, vad_filter=True)
    ↓
Get all_language_probs from TranscriptionInfo
    ↓
constrain_and_renormalize(prob_dict)
    ↓
If constrained language == auto-detected:
    Consume existing segments (no second encode)
Else:
    Re-transcribe with language=constrained_code
    ↓
os.unlink(audio_path) in finally block
    ↓
Return TranscriptResult(language, confidence, text)
```

#### One-Pass with Constrained Fallback

The design uses a single encode pass when possible:

1. **First pass**: Transcribe with `language=None` (auto-detect)
2. **Constrain**: Mask probability distribution to 5 supported codes
3. **Renormalize**: Re-normalize to sum to 1.0
4. **Compare**: If constrained winner == auto-detected winner, done
5. **Fallback**: If they differ (e.g., Whisper says "ur", we want "hi"), re-transcribe locked to the constrained language

This avoids the ~5% of cases where Whisper misclassifies Hindi as Urdu or Telugu as another Dravidian language.

#### Privacy Contract

```python
try:
    # Transcription logic
    ...
finally:
    os.unlink(audio_path)  # Guaranteed deletion
```

The `finally` block ensures audio is deleted even if transcription raises an exception. This is tested in `tests/test_handler.py`.

#### transcribe_locked()

For the doctor turn (known to be English), `transcribe_locked()` skips language detection entirely:

```python
def transcribe_locked(self, audio_path: Path | str, language: str) -> TranscriptResult:
    _raw = self._model.transcribe(
        str(audio_path),
        language=language,  # Skip detection
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
        ...
    )
    # ... deletion in finally block
```

This saves ~1-2 seconds per doctor turn on CPU.

### Whisper Hallucination Guard (`audio/handler.py::_is_repeat_hallucination`)

The default Whisper `base` model occasionally locks onto a repeating 1-4 token window on noisy / silent clips, producing transcripts like:

```
هاں பي ال هاں பي ال هاں பي ال هاں பي ال هاں பي ال …
```

Both `transcribe()` and `transcribe_locked()` apply the same check after Whisper returns and before script normalisation:

```python
if _is_repeat_hallucination(text):
    text = "[no speech detected]"
```

The detector tokenises on whitespace and, for window sizes 1-4, declares a hallucination if a single fixed window recurs ≥4 times and covers ≥80% of all tokens. Transcripts shorter than 12 tokens are exempt (genuine short utterances would false-positive). The sentinel `[no speech detected]` is treated by the server as an early-exit — Gemma 4 is never called on it, and the UI receives a structured warning explaining what to do.

### IndicConformer Handler (`audio/indic_conformer.py`) — optional ASR

`IndicConformerHandler` wraps the AI4Bharat `ai4bharat/indic-conformer-600m-multilingual` checkpoint with the same `TranscriptResult` shape as `AudioHandler`, so the rest of the pipeline doesn't care which backend produced the text.

#### Why offer a second backend

Whisper was trained on a heavily English-skewed multilingual corpus and produces several recurring failure modes on Indian languages: romanised-Latin or Urdu-script output for Hindi, dialect confusion among Telugu / Kannada / Tamil, and the repeat-token hallucinations described above. IndicConformer was trained specifically on Indian languages with high-quality crowd-sourced corpora; it produces native-script output (Devanagari / Telugu / Kannada / Tamil) on the first pass without any script-recovery patches.

The trade-offs are weight (~600 MB on disk vs ~140 MB for Whisper-base), latency (~2-3× slower per request on CPU), and the lack of a built-in language detector — IndicConformer must be told which language to transcribe.

#### Loading

The HuggingFace checkpoint is fetched once via `AutoModel.from_pretrained(..., trust_remote_code=True)` and cached at `~/.cache/pdb/models/indic-conformer/`. The model is held as a class-level singleton so every request shares the same loaded instance.

#### Decoder choice

`INDIC_CONFORMER_DECODER` (default `"rnnt"`) selects between the checkpoint's two decoders. `"rnnt"` is more accurate and slightly slower; `"ctc"` is faster. RNNT is the right default for a clinical-translation use case where accuracy beats latency.

### `_dispatch_asr()` — single source of truth for routing

`web/server.py::_dispatch_asr(tmp_path, locked_lang, asr_backend)` is called by both the Bridge patient endpoint and the Triage SSE endpoint, so they never disagree on which backend ran or how auto-detect was handled. Four cases:

| `asr_backend` | `locked_lang` | Behaviour |
|---------------|----------------|-----------|
| `whisper` | one of `hi/te/kn/ta/en` | `AudioHandler.transcribe_locked(tmp_path, locked_lang)` |
| `whisper` | empty / `auto`         | `AudioHandler.transcribe(tmp_path)` (constrained auto-detect) |
| `indic_conformer` | one of `hi/te/kn/ta/en` | `IndicConformerHandler.transcribe(tmp_path, locked_lang)` |
| `indic_conformer` | empty / `auto`         | Hybrid: Whisper on a copy of the file for cheap detection → IndicConformer on the original tmp file for transcription |

#### Why the hybrid uses `shutil.copyfile` not `_save_temp_audio` twice

Flask's `FileStorage` is a one-shot stream backed by the request body. After the first `.save(tmp_path)` drains it, calling `.save()` again writes 0 bytes — ffmpeg then raises `Invalid data found when processing input`. The dispatch copies the already-saved on-disk file to `tmp_path + ".detect"` for Whisper to consume; Whisper deletes its copy in its `finally` block as always, and IndicConformer consumes the original.

### Language Identification (`audio/language_id.py`)

The `constrain_and_renormalize()` function implements the language constraint strategy.

#### Strategy

Whisper detects ~99 languages and returns a full probability distribution. We:

1. **Mask**: Keep only probabilities for our 5 supported codes
2. **Renormalize**: Divide by sum so they sum to 1.0
3. **Select**: Pick the highest probability

```python
def constrain_and_renormalize(all_probs: dict[str, float]) -> tuple[str, float]:
    subset = {k: v for k, v in all_probs.items() if k in SUPPORTED_LANG_CODES}
    
    if not subset:
        return "en", 0.0  # Fallback
    
    total = sum(subset.values())
    normalized = {k: v / total for k, v in subset.items()}
    best = max(normalized, key=normalized.__getitem__)
    return best, normalized[best]
```

#### Supported Languages

```python
SUPPORTED_LANG_CODES: frozenset[str] = frozenset({"hi", "te", "kn", "en", "ta"})
```

- **hi**: Hindi
- **te**: Telugu
- **kn**: Kannada
- **ta**: Tamil
- **en**: English

#### Confidence Threshold

```python
CONFIDENCE_THRESHOLD: float = 0.40
```

Below this threshold, the CLI displays a "⚠ low confidence" warning. The threshold applies to the renormalized confidence (within the 5-code constraint), not the raw Whisper confidence.

---

## Translation Services

### TranslationService (`translation/service.py`)

Wraps `GemmaEngine` with domain-specific prompt construction.

```python
class TranslationService:
    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    def patient_to_doctor(self, text: str, lang_code: str) -> str:
        """Translate patient speech to clinical English."""
        lang_name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
        prompt = PATIENT_TO_DOCTOR_PROMPT.format(language=lang_name, text=text)
        return self._engine.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)

    def doctor_to_patient(self, text: str, lang_code: str) -> str:
        """Translate doctor's English to patient language."""
        lang_name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
        prompt = DOCTOR_TO_PATIENT_PROMPT.format(language=lang_name, text=text)
        return self._engine.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)
```

#### Prompt Templates (`translation/prompts.py`)

**Patient → Doctor**:
```
You are a medical interpreter in an Indian clinic. 
Translate the patient's statement from {language} into clear, clinical English for the doctor.

Rules:
- Preserve every symptom, its location, severity, and duration exactly.
- Do not diagnose, add opinions, or omit anything.
- If a term has no direct English equivalent, transliterate it and add a brief parenthetical note.
- Output only the English translation -- no preamble, no labels.

Patient ({language}): {text}

English for doctor:
```

**Doctor → Patient**:
```
You are a medical interpreter in an Indian clinic. 
Translate the doctor's English instructions into {language} for the patient.

Rules:
- Use simple, everyday {language} that a non-medical person can understand.
- Convert medical jargon into plain language.
- Preserve dosage, frequency, and timing instructions exactly.
- Warm, reassuring tone.
- Output only the {language} translation -- no preamble, no labels.

Doctor (English): {text}

{language} for patient:
```

### TriageService (`translation/triage.py`)

Extracts structured emergency triage data from patient speech.

#### TriageResult TypedDict

```python
class TriageResult(TypedDict):
    chief_complaint: str          # One-sentence English summary
    severity: str                 # "mild" | "moderate" | "severe" | "critical"
    duration: str                 # e.g. "since morning, ~3 hours"
    symptoms: list[str]           # Every distinct symptom reported
    vitals_mentioned: list[str]   # Any vitals the patient named
    needs_immediate_attention: bool
    language: str                 # ISO 639-1 code
```

#### JSON Parsing Strategy

Gemma 4 may return JSON wrapped in markdown fences or with extra text. The `_parse_json()` function uses three strategies:

```python
def _parse_json(raw: str) -> dict:
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```\s*$", "", fenced)
    try:
        return json.loads(fenced.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 3: Regex extract first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Gemma 4 returned non-JSON output")
```

#### Severity Normalization

```python
def _normalise_severity(value: object) -> str:
    valid = {"mild", "moderate", "severe", "critical"}
    s = str(value).lower().strip()
    return s if s in valid else "unknown"
```

### PrescriptionService (`translation/prescription.py`)

Extracts structured medicine data from prescription images via Gemma 4's multimodal vision.

#### MedicineItem TypedDict

```python
class MedicineItem(TypedDict):
    name: str
    dosage: str
    form: str
    frequency: str
    duration: str
    instructions: str
```

#### Vision Integration

```python
def transcribe_prescription(self, image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = self._reasoning_client.generate(
        model="gemma4:e4b",
        prompt=PRESCRIPTION_OCR_PROMPT,
        images=[image_b64],  # Multimodal param
        options=options,
        keep_alive=GEMMA_KEEP_ALIVE,
    )
    return response["response"].strip()
```

The image is base64-encoded and passed to Ollama's `images` parameter. The temp file is deleted in a `finally` block in the web server.

### ReassuranceService (`translation/reassurance.py`)

Translates emergency comfort phrases to patient language.

#### Phrase Bank

```python
REASSURANCE_PHRASES: list[tuple[str, str]] = [
    ("URGENT",  "Help is coming. You are safe."),
    ("URGENT",  "We are calling an ambulance right now."),
    ("URGENT",  "The doctor is on their way. Please stay calm."),
    ("MEDICAL", "We are going to give you medicine to help with the pain."),
    ("MEDICAL", "We need to do a small procedure. It will be over quickly."),
    ("MEDICAL", "Please do not eat or drink anything right now."),
    ("COMFORT", "Do not worry. You are in good hands."),
    ("COMFORT", "Your family has been informed and is coming."),
    ("COMFORT", "You are doing very well. Keep breathing slowly."),
    ("INFO",    "The doctor will explain everything to you shortly."),
    ("INFO",    "You will need to stay in hospital for observation."),
    ("INFO",    "You can go home today. Please rest and take your medicines."),
]
```

#### English Passthrough

```python
def translate(self, phrase: str, target_lang: str) -> str:
    if target_lang == "en":
        return phrase  # No translation needed
    return self._engine.emergency_reassurance(phrase, target_lang)
```

---

## CLI Layer

### Command Router (`cli/main.py`)

The CLI uses argparse with subcommands:

```python
pdb listen [--device N] [--duration N]       # Phase 1
pdb bridge [--device N]                      # Phase 2
pdb triage [--device N]                      # Phase 3
pdb prescription --image PATH                # Phase 4
pdb reassure                                 # Phase 5
pdb server [--host HOST] [--port PORT] [--debug]  # Phase 6
```

### Bridge Command (`cli/bridge.py`)

Implements the two-way patient ↔ doctor translation loop.

#### Exchange Flow

```
┌─────────────────────────────────────────────────────────────┐
│ TURN 1: Patient                                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Record patient audio (any supported language)            │
│ 2. Transcribe + detect language (AudioHandler.transcribe)   │
│ 3. Translate to English (TranslationService.patient_to_doctor) │
│ 4. Display Rich panel with translation                      │
│ 5. Store in session.append("exchanges", {...})              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ TURN 2: Doctor                                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Record doctor audio (English)                             │
│ 2. Transcribe locked to "en" (AudioHandler.transcribe_locked)│
│ 3. Translate to patient language (TranslationService.doctor_to_patient) │
│ 4. Display Rich panel with translation                      │
│ 5. Store in session.append("exchanges", {...})              │
└─────────────────────────────────────────────────────────────┘
```

#### Session Storage

```python
session.append("exchanges", {
    "role": "patient",
    "language": patient_result["language"],
    "original": patient_result["text"],
    "translated": for_doctor,
})
```

### Triage Command (`cli/triage.py`)

Implements emergency triage extraction with severity-styled display.

#### Severity Styling

```python
_SEVERITY_STYLE: dict[str, tuple[str, str]] = {
    "critical": ("bold red on dark_red", "CRITICAL ⚠"),
    "severe":   ("bold red", "SEVERE"),
    "moderate": ("bold yellow", "MODERATE"),
    "mild":     ("bold green", "MILD"),
    "unknown":  ("bold white", "UNKNOWN"),
}
```

#### Immediate Attention Banner

```python
if immediate:
    console.print(
        Panel(
            Text("⚠  IMMEDIATE ATTENTION REQUIRED  ⚠", style="bold white", justify="center"),
            style="bold red",
            padding=(0, 2),
        )
    )
```

### Prescription Command (`cli/prescription.py`)

Handles prescription OCR with image validation.

#### File Validation

```python
path = Path(image_path)
if not path.exists():
    console.print(f"[red]Error: file not found: {path}[/red]")
    sys.exit(1)

if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
    console.print("[yellow]Warning: file may not be a supported image type[/yellow]")
```

### Reassure Command (`cli/reassure.py`)

Interactive phrase menu with category coloring.

#### Category Styling

```python
_CAT_STYLE: dict[str, str] = {
    "URGENT":  "bold red",
    "MEDICAL": "bold yellow",
    "COMFORT": "bold green",
    "INFO":    "bold cyan",
}
```

#### Input Loop

```python
while True:
    _display_phrase_menu()
    phrase = _pick_phrase()  # Number or 'C' for custom
    lang_code = _pick_language()  # hi/te/kn/ta/en
    translated = service.translate(phrase, lang_code)
    _display_translation(phrase, translated, lang_display)
    if not _ask_continue():
        break
```

---

## Web UI Architecture

### Flask Server (`web/server.py`)

#### Lazy Singleton Pattern

Services are initialized lazily as global singletons to avoid per-request construction overhead:

```python
_engine: GemmaEngine | None = None
_audio_handler: AudioHandler | None = None
_triage_svc: TriageService | None = None
_reassure_svc: ReassuranceService | None = None
_prescription_svc: PrescriptionService | None = None

def _get_engine() -> GemmaEngine:
    global _engine
    if _engine is None:
        _engine = GemmaEngine()
    return _engine
```

This shaves ~5-50ms off every request compared to per-request instantiation.

#### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve index.html |
| `/api/health` | GET | Check Ollama connectivity |
| `/api/phrases` | GET | List reassurance phrases |
| `/api/bridge/patient` | POST | Patient audio → English |
| `/api/bridge/doctor` | POST | Doctor audio → Patient language |
| `/api/triage` | POST | Patient audio → Triage JSON |
| `/api/reassure` | POST | Phrase → Translation |
| `/api/prescription` | POST | Image → Medicines JSON |

#### Audio Upload Handling

```python
def _handle_audio_translation(role: str):
    audio_file = request.files.get("audio")
    lang_code = request.form.get("language", "hi")
    
    tmp_path = _save_temp_audio(audio_file)
    try:
        handler = _get_audio_handler()
        if role == "doctor":
            result = handler.transcribe_locked(tmp_path, language="en")
        else:
            result = handler.transcribe(tmp_path)
        # ... translation logic
    finally:
        _delete_temp(tmp_path)
```

Audio is saved to a temp file, processed, and deleted in the `finally` block.

#### Server Warmup

```python
def run_server(host: str | None = None, port: int | None = None, debug: bool = False):
    app = create_app()
    
    # Warm up everything BEFORE serving requests
    console.print("[dim]Warming up Whisper + Gemma 4 (one-time)...[/dim]")
    _preload_audio_handler()  # Load Whisper
    try:
        _get_engine().warmup()  # Pin both Gemma models
    except Exception as exc:
        console.print(f"[yellow]Gemma warmup skipped: {exc}[/yellow]")
    
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
```

**Critical**: `use_reloader=False` prevents the warmup from running twice in dev mode.

### Single-Page App (`web/static/index.html`)

#### Tab Navigation

```javascript
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}
```

Four tabs: Bridge, Triage, Prescription, Reassure.

#### MediaRecorder Integration

```javascript
async function startRecord(role) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const chunks = [];
  const mr = new MediaRecorder(stream);
  mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mr.onstop = () => { uploadAudio(role, chunks); stream.getTracks().forEach(t => t.stop()); };
  mr.start(250);
}
```

Audio is recorded as WebM chunks, uploaded as multipart/form-data.

#### Health Check Polling

```javascript
async function checkHealth() {
  const res = await fetch('/api/health');
  const data = await res.json();
  if (data.ok) {
    const all = Object.values(data.models || {}).every(Boolean);
    setStatus(all ? 'Ollama OK · Gemma 4 models ready.' : 'Ollama OK · some models missing', all ? 'ok' : 'warn');
  } else {
    setStatus('Ollama not reachable — run: ollama serve', 'err');
  }
}
```

The status bar at the bottom shows Ollama connectivity in real-time.

---

## Configuration Management

### Languages (`config/languages.py`)

Single source of truth for language codes and display names.

```python
SUPPORTED_LANG_CODES: frozenset[str] = frozenset({"hi", "te", "kn", "en", "ta"})

LANGUAGE_DISPLAY: dict[str, str] = {
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "en": "English",
    "ta": "Tamil",
}

LANGUAGE_NATIVE: dict[str, str] = {
    "hi": "हिन्दी",
    "te": "తెలుగు",
    "kn": "ಕನ್ನಡ",
    "en": "English",
    "ta": "தமிழ்",
}

CONFIDENCE_THRESHOLD: float = 0.40
```

### Settings (`config/settings.py`)

All tunable knobs in one place. Import from here - never hard-code values in other modules.

#### Whisper Configuration

```python
WHISPER_MODEL_SIZE: str = "base"  # "tiny" | "base" | "small" | "medium"
WHISPER_BEAM_SIZE: int = 1        # Greedy decoding (4-5x faster than beam=5)
WHISPER_VAD_FILTER: bool = True   # Pre-filter silence
WHISPER_CONDITION_ON_PREVIOUS: bool = False  # Skip prefix conditioning
```

**Pi 5 Migration Notes**:
- Downgrade to "tiny" if TTFT > 500ms
- Upgrade to "small" for better Indic accuracy on laptop

#### Audio Configuration

```python
SAMPLE_RATE: int = 16_000
CHANNELS: int = 1
MAX_RECORDING_SECONDS: int = 30
MIN_RECORDING_SECONDS: float = 2.0
AUDIO_TEMP_DIR: Path = Path(tempfile.gettempdir())
MODEL_CACHE_DIR: Path = Path.home() / ".cache" / "pdb" / "models"
```

#### Ollama Configuration

```python
OLLAMA_HOST: str = "http://localhost:11434"
OLLAMA_TIMEOUT: int = 120            # Fast translation
OLLAMA_TIMEOUT_REASONING: int = 600  # Triage/OCR
```

#### Gemma Generation Defaults

```python
GEMMA_TEMPERATURE: float = 0.2
GEMMA_TOP_P: float = 0.9
GEMMA_NUM_CTX: int = 4096          # Full context for reasoning/OCR
GEMMA_NUM_CTX_FAST: int = 1024     # Small window for translation
GEMMA_NUM_PREDICT_FAST: int = 256        # Short utterances
GEMMA_NUM_PREDICT_REASONING: int = 1024  # Triage JSON + OCR
GEMMA_KEEP_ALIVE: str = "30m"       # Pin models in (V)RAM
```

#### Web Server Configuration

```python
WEB_HOST: str = "127.0.0.1"
WEB_PORT: int = 5000
```

#### Triage Reasoning Mode

```python
TRIAGE_THINK_MODE: bool = True
```

- `True`: Gemma 4 emits internal reasoning trace before JSON (more accurate, 10-30s slower)
- `False`: Direct JSON output (3-5x faster, less accurate)

---

## Data Flow Diagrams

### Phase 1: Listen & Detect

```
User speaks
    ↓
Recorder.record() → temp WAV file
    ↓
AudioHandler.transcribe(audio_path)
    ↓
WhisperModel (one-pass, language=None)
    ↓
all_language_probs
    ↓
constrain_and_renormalize() → {hi, te, kn, en, ta}
    ↓
If constrained != auto-detected:
    Re-transcribe with language=constrained
    ↓
os.unlink(audio_path) [finally block]
    ↓
TranscriptResult(language, confidence, text)
    ↓
Rich display + session.append("transcripts", {...})
```

### Phase 2: Bridge (Patient Turn)

```
Patient speaks
    ↓
Recorder.record() → temp WAV
    ↓
AudioHandler.transcribe() → TranscriptResult
    ↓
TranslationService.patient_to_doctor(text, lang_code)
    ↓
GemmaEngine.generate(prompt, mode=FAST_TRANSLATION)
    ↓
Ollama client (gemma4:e2b, 120s timeout)
    ↓
English translation
    ↓
Rich panel display
    ↓
session.append("exchanges", {...})
```

### Phase 2: Bridge (Doctor Turn)

```
Doctor speaks (English)
    ↓
Recorder.record() → temp WAV
    ↓
AudioHandler.transcribe_locked(language="en")
    ↓
TranslationService.doctor_to_patient(text, patient_lang_code)
    ↓
GemmaEngine.generate(prompt, mode=FAST_TRANSLATION)
    ↓
Ollama client (gemma4:e2b, 120s timeout)
    ↓
Patient language translation
    ↓
Rich panel display
    ↓
session.append("exchanges", {...})
```

### Phase 3: Emergency Triage

```
Patient speaks
    ↓
AudioHandler.transcribe() → TranscriptResult
    ↓
TriageService.extract(text, lang_code)
    ↓
GemmaEngine.emergency_triage(text, lang_code)
    ↓
Ollama client (gemma4:e4b, think=True, 600s timeout)
    ↓
Raw JSON string
    ↓
_parse_json() → dict (3 strategies)
    ↓
TriageResult TypedDict
    ↓
Rich triage card (severity-styled)
    ↓
session.append("triage_results", {...})
```

### Phase 4: Prescription OCR

```
User provides image path
    ↓
PrescriptionService.extract(image_path)
    ↓
GemmaEngine.transcribe_prescription(image_path)
    ↓
Base64 encode image
    ↓
Ollama client (gemma4:e4b, images=[b64], 600s timeout)
    ↓
Raw JSON string
    ↓
_parse_json() → dict
    ↓
PrescriptionResult TypedDict
    ↓
Rich table display
```

### Phase 5: Emergency Reassurance

```
User selects phrase (or custom)
    ↓
User selects target language
    ↓
ReassuranceService.translate(phrase, target_lang)
    ↓
If target_lang == "en":
    Return phrase (no-op)
Else:
    GemmaEngine.emergency_reassurance(phrase, target_lang)
    ↓
Ollama client (gemma4:e2b, 120s timeout)
    ↓
Translated phrase
    ↓
Rich panel display
```

### Phase 6: Web UI Flow

```
Browser: MediaRecorder.start()
    ↓
Browser: Audio chunks → Blob
    ↓
Browser: fetch('/api/bridge/patient', FormData(audio, language))
    ↓
Flask: _save_temp_audio() → temp file
    ↓
Flask: AudioHandler.transcribe()
    ↓
Flask: GemmaEngine.translate_patient_to_doctor()
    ↓
Flask: jsonify({transcript, translation})
    ↓
Browser: Render results
    ↓
Flask: _delete_temp() [finally block]
```

---

## Privacy Contract

### Audio Deletion

**Rule**: Audio files are deleted immediately after transcription, guaranteed by `finally` blocks.

```python
try:
    result = handler.transcribe(audio_path)
    # ... processing
finally:
    os.unlink(audio_path)  # Runs even if exception raised
```

**Enforcement**:
- `AudioHandler.transcribe()`: Audio deleted in finally
- `AudioHandler.transcribe_locked()`: Audio deleted in finally
- `web/server.py`: Temp files deleted in finally
- Test: `tests/test_handler.py` asserts file is gone after transcribe()

### Session Data

**Rule**: Session data is memory-only, wiped on `end_session()`.

```python
session = Session()
try:
    # ... interaction loop
    session.append("transcripts", result)
finally:
    session.end_session()  # Clears _data dict
```

**Enforcement**:
- All CLI commands call `end_session()` in finally block
- `Session._data` is never written to disk
- No persistence layer exists for session data

### Inference Privacy

**Rule**: All inference runs locally via Ollama. No data leaves the device.

**Enforcement**:
- Ollama host is hardcoded to `http://localhost:11434`
- No external API calls in the codebase
- Prescription images are processed locally (base64 to local Ollama)
- Competition rule: ONLY gemma4:* models permitted

### Logging

**Rule**: Audio paths are not logged above DEBUG level.

**Enforcement**:
- Audio paths are passed as Path objects, not strings in logs
- No INFO/WARNING/ERROR logs contain audio file paths
- Session data is not logged

---

## Performance Optimization

### Whisper Optimization

**Greedy Decoding**:
```python
WHISPER_BEAM_SIZE: int = 1  # Greedy, 4-5x faster than beam=5
```

For GPU or Pi 5 + NPU, increase to 3-5 for better accuracy.

**VAD Pre-Filtering**:
```python
WHISPER_VAD_FILTER: bool = True
```

Pre-filters silence before decoding. Saves time on recordings with long pauses. Adds ~100ms overhead on dense speech.

**Model Size**:
```python
WHISPER_MODEL_SIZE: str = "base"  # ~74M params
```

- "tiny" (~39M): Fastest, lower accuracy
- "base" (~74M): Good balance
- "small" (~244M): Better Indic accuracy
- "medium" (~769M): Best accuracy, slowest

### Gemma 4 Optimization

**Context Window Sizing**:
```python
GEMMA_NUM_CTX_FAST: int = 1024      # Translation: small window
GEMMA_NUM_CTX: int = 4096           # Reasoning: full window
```

Smaller context window = faster prompt evaluation.

**Output Length Capping**:
```python
GEMMA_NUM_PREDICT_FAST: int = 256        # Short utterances
GEMMA_NUM_PREDICT_REASONING: int = 1024  # Structured output
```

Prevents the model from generating multi-paragraph trailing text.

**Keep-Alive**:
```python
GEMMA_KEEP_ALIVE: str = "30m"
```

Pins models in (V)RAM between calls. Without this, cold-load penalty is 20-60 seconds.

### Server Optimization

**Lazy Singletons**:
```python
_engine: GemmaEngine | None = None
def _get_engine() -> GemmaEngine:
    global _engine
    if _engine is None:
        _engine = GemmaEngine()
    return _engine
```

Avoids per-request construction overhead (~5-50ms saved per request).

**Warmup Strategy**:
```python
_preload_audio_handler()  # Load Whisper
_get_engine().warmup()    # Pin both Gemma models
app.run(...)
```

First user request gets warm models instead of cold start.

**Doctor Turn Optimization**:
```python
if role == "doctor":
    result = handler.transcribe_locked(tmp_path, language="en")
```

Skips language detection for known English audio (~1-2s saved per turn).

---

## Testing Strategy

### Test Organization

| File | Tests | Coverage |
|------|-------|----------|
| `test_engine.py` | 14 | Model routing, response handling, Ollama error propagation, think=True |
| `test_translation.py` | 13 | Patient→doctor and doctor→patient prompts, language display |
| `test_triage.py` | 32 | Model routing, prompt content, JSON parsing, TriageResult fields |
| `test_prescription.py` | 21 | JSON parsing strategies, medicine normalization |
| `test_reassurance.py` | 16 | Phrase bank integrity, English passthrough, engine routing |
| `test_handler.py` | 7 | Audio deletion contract, transcription return values |
| `test_language_id.py` | 11 | Language constraint/renormalization, edge cases |
| `test_recorder.py` | 4 | Recording duration checks |

**Total**: 118 tests

### Mocking Strategy

All external dependencies are mocked in `tests/conftest.py`:

```python
# Mock sounddevice
@pytest.fixture
def mock_sounddevice(monkeypatch):
    # ...

# Mock Ollama
@pytest.fixture
def mock_ollama(monkeypatch):
    # ...

# Mock Whisper
@pytest.fixture
def mock_whisper(monkeypatch):
    # ...
```

This allows tests to run without:
- Microphone hardware
- Ollama server
- Whisper model download

### Critical Tests

**Audio Deletion Contract** (`test_handler.py`):
```python
def test_audio_deleted_after_transcribe(tmp_path, mock_whisper):
    handler = AudioHandler()
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")
    
    handler.transcribe(audio_file)
    
    assert not audio_file.exists()  # File must be deleted
```

**Language Constraint** (`test_language_id.py`):
```python
def test_constrain_masks_to_supported_codes():
    probs = {"hi": 0.40, "ur": 0.30, "en": 0.15, "ml": 0.10, "te": 0.05}
    lang, conf = constrain_and_renormalize(probs)
    assert lang in SUPPORTED_LANG_CODES
    assert conf > 0
```

**JSON Parsing Robustness** (`test_triage.py`):
```python
def test_parse_json_handles_markdown_fences():
    raw = "```json\n{\"severity\": \"severe\"}\n```"
    data = _parse_json(raw)
    assert data["severity"] == "severe"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_triage.py -v
```

---

## Future Extensions

### Phase 7: Raspberry Pi 5 + Hailo-10H NPU

Target TTFT: < 500ms

**Planned Optimizations**:
- Downgrade Whisper to "tiny" model
- Use Hailo-10H for Whisper inference (CTranslate2 with Hailo backend)
- Use Hailo-10H for Gemma 4 quantization (INT4/INT8)
- Optimize keep_alive for NPU memory constraints

### Phase 8: Mobile App

**Planned Architecture**:
- Flutter or React Native frontend
- REST API backend (reuse Flask server)
- Offline-first with local Ollama on device
- Bluetooth audio for clinic integration

### Phase 9: Advanced Features

**Planned Features**:
- Conversation history with export
- Custom phrase bank editor
- Multi-patient session management
- Integration with hospital EMR systems
- Voice activity detection for automatic recording

---

## Troubleshooting

### Common Issues

**Whisper fails to load**:
- Check `MODEL_CACHE_DIR` permissions
- Ensure sufficient disk space (~500MB for "base" model)
- Verify PortAudio installation

**Ollama connection timeout**:
- Run `ollama serve` in background
- Check `OLLAMA_HOST` setting
- Verify model availability: `ollama list`

**Audio not recording**:
- Check microphone permissions
- List devices: `pdb listen --device list`
- Try specific device: `pdb listen --device 2`

**Triage very slow**:
- Set `TRIAGE_THINK_MODE = False` in settings
- This disables extended reasoning but speeds up 3-5x

**Web UI not loading**:
- Check browser console for errors
- Verify Flask server is running
- Check port: default is 5000

### Debug Mode

Enable Flask debug mode:
```bash
pdb server --debug
```

This enables auto-reload and detailed error pages.

### Logging

Increase Whisper logging:
```python
import logging
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

Check Ollama logs:
```bash
ollama logs
```

---

## References

### Competition Constraints

- **Kaggle Gemma 4 Impact Challenge**
- Only `gemma4:*` model tags permitted
- All inference must run locally
- Python 3.11 only (TTS dependency constraint)
- Privacy-first design

### Dependencies

- **faster-whisper**: Whisper ASR with CTranslate2 backend
- **ollama-python**: Local LLM inference
- **sounddevice**: Audio capture
- **Flask**: Web server
- **Rich**: Terminal UI
- **pytest**: Testing framework

### External Resources

- [Ollama Documentation](https://ollama.com)
- [Gemma 4 Models](https://ollama.com/library/gemma4)
- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Rich Documentation](https://rich.readthedocs.io)

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-15  
**Project**: PatientDoctorBridge (Kaggle Gemma 4 Impact Challenge)
