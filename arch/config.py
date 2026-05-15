import os
from pathlib import Path
from enum import Enum

# ================== Languages ==================
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ta": "Tamil"
}

# ================== Inference Modes ==================
class InferenceMode(str, Enum):
    FAST_TRANSLATION = "fast_translation"
    REASONING_EXTRACTION = "reasoning_extraction"

# ================== Gemma 4 Models (MANDATORY) ==================
MODELS = {
    InferenceMode.FAST_TRANSLATION: "gemma4:e2b",
    InferenceMode.REASONING_EXTRACTION: "gemma4:e4b",
}

# ================== Paths & Audio ==================
BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ================== Whisper & Language Detection ==================
WHISPER_MODEL = "base"            # For transcription

# AI4Bharat Indic Conformer for language identification (better for Indian languages)
LANG_ID_MODEL = "ai4bharat/indic-conformer-600m-multilingual"

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 30