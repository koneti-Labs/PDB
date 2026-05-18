"""
audio/indic_conformer.py

Optional ASR backend using AI4Bharat's IndicConformer-600M multilingual model.

Why this exists alongside Whisper
---------------------------------
Whisper's small "base" weights are convenient (~140 MB) but they hallucinate
on noisy clips and often confuse Hindi/Telugu/Kannada/Tamil because Whisper
was trained on a heavily English-skewed multilingual corpus.

AI4Bharat trained IndicConformer-600M specifically on Indian languages with
high-quality crowd-sourced corpora.  In practice it:

  • produces native-script output (Devanagari / Telugu / Kannada / Tamil)
    on the first pass, so the romanised-Latin and Urdu-script bugs go away;
  • handles dialectal variation and code-mixing (English words inside a
    Hindi utterance) much better than Whisper;
  • is heavier — about 600 MB on disk, ~2-3x slower per request on CPU —
    which is why we offer it as an opt-in second backend rather than
    replacing Whisper outright.

The handler exposes the same TranscriptResult shape as :class:`AudioHandler`
so the web server can switch between backends without any downstream code
changes.

Privacy contract is identical to ``audio/handler.py``: the temp audio file
is unconditionally deleted in a finally block — even when transcription
raises.

Competition note
----------------
The Kaggle Gemma 4 Impact Challenge mandates Gemma 4 for **LLM inference**
(translation, triage, OCR).  ASR is a separate subsystem; using a non-Gemma
ASR is allowed and is what the project already does with Whisper.  This
module is therefore policy-compliant — it does not affect any
:class:`InferenceMode` routing in ``core/engine.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import numpy as np
from rich.console import Console

from audio.preprocessor import AudioPreprocessor
from audio.script_normalizer import ScriptNormalizer
from config.settings import (
    INDIC_CONFORMER_DECODER,
    INDIC_CONFORMER_MODEL_ID,
    MODEL_CACHE_DIR,
)

console = Console()

# AI4Bharat's checkpoint supports 22 Indian languages.  We constrain to the
# five PDB supports so the rest of the pipeline (prompts, TTS, UI) stays
# coherent.  English is included because the AI4Bharat model can still
# transcribe English fine and Bridge users sometimes pick it.
_SUPPORTED = {"hi", "te", "kn", "ta", "en"}


class TranscriptResult(TypedDict):
    """Same shape as audio.handler.TranscriptResult so callers don't care
    which backend produced it."""
    language: str
    confidence: float
    text: str


class IndicConformerHandler:
    """
    Lazy-loaded singleton wrapper around AI4Bharat IndicConformer-600M.

    Loading
    -------
    The HuggingFace checkpoint is downloaded on first use into
    ``MODEL_CACHE_DIR/indic-conformer/`` (~600 MB).  Subsequent process
    starts read from disk in ~3-8 s on a laptop CPU.

    Language handling
    -----------------
    Unlike Whisper, IndicConformer does **not** detect the source language —
    the caller must specify it.  The web server's ``/api/bridge/patient``
    endpoint covers this two ways:

      • If the UI sends a specific language code, we pass it straight through.
      • If the UI sends ``language=auto``, the server first runs Whisper for
        cheap language detection (re-uses the constrained 5-language mask)
        and then calls IndicConformer with that locked language for the
        actual transcript.  That hybrid gets you Whisper's auto-detect
        ergonomics with IndicConformer's transcription quality.
    """

    # Class-level singleton — every instance reuses the same loaded model.
    _model = None

    def __init__(self) -> None:
        # Same auxiliary helpers used by AudioHandler so we get the audio
        # quality logging and Hindi script-normalisation safety net.
        self.preprocessor = AudioPreprocessor()
        self.script_normalizer = ScriptNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path | str, language: str) -> TranscriptResult:
        """
        Transcribe *audio_path* in the given language and delete the file.

        Parameters
        ----------
        audio_path:
            Path (or string) to the WAV/webm/ogg audio.  Strings are coerced.
        language:
            ISO 639-1 code — must be one of ``hi / te / kn / ta / en``.
            IndicConformer cannot auto-detect; the web server provides this.

        Returns
        -------
        TranscriptResult with the resolved language, a confidence of 1.0
        (the model does not surface a confidence number), and the
        transcribed text in native script.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist before transcription starts.
        RuntimeError
            If the model fails to load or inference raises — propagated with
            a helpful message so the UI can show something better than 500.
        """
        self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        lang = (language or "").strip().lower()
        if lang not in _SUPPORTED:
            console.print(
                f"[yellow]IndicConformer does not support '{lang}' in this "
                f"project — falling back to 'hi'.[/yellow]"
            )
            lang = "hi"

        try:
            audio = self._load_audio(audio_path)

            # Optional quality logging — mirrors AudioHandler so users see
            # consistent diagnostics regardless of backend.
            try:
                self.preprocessor.analyze_and_log(
                    audio, sample_rate=16_000, label="patient (IndicConformer)"
                )
            except Exception:  # noqa: BLE001 — diagnostics only
                pass

            text = self._run_inference(audio, lang)

            # Post-process: same script normalisation as Whisper path.
            text = (text or "").strip()
            text = self.script_normalizer.normalize_hindi_script(text, lang)

            if not text:
                text = "[no speech detected]"

        finally:
            # Privacy contract — same as audio/handler.py
            try:
                os.unlink(audio_path)
            except Exception:  # noqa: BLE001
                pass

        return TranscriptResult(language=lang, confidence=1.0, text=text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_model(cls) -> None:
        """Load IndicConformer-600M once per process; raise a clear error
        if torch / transformers are not installed."""
        if cls._model is not None:
            return

        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"IndicConformer requires the '{exc.name}' Python package. "
                f"Install it with: pip install torch transformers torchaudio"
            ) from exc

        cache_dir = str(MODEL_CACHE_DIR / "indic-conformer")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        console.print(
            f"[yellow]Loading AI4Bharat IndicConformer "
            f"({INDIC_CONFORMER_MODEL_ID}) — first run downloads ~600 MB to "
            f"{cache_dir}…[/yellow]"
        )
        try:
            cls._model = AutoModel.from_pretrained(
                INDIC_CONFORMER_MODEL_ID,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load IndicConformer from HuggingFace "
                f"({INDIC_CONFORMER_MODEL_ID}): {exc}\n"
                f"Make sure the package transformers>=4.40 is installed "
                f"and you have internet access for the first download."
            ) from exc
        console.print("[green]IndicConformer ready.[/green]")

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Decode any audio file at 16 kHz mono float32.

        Reuses faster-whisper's ffmpeg-backed decoder so we accept the same
        formats as the Whisper path (webm/ogg/wav/mp3) without adding a
        second audio dependency.
        """
        try:
            from faster_whisper.audio import decode_audio
        except ImportError:  # older versions
            from faster_whisper import decode_audio  # type: ignore[no-redef]
        return decode_audio(str(audio_path), sampling_rate=16_000)

    def _run_inference(self, audio: np.ndarray, lang: str) -> str:
        """Run IndicConformer on *audio* and return the raw transcript."""
        import torch

        # The HF model card expects a (1, T) float tensor sampled at 16 kHz.
        wav = torch.from_numpy(audio).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        decoder = (INDIC_CONFORMER_DECODER or "rnnt").lower()
        if decoder not in ("ctc", "rnnt"):
            decoder = "rnnt"

        # Per the AI4Bharat HF model card the model is callable as
        #   model(waveform, language_code, decoder_type)
        # and returns the transcribed string.
        with torch.no_grad():
            result = self._model(wav, lang, decoder)

        # Some checkpoints return a dict; normalise to a string.
        if isinstance(result, dict):
            for k in ("text", "transcription", "result"):
                if k in result and isinstance(result[k], str):
                    return result[k]
            return str(next(iter(result.values()), ""))
        return str(result)
