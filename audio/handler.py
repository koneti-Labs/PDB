"""
audio/handler.py

AudioHandler -- the single, authoritative transcription entry point.

Privacy contract (Phase 1 -> forever, never relaxes):
1. Audio arrives as a temp-file Path from Recorder.record().
2. transcribe() is the ONLY function that reads that file.
3. os.unlink(path) executes in a finally block --
   guaranteed to run even when transcription raises an exception.
4. No audio path is logged to stdout/stderr at INFO level or above.
5. A unit test (tests/test_handler.py) asserts the file is gone
   after transcribe() returns OR raises.

Detection strategy (one-pass, with constrained fallback):
Pass 1 -- transcribe() without specifying language (language=None).
  faster-whisper auto-detects language and returns the full probability
  distribution in TranscriptionInfo.all_language_probs.
  constrain_and_renormalize() masks to our 5 supported codes.

If auto-detected language == constrained language: done in one encode.
If they differ (e.g. Urdu detected instead of Hindi): re-transcribe
  locked to the constrained code (rare but important for accuracy).

Speed knobs (config/settings.py):
  WHISPER_BEAM_SIZE=1            greedy decoding, ~4-5x faster than beam=5
  WHISPER_VAD_FILTER=True        pre-filter silence (set False to skip)
  WHISPER_CONDITION_ON_PREVIOUS=False  skip prefix conditioning
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import numpy as np
from rich.console import Console

from audio.language_id import constrain_and_renormalize
from audio.preprocessor import AudioPreprocessor
from audio.script_normalizer import ScriptNormalizer
from config.settings import (
    MODEL_CACHE_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_CONDITION_ON_PREVIOUS,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
)

console = Console()


class TranscriptResult(TypedDict):
    language: str      # ISO 639-1 code, always one of {hi, te, kn, en, ta}
    confidence: float  # renormalized, 0-1
    text: str          # raw transcript text


def _is_repeat_hallucination(text: str) -> bool:
    """
    Detect Whisper's repeat-token hallucination pattern.

    On near-silent, noisy, or background-music-only clips, Whisper (especially
    the smaller "base" model used by default) sometimes emits the same short
    fragment over and over — e.g.::

        "هاں பي ال هاں பي ال هاں பي ال هاں பي ال هاں பي ال …"

    Forwarding such gibberish to Gemma 4 wastes ~10–30 s of inference and
    returns nothing useful; the user just sees "—" in the UI.  We flag it
    early and treat the clip as no-speech instead.

    Heuristic
    ---------
    For each window size 1..4, take the first window of N tokens, count how
    often that exact window recurs at multiples of N positions, and declare a
    hallucination if it dominates the transcript (>= 4 repetitions AND >= 80%
    of all tokens).

    Returns
    -------
    True if the text looks like a repeat-token hallucination, False otherwise.
    """
    if not text:
        return False
    tokens = text.split()
    n = len(tokens)
    if n < 12:
        # Too short to call confidently — short messages are often genuine.
        return False
    for w in (1, 2, 3, 4):
        if n < w * 4:
            continue
        window = tuple(tokens[:w])
        # Slide non-overlapping windows of size w across the transcript.
        repeats = 0
        for i in range(0, n - w + 1, w):
            if tuple(tokens[i:i + w]) == window:
                repeats += 1
        if repeats >= 4 and repeats * w >= n * 0.8:
            return True
    return False


class AudioHandler:
    """
    Load Whisper once; transcribe + unconditionally delete in a single call.

    The Whisper model is held at *class level* (process-wide singleton).
    This means every AudioHandler() instance shares the same loaded model —
    critical for the Flask web server where a new AudioHandler is created
    per request.  Without this, Whisper was being re-loaded on every
    request (visible in logs as repeated "Loading Whisper 'base' model...").
    """

    # Class-level model cache: loaded once per Python process, shared across
    # every AudioHandler() instance.  Thread-safe-ish: faster-whisper releases
    # the GIL during inference, and concurrent transcribe() calls on one model
    # are supported by CTranslate2.
    _model = None

    def __init__(self) -> None:
        # Instance attr is unused now; kept for backwards compat with any
        # callers that may have set it directly.
        self.preprocessor = AudioPreprocessor()
        self.script_normalizer = ScriptNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path | str) -> TranscriptResult:
        """
        Transcribe *audio_path*, then unconditionally delete it.

        Parameters
        ----------
        audio_path:
            Path (or string path) to an existing WAV file (16 kHz mono int16),
            as returned by Recorder.record() or _save_temp_audio() in the
            web server. Strings are accepted and coerced to Path.

        Returns
        -------
        TranscriptResult with language, confidence, text.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist before transcription starts.
        Any exception from faster-whisper is re-raised AFTER the file
        has been deleted (the finally block runs first).
        """
        self._ensure_model()

        # Accept str or Path from CLI/web callers (web/server.py passes str).
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load and preprocess audio for better quality
            audio_data = self._load_audio_for_preprocessing(audio_path)

            # Analyze audio quality
            quality_metrics = self.preprocessor.analyze_and_log(
                audio_data, sample_rate=16000, label="patient"
            )

            # Preprocess audio if quality is poor
            if not quality_metrics["is_acceptable"]:
                console.print("[yellow]Applying audio preprocessing to improve quality...[/yellow]")
                preprocessed_audio = self.preprocessor.preprocess(audio_data, sample_rate=16000)
                # Save preprocessed audio back to temp file
                temp_preprocessed = self._save_preprocessed_audio(preprocessed_audio, audio_path)
                transcribe_path = temp_preprocessed
            else:
                transcribe_path = audio_path

            # One-pass: transcribe with language=None so Whisper auto-detects.
            # TranscriptionInfo.all_language_probs gives the full distribution --
            # we apply our 5-language constraint without a separate encode pass.
            _raw = self._model.transcribe(
                str(transcribe_path),
                language=None,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": 500},
                temperature=0.0,
                condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
            )
            segments = _raw[0] if isinstance(_raw, tuple) else _raw
            info     = _raw[1] if isinstance(_raw, tuple) and len(_raw) > 1 else None

            # Build probability dict from TranscriptionInfo when available
            all_language_probs = getattr(info, "all_language_probs", None)
            if all_language_probs:
                prob_dict = {code: float(p) for code, p in all_language_probs}
            else:
                # Fallback: CTranslate2 internal path (extra encode, but safe)
                prob_dict = self._detect_language_probs(audio_path)

            language, confidence = constrain_and_renormalize(prob_dict)
            auto_detected = getattr(info, "language", None)

            if language == auto_detected:
                # Common path: auto-detect already gave us the right language.
                # Consume the generator we already have -- no second encode needed.
                text = " ".join(seg.text for seg in segments).strip()
            else:
                # Constrained language differs (e.g. Whisper said "ur", we want "hi").
                # Re-transcribe locked to the correct language (rare, ~5% of cases).
                _raw2 = self._model.transcribe(
                    str(transcribe_path),
                    language=language,
                    beam_size=WHISPER_BEAM_SIZE,
                    vad_filter=WHISPER_VAD_FILTER,
                    vad_parameters={"min_silence_duration_ms": 500},
                    temperature=0.0,
                    condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
                )
                segs2 = _raw2[0] if isinstance(_raw2, tuple) else _raw2
                text = " ".join(seg.text for seg in segs2).strip()

            # Check if we need to re-transcribe due to script mismatch
            # (e.g., Hindi detected but transcribed in Urdu script or romanized)
            if self.script_normalizer.should_retranscribe(text, language, confidence):
                script_type = self.script_normalizer.detect_script(text)
                console.print(
                    f"[yellow]Detected {script_type} script for {language} - "
                    f"re-transcribing with language locked to '{language}'...[/yellow]"
                )
                _raw3 = self._model.transcribe(
                    str(transcribe_path),
                    language=language,
                    beam_size=WHISPER_BEAM_SIZE,
                    vad_filter=WHISPER_VAD_FILTER,
                    vad_parameters={"min_silence_duration_ms": 500},
                    temperature=0.0,
                    condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
                )
                segs3 = _raw3[0] if isinstance(_raw3, tuple) else _raw3
                text = " ".join(seg.text for seg in segs3).strip()

            # Normalize script for Hindi (convert Urdu script to Devanagari if needed)
            text = self.script_normalizer.normalize_hindi_script(text, language)

            # Defence against Whisper repeat-token hallucination on near-silent
            # or noisy clips.  Symptoms in production logs: "هاں பي ال هاں
            # பي ال هاں பي ال …" repeating endlessly.  If we forward that to
            # Gemma 4 it returns empty output and the user sees "—".
            if _is_repeat_hallucination(text):
                console.print(
                    f"[yellow]⚠ Whisper hallucination detected for '{language}' "
                    f"(repeat-token loop) — treating as no-speech.[/yellow]"
                )
                text = "[no speech detected]"

            if not text:
                text = "[no speech detected]"

        finally:
            # Privacy contract -- runs unconditionally
            os.unlink(audio_path)
            # Also delete preprocessed temp file if it was created
            if 'temp_preprocessed' in locals() and temp_preprocessed != audio_path:
                try:
                    os.unlink(temp_preprocessed)
                except Exception:
                    pass

        return TranscriptResult(language=language, confidence=confidence, text=text)

    def transcribe_locked(self, audio_path: Path | str, language: str) -> TranscriptResult:
        """
        Transcribe *audio_path* with language detection skipped.

        Used for:
          • The doctor turn (language always English).
          • The patient turn whenever the user has explicitly chosen a
            language in the UI (Bridge tab dropdown).
          • The Triage tab when the user selects a specific language.

        Locking Whisper to the target language is the most reliable way to
        get native-script output (Devanagari for Hindi, etc.).  We also run
        the script normaliser at the end as a defence-in-depth step: even
        with the language locked Whisper occasionally emits Arabic/Urdu
        glyphs for Hindi, and the normaliser transliterates those back to
        Devanagari.

        The audio file is still deleted unconditionally in the finally block.
        Accepts str or Path (web server passes str via tempfile.mkstemp).
        """
        self._ensure_model()

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            _raw = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": 500},
                temperature=0.0,
                condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
            )
            segments = _raw[0] if isinstance(_raw, tuple) else _raw
            info     = _raw[1] if isinstance(_raw, tuple) and len(_raw) > 1 else None
            text = " ".join(seg.text for seg in segments).strip()

            # Defence-in-depth: even with the language locked the model can
            # still emit Arabic-script Hindi.  Transliterate to Devanagari
            # so downstream prompts see the patient's words in the expected
            # script.  This is a no-op for languages other than Hindi.
            text = self.script_normalizer.normalize_hindi_script(text, language)

            # Same repeat-token hallucination guard as transcribe().
            if _is_repeat_hallucination(text):
                console.print(
                    f"[yellow]⚠ Whisper hallucination detected for '{language}' "
                    f"(locked) — treating as no-speech.[/yellow]"
                )
                text = "[no speech detected]"

            if not text:
                text = "[no speech detected]"
            confidence = float(getattr(info, "language_probability", 1.0))

        finally:
            os.unlink(audio_path)

        return TranscriptResult(language=language, confidence=confidence, text=text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_model(cls) -> None:
        """
        Lazy-load Whisper once per process.  Downloads to MODEL_CACHE_DIR on
        first call and caches to disk for subsequent runs.

        Device selection is automatic:
          • CUDA GPU present → float16 on GPU (fastest, Kaggle T4/P100/A100)
          • CPU only         → int8 quantized (universal fallback, Pi 5)
        """
        if cls._model is not None:
            return

        from faster_whisper import WhisperModel

        from config.hardware import HARDWARE  # import here to avoid circular deps

        console.print(
            f"[yellow]Loading Whisper '{WHISPER_MODEL_SIZE}' model "
            f"on {HARDWARE.device.upper()} ({HARDWARE.compute_type})…"
            "  (first run downloads model, cached afterwards)[/yellow]"
        )
        cls._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=HARDWARE.device,
            compute_type=HARDWARE.compute_type,
            cpu_threads=HARDWARE.num_cpu_threads,
            download_root=str(MODEL_CACHE_DIR),
        )
        console.print(
            f"[green]Whisper model ready "
            f"({'GPU ⚡' if HARDWARE.has_gpu else 'CPU'}).[/green]"
        )

    def _load_audio_for_preprocessing(self, audio_path: Path) -> np.ndarray:
        """
        Load audio file for preprocessing analysis.

        Returns audio as numpy array (int16 or float32).
        """
        try:
            from faster_whisper.audio import decode_audio
        except ImportError:
            from faster_whisper import decode_audio  # type: ignore[no-redef]

        audio = decode_audio(str(audio_path), sampling_rate=16_000)
        return audio

    def _save_preprocessed_audio(self, audio: np.ndarray, original_path: Path) -> Path:
        """
        Save preprocessed audio to a temporary file.

        Parameters
        ----------
        audio:
            Preprocessed audio as float32 numpy array
        original_path:
            Original audio file path (for naming)

        Returns
        -------
        Path to the temporary preprocessed audio file
        """
        import tempfile

        from scipy.io.wavfile import write as wav_write

        # Convert float32 to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix="_preprocessed.wav",
            dir=original_path.parent,
            delete=False,
        )
        wav_write(tmp.name, 16000, audio_int16)
        tmp.close()

        return Path(tmp.name)

    def _detect_language_probs(self, audio_path: Path) -> dict[str, float]:
        """
        Return Whisper's full language probability distribution as a dict.

        Fallback used when TranscriptionInfo.all_language_probs is unavailable.
        Uses the CTranslate2 encode -> detect_language pipeline to get all
        ~99 language probabilities, not just the top-1.

        Falls back to top-1 detection if the internal API is unavailable.
        """
        import numpy as np

        try:
            from faster_whisper.audio import decode_audio
        except ImportError:
            from faster_whisper import decode_audio  # type: ignore[no-redef]

        audio = decode_audio(str(audio_path), sampling_rate=16_000)

        # Pad or trim to 30s -- Whisper's standard language-detection window
        target_len = 16_000 * 30
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        try:
            # CTranslate2 internal path (full distribution)
            features = self._model.feature_extractor(audio)
            features = features[np.newaxis, :]  # add batch dim -> (1, n_mels, n_frames)

            encoder_output = self._model.model.encode(features, to_cpu=False)
            lang_results = self._model.model.detect_language(encoder_output)

            probs: dict[str, float] = {}
            for item in lang_results[0]:
                # item is (token, prob); token formatted as "<|hi|>" -> "hi"
                token = item[0] if isinstance(item, (tuple, list)) else item
                prob  = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else 0.0
                code = str(token)[2:-2]
                probs[code] = float(prob)

            return probs

        except Exception:
            # Fallback: public detect_language (top-1 only)
            _det = self._model.detect_language(audio)
            lang = _det[0] if isinstance(_det, (tuple, list)) else _det
            prob = _det[1] if isinstance(_det, (tuple, list)) and len(_det) > 1 else 1.0
            return {str(lang): float(prob)}
