"""
audio/tts.py

TTSService — server-side text-to-speech using Facebook MMS-TTS.

Facebook MMS-TTS (Massively Multilingual Speech) covers 1000+ languages
including Hindi, Telugu, Tamil, and Kannada.  Models (~80 MB each) are
downloaded once to MODEL_CACHE_DIR on first use and then cached in RAM
between requests.

No OS voice packs, no cloud API calls, no browser configuration required.
Works on any OS (Windows, Linux, macOS) straight out of the box.

Supported language codes: hi, te, ta, kn, en

Usage
-----
    svc = TTSService()
    wav_bytes, sr = svc.synthesize("నమస్కారం", "te")
    # wav_bytes is a valid audio/wav byte string ready to stream to the browser.
"""
from __future__ import annotations

import io
import logging
from typing import ClassVar

import numpy as np
import scipy.io.wavfile as wavfile

from config.settings import MODEL_CACHE_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# ISO 639-1 → HuggingFace MMS-TTS model ID (VITS architecture)
MMS_MODEL_IDS: dict[str, str] = {
    "hi": "facebook/mms-tts-hin",   # Hindi
    "te": "facebook/mms-tts-tel",   # Telugu
    "ta": "facebook/mms-tts-tam",   # Tamil
    "kn": "facebook/mms-tts-kan",   # Kannada
    "en": "facebook/mms-tts-eng",   # English
}

_MMS_CACHE_DIR = str(MODEL_CACHE_DIR / "mms-tts")


# ---------------------------------------------------------------------------
# TTSService
# ---------------------------------------------------------------------------

class TTSService:
    """
    Server-side TTS using Facebook MMS-TTS via HuggingFace Transformers.

    Models are lazy-loaded on first request per language and cached for all
    subsequent calls.  First call per language takes 5-30 s (model download
    + load); thereafter synthesis is typically < 2 s for short clinical phrases.

    The model files are stored under MODEL_CACHE_DIR / "mms-tts" so they
    survive server restarts.
    """

    # Class-level cache: {lang_code: (VitsModel, AutoTokenizer)}
    # Class-level so all TTSService() instances share one copy in RAM.
    _cache: ClassVar[dict[str, tuple]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, lang_code: str) -> tuple[bytes, int]:
        """
        Convert *text* to WAV audio bytes in *lang_code*.

        Parameters
        ----------
        text : str
            Text to speak.  Must be in the script of *lang_code*
            (e.g. Telugu Unicode for "te").  If you pass an empty string
            the model returns silence — callers should guard against that.
        lang_code : str
            ISO 639-1 code — one of: hi / te / ta / kn / en.

        Returns
        -------
        (wav_bytes, sample_rate)
            wav_bytes   : raw WAV file content, ready to stream as audio/wav.
            sample_rate : int — typically 16000 for MMS-TTS models.

        Raises
        ------
        ValueError
            If *lang_code* is not in MMS_MODEL_IDS.
        RuntimeError
            If the HuggingFace model download or PyTorch inference fails.
        """
        if lang_code not in MMS_MODEL_IDS:
            raise ValueError(
                f"Unsupported TTS language '{lang_code}'. "
                f"Supported codes: {sorted(MMS_MODEL_IDS)}"
            )

        import torch

        model, tokenizer = self._load(lang_code)

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            # VitsModel returns a ModelOutput; .waveform shape: (batch, T)
            output = model(**inputs)
            waveform = output.waveform  # float32, values in [-1.0, 1.0]

        wav_np = waveform.squeeze(0).cpu().numpy()  # (T,) float32

        # Convert float32 → int16 for maximum WAV compatibility across browsers
        wav_int16 = (wav_np * 32767.0).clip(-32768, 32767).astype(np.int16)

        sample_rate: int = model.config.sampling_rate

        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, wav_int16)
        return buf.getvalue(), sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load(cls, lang_code: str) -> tuple:
        """
        Lazy-load and cache the MMS-TTS model + tokenizer for *lang_code*.

        Thread safety note: on the very first simultaneous call for the same
        language two threads may both load the model in parallel.  The second
        write to _cache overwrites the first with an identical model — harmless
        but wastes 5-15 s and ~160 MB RAM transiently.  For a single-server
        Flask deployment (threaded=True) this is acceptable.
        """
        if lang_code in cls._cache:
            return cls._cache[lang_code]

        from transformers import AutoTokenizer, VitsModel

        model_id = MMS_MODEL_IDS[lang_code]
        log.info(
            "Loading MMS-TTS model %s into RAM "
            "(first use — downloading to %s if not cached)…",
            model_id,
            _MMS_CACHE_DIR,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=_MMS_CACHE_DIR
        )
        model = VitsModel.from_pretrained(
            model_id, cache_dir=_MMS_CACHE_DIR
        )
        model.eval()

        cls._cache[lang_code] = (model, tokenizer)
        log.info("MMS-TTS model %s ready.", model_id)
        return cls._cache[lang_code]

    @classmethod
    def prewarm(cls, lang_codes: list[str] | None = None) -> dict[str, bool]:
        """
        Eagerly load TTS models for *lang_codes* into RAM in parallel.

        Why parallel?
        -------------
        Loading each MMS-TTS model is a blocking I/O + CPU task:
          • First run: HTTP download (~80 MB per model) — pure I/O, GIL released.
          • Subsequent runs: disk read + PyTorch deserialization — GIL mostly released
            during C-extension work.

        Without this, models load serially on first use:
          5 langs × ~10–30 s each = up to 150 s of user-visible lag.
        With ThreadPoolExecutor (max_workers = num_langs):
          All 5 langs load simultaneously = just the slowest single model's time.

        Parameters
        ----------
        lang_codes : list[str] or None
            ISO 639-1 codes to preload.  None → all codes in MMS_MODEL_IDS.

        Returns
        -------
        dict[str, bool]
            Map of lang_code → True if loaded successfully, False on error.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        codes = lang_codes if lang_codes is not None else list(MMS_MODEL_IDS.keys())
        # Filter out already-cached languages so we don't waste threads on no-ops.
        to_load = [c for c in codes if c not in cls._cache and c in MMS_MODEL_IDS]

        if not to_load:
            log.info("TTS prewarm: all requested languages already cached.")
            return {c: True for c in codes}

        results: dict[str, bool] = {c: True for c in codes if c not in to_load}

        log.info(
            "TTS prewarm: loading %d language model(s) in parallel: %s",
            len(to_load), ", ".join(to_load),
        )

        # Each worker calls _load(), which downloads (if needed) and
        # caches the model.  The GIL is released during both the HTTP
        # download and the PyTorch C++ model load, so true parallelism
        # is achieved even without multiprocessing.
        with ThreadPoolExecutor(
            max_workers=len(to_load),
            thread_name_prefix="tts-prewarm",
        ) as pool:
            futures = {pool.submit(cls._load, code): code for code in to_load}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    future.result()
                    results[code] = True
                    log.info("TTS prewarm: %s ready.", code)
                except Exception as exc:
                    results[code] = False
                    log.warning("TTS prewarm: %s failed — %s", code, exc)

        return results
