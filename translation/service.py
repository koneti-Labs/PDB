"""
translation/service.py

TranslationService — the single point of truth for all Gemma 4 translation calls.

Phase 2 implements:
  patient_to_doctor()   — Indic language transcript → clinical English
  doctor_to_patient()   — doctor's English → patient's Indic language

Both methods are pure functions of text + language code; they do not touch
audio files or session state directly (the CLI layer owns that).
"""
from __future__ import annotations

from config.languages import LANGUAGE_DISPLAY
from core.engine import GemmaEngine, InferenceMode
from translation.prompts import DOCTOR_TO_PATIENT_PROMPT, PATIENT_TO_DOCTOR_PROMPT


class TranslationService:
    """
    Wraps GemmaEngine with domain-specific prompt construction.

    Parameters
    ----------
    engine:
        A GemmaEngine instance.  Injected rather than created here so
        tests can substitute a mock without patching module globals.
    """

    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def patient_to_doctor(self, text: str, lang_code: str) -> str:
        """
        Translate patient's transcribed speech to clinical English.

        Parameters
        ----------
        text:
            Raw Whisper transcript in the patient's language.
        lang_code:
            ISO 639-1 code from the 5 supported codes (hi/te/kn/en/ta).

        Returns
        -------
        English translation suitable for the doctor to read.
        """
        lang_name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
        prompt = PATIENT_TO_DOCTOR_PROMPT.format(language=lang_name, text=text)
        return self._engine.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)

    def doctor_to_patient(self, text: str, lang_code: str) -> str:
        """
        Translate the doctor's English response to the patient's language.

        Parameters
        ----------
        text:
            Doctor's words in English (from Whisper transcription).
        lang_code:
            Target language code matching the patient's detected language.

        Returns
        -------
        Translation in the patient's language.
        """
        lang_name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
        prompt = DOCTOR_TO_PATIENT_PROMPT.format(language=lang_name, text=text)
        return self._engine.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)
