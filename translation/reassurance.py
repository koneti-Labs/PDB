"""
translation/reassurance.py

ReassuranceService -- translates emergency comfort phrases to patient language.

Phase 5: uses GemmaEngine.emergency_reassurance() with FAST_TRANSLATION
(gemma4:e2b) for the fastest possible response in an emergency.

Built-in phrase bank covers the most critical clinical scenarios.
"""
from __future__ import annotations

from config.languages import LANGUAGE_DISPLAY
from core.engine import GemmaEngine

# Ordered from most urgent to informational
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


class ReassuranceService:
    """
    Translates built-in emergency phrases to the patient's language.

    Parameters
    ----------
    engine : GemmaEngine
        Injected for testability.
    """

    def __init__(self, engine: GemmaEngine) -> None:
        self._engine = engine

    @staticmethod
    def phrases() -> list[tuple[str, str]]:
        """Return the full phrase bank as (category, english_phrase) pairs."""
        return REASSURANCE_PHRASES

    def translate(self, phrase: str, target_lang: str) -> str:
        """
        Translate *phrase* into *target_lang*.

        Parameters
        ----------
        phrase : str
            English phrase (typically from REASSURANCE_PHRASES).
        target_lang : str
            ISO 639-1 code (hi/te/kn/en/ta).

        Returns
        -------
        str
            Translated phrase in the patient's language.
        """
        if target_lang == "en":
            return phrase
        return self._engine.emergency_reassurance(phrase, target_lang)

    @staticmethod
    def language_display(lang_code: str) -> str:
        return LANGUAGE_DISPLAY.get(lang_code, lang_code)
