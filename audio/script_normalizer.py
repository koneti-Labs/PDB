"""
audio/script_normalizer.py

Script normalization for Hindi transcriptions.

Problem: Whisper sometimes transcribes Hindi speech in Urdu (Arabic) script
instead of Devanagari, even when the language is correctly detected as "hi".

Solution: Post-process Whisper output to ensure Hindi text uses Devanagari script.
If Urdu script is detected, transliterate it to Devanagari.

This module provides:
1. Script detection (Devanagari vs Arabic/Urdu)
2. Urdu-to-Hindi transliteration
3. Mixed script normalization
"""
from __future__ import annotations

from rich.console import Console

console = Console()


class ScriptNormalizer:
    """
    Normalize Hindi transcriptions to Devanagari script.

    Detects if Whisper output is in Urdu (Arabic) script and converts
    it to Devanagari if needed.
    """

    # Unicode ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)  # Devanagari block
    ARABIC_RANGE = (0x0600, 0x06FF)      # Arabic block (used for Urdu)
    ARABIC_SUPPLEMENT = (0x0750, 0x077F)  # Arabic Supplement
    ARABIC_EXTENDED = (0x08A0, 0x08FF)    # Arabic Extended-A

    def __init__(self):
        pass

    def detect_script(self, text: str) -> str:
        """
        Detect the primary script used in text.

        Parameters
        ----------
        text:
            Text to analyze

        Returns
        -------
        Script type: "devanagari", "arabic", "latin", "romanized", or "mixed"
        """
        if not text:
            return "latin"

        devanagari_count = 0
        arabic_count = 0
        latin_count = 0

        for char in text:
            code = ord(char)

            # Check Devanagari
            if self.DEVANAGARI_RANGE[0] <= code <= self.DEVANAGARI_RANGE[1]:
                devanagari_count += 1
            # Check Arabic/Urdu
            elif (self.ARABIC_RANGE[0] <= code <= self.ARABIC_RANGE[1] or
                  self.ARABIC_SUPPLEMENT[0] <= code <= self.ARABIC_SUPPLEMENT[1] or
                  self.ARABIC_EXTENDED[0] <= code <= self.ARABIC_EXTENDED[1]):
                arabic_count += 1
            # Check Latin
            elif (ord('A') <= code <= ord('Z')) or (ord('a') <= code <= ord('z')):
                latin_count += 1

        total_script_chars = devanagari_count + arabic_count + latin_count

        if total_script_chars == 0:
            return "latin"

        # Determine primary script
        if devanagari_count > arabic_count and devanagari_count > latin_count:
            return "devanagari"
        elif arabic_count > devanagari_count and arabic_count > latin_count:
            return "arabic"
        elif latin_count > devanagari_count and latin_count > arabic_count:
            # Check if this is romanized Indic text (Latin script but Indic language)
            if self._is_romanized_indic(text):
                return "romanized"
            return "latin"
        else:
            return "mixed"

    def _is_romanized_indic(self, text: str) -> bool:
        """
        Detect if text is romanized Indic language (Hindi/Telugu/etc in Latin script).

        Uses heuristics like presence of common Indic words in romanized form.
        """
        text_lower = text.lower()

        # Common Hindi words in romanized form
        hindi_indicators = [
            'hai', 'hain', 'mein', 'aur', 'ka', 'ki', 'ke', 'ko', 'se', 'ne',
            'tha', 'thi', 'the', 'hoon', 'ho', 'kya', 'nahin', 'nahi',
            'kaise', 'kahan', 'kyun', 'jab', 'tab', 'yeh', 'woh', 'yahan',
            'dhar', 'dhaha', 'bukhar', 'dard', 'pate', 'pet', 'sir', 'bhe'
        ]

        # Count how many indicators are present
        matches = sum(1 for word in hindi_indicators if word in text_lower)

        # If we find 2+ indicators, likely romanized Hindi
        return matches >= 2

    def normalize_hindi_script(self, text: str, detected_language: str) -> str:
        """
        Normalize Hindi text to Devanagari script.

        If the detected language is Hindi ("hi") but the text is in
        Urdu/Arabic script, attempt to transliterate it to Devanagari.

        Parameters
        ----------
        text:
            Transcribed text from Whisper
        detected_language:
            ISO 639-1 language code (e.g., "hi", "ur", "en")

        Returns
        -------
        Normalized text in Devanagari (if Hindi) or original text
        """
        if not text or detected_language != "hi":
            # Only process Hindi language
            return text

        script = self.detect_script(text)

        if script == "devanagari":
            # Already in correct script
            return text

        if script == "arabic":
            # Urdu script detected for Hindi language - needs transliteration
            console.print(
                "[yellow]⚠ Detected Urdu script for Hindi language - "
                "applying transliteration to Devanagari[/yellow]"
            )
            return self._transliterate_urdu_to_hindi(text)

        if script == "mixed":
            # Mixed script - try to fix Arabic portions
            console.print(
                "[yellow]⚠ Detected mixed script for Hindi - "
                "normalizing to Devanagari[/yellow]"
            )
            return self._transliterate_urdu_to_hindi(text)

        # Latin or other - return as is
        return text

    def _transliterate_urdu_to_hindi(self, text: str) -> str:
        """
        Transliterate Urdu (Arabic script) to Hindi (Devanagari).

        This is a basic phonetic mapping. For production use, consider
        using a library like `indic-transliteration` or an API.

        Note: This is a simplified mapping and may not be perfect.
        For better results, consider using:
        - indic-transliteration library
        - Google Translate API
        - IndicTrans2 model

        Parameters
        ----------
        text:
            Text in Urdu script

        Returns
        -------
        Text transliterated to Devanagari
        """
        # Basic Urdu to Hindi character mapping
        # This is a simplified mapping - a full implementation would need
        # more sophisticated handling of diacritics, conjuncts, etc.

        urdu_to_devanagari = {
            # Vowels
            'ا': 'अ',
            'آ': 'आ',
            'ب': 'ब',
            'پ': 'प',
            'ت': 'त',
            'ٹ': 'ट',
            'ث': 'स',
            'ج': 'ज',
            'چ': 'च',
            'ح': 'ह',
            'خ': 'ख',
            'د': 'द',
            'ڈ': 'ड',
            'ذ': 'ज़',
            'ر': 'र',
            'ڑ': 'ड़',
            'ز': 'ज़',
            'ژ': 'झ',
            'س': 'स',
            'ش': 'श',
            'ص': 'स',
            'ض': 'ज़',
            'ط': 'त',
            'ظ': 'ज़',
            'ع': '',
            'غ': 'ग़',
            'ف': 'फ',
            'ق': 'क़',
            'ک': 'क',
            'گ': 'ग',
            'ل': 'ल',
            'م': 'म',
            'ن': 'न',
            'ں': 'ं',
            'و': 'व',
            'ہ': 'ह',
            'ھ': 'ह',
            'ء': '',
            'ی': 'य',
            'ے': 'े',
            # Diacritics
            'َ': 'ा',
            'ِ': 'ि',
            'ُ': 'ु',
            'ً': 'ां',
            'ٍ': 'िं',
            'ٌ': 'ुं',
            'ّ': '्',
            'ْ': '',
        }

        # Apply character-by-character transliteration
        result = []
        for char in text:
            if char in urdu_to_devanagari:
                result.append(urdu_to_devanagari[char])
            else:
                result.append(char)

        transliterated = ''.join(result)

        # Log the transliteration
        console.print(f"[dim]Transliterated: {text[:50]}... → {transliterated[:50]}...[/dim]")

        return transliterated

    def should_retranscribe(self, text: str, detected_language: str, confidence: float) -> bool:
        """
        Determine if audio should be re-transcribed with language locked to native script.

        Criteria for re-transcription:
        1. Detected language is one of our Indic codes (hi/te/kn/ta).
        2. The transcript is in the wrong script for that language —
           either Arabic/Urdu script or romanised Latin.

        Note on confidence:
            Earlier versions of this function required confidence > 0.4 before
            retrying.  In practice that caused the most-broken cases to slip
            through: when Whisper is genuinely confused it returns LOW
            confidence AND wrong-script garbage at the same time (the Triage
            "Kannada with Arabic-script gibberish" bug).  Wrong script is
            itself a hard signal of model confusion, so we now retry
            unconditionally whenever the script does not match the
            constrained language.  A locked-language second pass is the best
            available remedy regardless of the first-pass confidence.

        Parameters
        ----------
        text:
            Transcribed text from the first pass.
        detected_language:
            Constrained ISO 639-1 code chosen for this clip.
        confidence:
            Detection confidence (0-1).  Kept in the signature for
            backwards compatibility with existing callers and tests.

        Returns
        -------
        True if the clip should be re-transcribed with the language locked.
        """
        # Confidence is intentionally unused — see the docstring above.
        del confidence

        if detected_language not in ["hi", "te", "kn", "ta"]:
            # Only re-transcribe for Indic languages.
            return False

        script = self.detect_script(text)

        # Re-transcribe if wrong script detected:
        #   • Arabic / Urdu script for any Indic language
        #   • Romanised (Latin) script for any Indic language
        return script in ["arabic", "romanized"]
