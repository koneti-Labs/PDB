"""
tests/test_language_id.py

Unit tests for audio/language_id.py — constrain-and-renormalize logic.

No model loading, no audio files, no network.  Pure function tests.
"""
from __future__ import annotations

import pytest

from audio.language_id import constrain_and_renormalize
from config.languages import SUPPORTED_LANG_CODES


class TestConstrainAndRenormalize:
    # ------------------------------------------------------------------ basic correctness

    def test_output_language_always_in_supported_set(self) -> None:
        """The returned language code is always one of the 5 supported codes."""
        probs = {"hi": 0.4, "ur": 0.3, "pa": 0.2, "en": 0.1}
        lang, _ = constrain_and_renormalize(probs)
        assert lang in SUPPORTED_LANG_CODES

    def test_confidence_is_between_0_and_1(self) -> None:
        probs = {"hi": 0.4, "en": 0.2, "te": 0.1, "kn": 0.05, "ta": 0.05, "ur": 0.2}
        _, conf = constrain_and_renormalize(probs)
        assert 0.0 <= conf <= 1.0

    # ------------------------------------------------------------------ winner selection

    def test_dominant_indic_wins(self) -> None:
        """Telugu wins when it has the highest probability among the 5."""
        probs = {
            "te": 0.60,
            "hi": 0.10,
            "en": 0.05,
            "kn": 0.03,
            "ta": 0.02,
            "ur": 0.10,   # noise — must be masked out
            "mr": 0.10,   # noise
        }
        lang, conf = constrain_and_renormalize(probs)
        assert lang == "te"
        # Renormalized: te=0.60 / (0.60+0.10+0.05+0.03+0.02) = 0.60/0.80 = 0.75
        assert abs(conf - 0.60 / 0.80) < 1e-6

    def test_hindi_beats_urdu_after_masking(self) -> None:
        """Hindi wins even when Urdu has higher raw probability."""
        probs = {"ur": 0.55, "hi": 0.30, "en": 0.10, "pa": 0.05}
        lang, _ = constrain_and_renormalize(probs)
        assert lang == "hi"  # ur and pa are masked; hi is the highest supported

    # ------------------------------------------------------------------ renormalization math

    def test_renormalization_is_correct(self) -> None:
        """Manually verify the renormalization arithmetic."""
        probs = {"hi": 0.4, "te": 0.2}   # only two supported langs present
        lang, conf = constrain_and_renormalize(probs)
        assert lang == "hi"
        # 0.4 / (0.4 + 0.2) = 0.4 / 0.6
        assert abs(conf - (0.4 / 0.6)) < 1e-6

    def test_renorm_when_all_five_present(self) -> None:
        """All 5 codes present — winner takes the correct renormalized share."""
        probs = {"hi": 0.50, "te": 0.20, "kn": 0.10, "ta": 0.10, "en": 0.10}
        # probs already sum to 1.0, so renormalized hi = 0.50 / 1.0 = 0.50
        lang, conf = constrain_and_renormalize(probs)
        assert lang == "hi"
        assert abs(conf - 0.50) < 1e-6

    # ------------------------------------------------------------------ edge / fallback cases

    def test_no_supported_languages_returns_english_zero(self) -> None:
        """If none of the 5 codes appear, fallback is ('en', 0.0)."""
        probs = {"ur": 0.5, "mr": 0.3, "pa": 0.2}
        lang, conf = constrain_and_renormalize(probs)
        assert lang == "en"
        assert conf == 0.0

    def test_empty_probs_returns_english_zero(self) -> None:
        lang, conf = constrain_and_renormalize({})
        assert lang == "en"
        assert conf == 0.0

    def test_all_zero_probabilities_returns_english_zero(self) -> None:
        probs = {"hi": 0.0, "te": 0.0}
        lang, conf = constrain_and_renormalize(probs)
        assert lang == "en"
        assert conf == 0.0

    # ------------------------------------------ parametrize over 5 languages

    @pytest.mark.parametrize("target", sorted(SUPPORTED_LANG_CODES))
    def test_each_supported_language_can_win(self, target: str) -> None:
        """Each of the 5 supported codes can be the winner."""
        probs: dict[str, float] = {target: 0.9}
        for other in SUPPORTED_LANG_CODES:
            if other != target:
                probs[other] = 0.01
        lang, conf = constrain_and_renormalize(probs)
        assert lang == target
        assert conf > 0.5
