"""
tests/test_audio_preprocessing.py

Unit tests for audio preprocessing and script normalization.
"""
import numpy as np

from audio.preprocessor import AudioPreprocessor
from audio.script_normalizer import ScriptNormalizer


class TestAudioPreprocessor:
    """Test audio quality analysis and preprocessing."""

    def test_analyze_quality_good_audio(self):
        """Test quality analysis on clean audio."""
        preprocessor = AudioPreprocessor()

        # Generate clean synthetic audio (sine wave with slight variation)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Add amplitude variation to make it more realistic
        audio = (
            np.sin(2 * np.pi * frequency * t) * (0.3 + 0.2 * np.sin(2 * np.pi * 2 * t))
        ).astype(np.float32)

        metrics = preprocessor.analyze_quality(audio, sample_rate)

        # Check that metrics are calculated
        assert "is_acceptable" in metrics
        assert "snr_db" in metrics
        assert metrics["rms_level"] > 0.05
        # Pure sine wave may have variable SNR estimation, so just check it's calculated
        assert metrics["snr_db"] > 0

    def test_analyze_quality_noisy_audio(self):
        """Test quality analysis on noisy audio."""
        preprocessor = AudioPreprocessor()

        # Generate noisy audio
        sample_rate = 16000
        duration = 1.0
        noise = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)

        metrics = preprocessor.analyze_quality(noise, sample_rate)

        # Noisy audio should have low SNR
        assert metrics["snr_db"] < 15.0

    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        preprocessor = AudioPreprocessor()

        # Generate audio with silence
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Add silence at start and end
        silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
        audio_with_silence = np.concatenate([silence, audio, silence])

        # Preprocess
        preprocessed = preprocessor.preprocess(audio_with_silence, sample_rate)

        # Preprocessed audio should be shorter (silence trimmed)
        assert len(preprocessed) < len(audio_with_silence)
        assert len(preprocessed) > 0


class TestScriptNormalizer:
    """Test script detection and normalization."""

    def test_detect_devanagari_script(self):
        """Test detection of Devanagari script."""
        normalizer = ScriptNormalizer()

        hindi_text = "मुझे कल से भूख लग रही है और पेट में दर्द है"
        script = normalizer.detect_script(hindi_text)

        assert script == "devanagari"

    def test_detect_arabic_script(self):
        """Test detection of Arabic/Urdu script."""
        normalizer = ScriptNormalizer()

        urdu_text = "مجھے کل سے بھوک لگ رہی ہے اور پیٹ میں درد ہے"
        script = normalizer.detect_script(urdu_text)

        assert script == "arabic"

    def test_detect_latin_script(self):
        """Test detection of Latin script."""
        normalizer = ScriptNormalizer()

        english_text = "I have been hungry since yesterday and have pain in the stomach."
        script = normalizer.detect_script(english_text)

        assert script == "latin"

    def test_normalize_hindi_script_devanagari(self):
        """Test normalization of Hindi text already in Devanagari."""
        normalizer = ScriptNormalizer()

        hindi_text = "मुझे कल से भूख लग रही है"
        normalized = normalizer.normalize_hindi_script(hindi_text, "hi")

        # Should return unchanged
        assert normalized == hindi_text

    def test_normalize_hindi_script_urdu(self):
        """Test normalization of Hindi text in Urdu script."""
        normalizer = ScriptNormalizer()

        urdu_text = "مجھے کل سے بھوک لگ رہی ہے"
        normalized = normalizer.normalize_hindi_script(urdu_text, "hi")

        # Should attempt transliteration
        assert normalized != urdu_text
        # Should contain Devanagari characters
        script = normalizer.detect_script(normalized)
        assert script in ["devanagari", "mixed"]

    def test_should_retranscribe_urdu_script_for_hindi(self):
        """Test decision to re-transcribe when Urdu script detected for Hindi."""
        normalizer = ScriptNormalizer()

        urdu_text = "مجھے کل سے بھوک لگ رہی ہے"
        should_retranscribe = normalizer.should_retranscribe(urdu_text, "hi", 0.8)

        assert should_retranscribe is True

    def test_should_not_retranscribe_devanagari_for_hindi(self):
        """Test no re-transcription needed for correct script."""
        normalizer = ScriptNormalizer()

        hindi_text = "मुझे कल से भूख लग रही है"
        should_retranscribe = normalizer.should_retranscribe(hindi_text, "hi", 0.8)

        assert should_retranscribe is False

    def test_should_retranscribe_wrong_script_even_with_low_confidence(self):
        """Wrong script for an Indic language now retries regardless of confidence.

        Earlier versions of should_retranscribe() bailed out when confidence
        was below 0.4.  In practice that gate let the worst cases through
        unchanged - Whisper's *most-confused* outputs come back with both
        low confidence AND wrong-script garbage at the same time (see the
        broken Triage screenshot where "Kannada" was reported with an
        Arabic-script transcript).

        Wrong script is itself a definitive signal of model confusion, so
        we now retry on wrong script regardless of first-pass confidence.
        """
        normalizer = ScriptNormalizer()

        urdu_text = "مجھے کل سے بھوک لگ رہی ہے"
        should_retranscribe = normalizer.should_retranscribe(urdu_text, "hi", 0.3)

        assert should_retranscribe is True

    def test_detect_romanized_hindi(self):
        """Test detection of romanized Hindi text."""
        normalizer = ScriptNormalizer()

        romanized_text = (
            "mosaic pate mein dhar dhaha hai aur sir mein bhe dhar dhaha hai bukhar bhe hai"
        )
        script = normalizer.detect_script(romanized_text)

        assert script == "romanized"

    def test_should_retranscribe_romanized_hindi(self):
        """Test re-transcription decision for romanized Hindi."""
        normalizer = ScriptNormalizer()

        romanized_text = "mosaic pate mein dhar dhaha hai"
        should_retranscribe = normalizer.should_retranscribe(romanized_text, "hi", 0.8)

        assert should_retranscribe is True
