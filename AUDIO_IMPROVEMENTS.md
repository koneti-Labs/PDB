# Audio Quality Improvements for Hindi Transcription

> **Status update (current revision).** The script-recovery layer described in this document is still in place and still useful for Whisper, but the project now also exposes an opt-in **AI4Bharat IndicConformer-600M** ASR backend (see [Optional ASR backend: IndicConformer-600M](#optional-asr-backend-indicconformer-600m) at the end). IndicConformer produces native-script output for Hindi / Telugu / Kannada / Tamil on the first pass, so when it's selected the script normaliser is essentially a no-op. Whisper remains the default; the script-recovery pipeline below is what makes the default path work reliably for Indic languages.

## Problem Statement

The Bridge Record Patient feature was experiencing issues where:
1. Hindi speech was being correctly detected as "hi" language
2. However, the transcription was appearing in **Urdu (Arabic) script** instead of **Devanagari (Hindi) script**
3. Audio quality issues were not being analyzed before transcription
4. **Update:** noisy / silent clips were also producing **repeat-token hallucinations** like `هاں பي ال هاں பي ال …`, which would otherwise be forwarded to Gemma 4 and waste ~30 s of inference returning "unintelligible" — these are now caught before they leave the audio handler.

## Solution Overview

Implemented a **two-layer solution** to improve transcription accuracy, plus a third safety net for Whisper hallucinations:

### 1. Audio Preprocessing (`audio/preprocessor.py`)

**Purpose**: Analyze and improve audio quality before it reaches the Whisper model.

**Features**:
- **Signal-to-Noise Ratio (SNR) estimation**: Detects noisy audio
- **Silence detection and trimming**: Removes leading/trailing silence
- **Audio normalization**: Normalizes RMS level for consistent volume
- **Quality metrics**: Provides detailed quality analysis with warnings

**Quality Thresholds**:
- SNR > 10 dB: Good quality
- SNR 5-10 dB: Acceptable
- SNR < 5 dB: Poor quality (triggers preprocessing)
- Silence ratio > 70%: Warning issued

**Preprocessing Steps**:
1. Convert to float32
2. Trim leading/trailing silence
3. Apply gentle noise gate (reduces background noise)
4. Normalize RMS level to 0.1
5. Soft clip to avoid distortion

### 2. Script Normalization (`audio/script_normalizer.py`)

**Purpose**: Ensure Hindi transcriptions use Devanagari script, not Urdu/Arabic script.

**Features**:
- **Script detection**: Identifies Devanagari, Arabic, Latin, or mixed scripts
- **Urdu-to-Hindi transliteration**: Converts Arabic script to Devanagari
- **Re-transcription logic**: Triggers Whisper re-run with language locked to "hi" when needed

**Unicode Ranges Detected**:
- Devanagari: U+0900 to U+097F
- Arabic (Urdu): U+0600 to U+06FF, U+0750 to U+077F, U+08A0 to U+08FF

**Decision Logic for Re-transcription**:
- Detected language is Hindi ("hi")
- Text is in Urdu/Arabic script
- Confidence is > 0.4 (high enough to trust the language detection)

### 3. Updated AudioHandler (`audio/handler.py`)

**Integration Flow**:

```
1. Load audio file
   ↓
2. Analyze audio quality (SNR, silence, RMS, peak)
   ↓
3. If quality is poor → Apply preprocessing
   ↓
4. Transcribe with Whisper (auto-detect language)
   ↓
5. Apply language constraint (5 supported languages)
   ↓
6. Check if script mismatch (Hindi detected but Urdu script)
   ↓
7. If mismatch → Re-transcribe with language="hi"
   ↓
8. Normalize script (Urdu → Devanagari if needed)
   ↓
9. Return final transcript
```

## Files Created/Modified

### New Files:
1. **`audio/preprocessor.py`** (374 lines)
   - AudioPreprocessor class
   - Quality analysis methods
   - Preprocessing pipeline

2. **`audio/script_normalizer.py`** (252 lines)
   - ScriptNormalizer class
   - Script detection
   - Urdu-to-Hindi transliteration

3. **`tests/test_audio_preprocessing.py`** (159 lines)
   - 11 unit tests
   - Tests for both preprocessor and script normalizer

### Modified Files:
1. **`audio/handler.py`**
   - Added preprocessing integration
   - Added script normalization
   - Added re-transcription logic for script mismatches

## Testing

All 11 tests pass successfully:

```bash
pytest tests/test_audio_preprocessing.py -v
```

**Test Coverage**:
- ✅ Audio quality analysis (good audio)
- ✅ Audio quality analysis (noisy audio)
- ✅ Audio preprocessing
- ✅ Devanagari script detection
- ✅ Arabic/Urdu script detection
- ✅ Latin script detection
- ✅ Hindi script normalization (already Devanagari)
- ✅ Hindi script normalization (Urdu to Devanagari)
- ✅ Re-transcription decision (Urdu script for Hindi)
- ✅ No re-transcription (correct script)
- ✅ No re-transcription (low confidence)

## Expected Behavior

### Before:
```
Patient speaks Hindi → Whisper detects "hi" → Transcribes in Urdu script
Result: "مجھے کل سے بھوک لگ رہی ہے" (Arabic script)
```

### After:
```
Patient speaks Hindi → Audio quality check → Preprocessing (if needed) 
→ Whisper detects "hi" → Script check detects Urdu 
→ Re-transcribe with language="hi" → Normalize to Devanagari
Result: "मुझे कल से भूख लग रही है" (Devanagari script)
```

## Console Output

The system now provides detailed feedback:

```
Audio quality (patient):
  SNR: 12.3 dB
  Silence: 15.2%
  RMS: 0.0845
  Peak: 0.67
✓ Audio quality acceptable
```

Or if quality is poor:

```
Audio quality (patient):
  SNR: 4.2 dB
  Silence: 8.5%
  RMS: 0.0123
  Peak: 0.34
⚠ Low SNR (4.2 dB) - noisy audio may affect accuracy
⚠ Very low RMS level (0.0123) - audio may be too quiet
Applying audio preprocessing to improve quality...
```

If script mismatch is detected:

```
⚠ Detected Urdu script for Hindi language - applying transliteration to Devanagari
```

Or if re-transcription is needed:

```
Detected Urdu script for Hindi - re-transcribing with language locked to 'hi'...
```

## Performance Impact

- **Audio quality analysis**: ~50-100ms (negligible)
- **Preprocessing** (when needed): ~200-500ms
- **Re-transcription** (when needed): ~2-4s (same as initial transcription)

Most requests will NOT trigger preprocessing or re-transcription, so the typical overhead is minimal.

## Future Improvements

1. **Better transliteration**: Consider using `indic-transliteration` library or IndicTrans2 model for more accurate Urdu→Hindi conversion
2. **Noise reduction**: Add spectral subtraction or Wiener filtering for better noise removal
3. **Voice activity detection**: Use more sophisticated VAD models
4. **Language-specific preprocessing**: Different preprocessing strategies for different languages

## Dependencies

All required dependencies are already in `pyproject.toml`:
- `numpy` - Audio processing
- `scipy` - WAV file I/O
- `faster-whisper` - Default Whisper ASR backend
- `rich` - Console output
- `torch` + `transformers` - Required by the TTS read-aloud feature and by the optional IndicConformer ASR backend

The optional IndicConformer backend additionally needs `torchaudio`, available via the `[indic-asr]` extra: `pip install -e .[indic-asr]`.

---

## Whisper Hallucination Guard

Even with the preprocessing and script-recovery layers above, Whisper (especially the small `base` model used by default) sometimes produces a different failure mode on noisy / silent / background-music clips:

```
هاں பي ال هاں பي ال هاں பي ال هاں பي ال هاں பي ال هاں பي ال …
```

It locks onto a 1-4 token window and emits it forever. If we forward that to Gemma 4, the LLM wastes 10-30 seconds of inference and returns nothing useful; the user just sees `—` in the UI.

### Detection

`audio/handler.py::_is_repeat_hallucination()` scans the transcript for this pattern:

1. Token-split the transcript on whitespace.
2. For each window size `w ∈ {1, 2, 3, 4}`, slide non-overlapping windows of `w` tokens across the transcript and count how many of them match the first window exactly.
3. If `repeats >= 4` **and** `repeats * w >= 0.8 * total_tokens`, the transcript is dominated by a single repeating fragment → flag as hallucination.
4. Short transcripts (< 12 tokens) are exempt; genuine short messages would otherwise false-positive.

### Recovery

When the guard fires, the handler:

- Replaces the transcript with the `[no speech detected]` sentinel.
- Logs `⚠ Whisper hallucination detected for '<lang>' (repeat-token loop) — treating as no-speech.` at `yellow`.

The web server's `_handle_audio_translation()` and the triage SSE endpoint both treat that sentinel as an early-exit: they skip the Gemma 4 call entirely and return a structured `warning` so the UI can show "No clear speech was detected in the recording — please record again, speak closer to the microphone, and reduce background noise. If Auto-detect keeps misidentifying the language, override it with the dropdown."

---

## Optional ASR backend: IndicConformer-600M

`audio/indic_conformer.py` adds **AI4Bharat's IndicConformer-600M multilingual** ASR as an opt-in alternative to Whisper. It exposes the same `TranscriptResult` shape so the rest of the pipeline doesn't care which backend produced the text.

### Why it's worth offering

Whisper was trained on a heavily English-skewed multilingual corpus and produces several recurring failure modes on Indian languages: romanised-Latin output for Hindi, Urdu-script output for Hindi, dialect confusion among Telugu / Kannada / Tamil, and the repeat-token hallucinations above on noisy clips. AI4Bharat trained IndicConformer specifically on Indian languages with high-quality crowd-sourced corpora — in practice it produces native-script output (Devanagari / Telugu / Kannada / Tamil) on the first pass, handles dialectal variation and code-mixing far more gracefully, and effectively bypasses the entire script-recovery layer described above.

### Why it's opt-in rather than the default

- The model is ~600 MB on disk (vs. ~140 MB for `whisper-base`).
- CPU inference is roughly 2-3× slower per request.
- It does **not** auto-detect the source language — it expects to be told.

### Hybrid auto-detect

To keep the "Auto-detect" UX from the Bridge / Triage tabs without forcing the user to pick a language manually, the web server runs a two-stage dispatch when the user picks IndicConformer + Auto-detect:

1. Whisper performs a constrained language-detection pass (it would have transcribed too — that work is acceptable as a cheap detector).
2. IndicConformer is then called with the detected language locked in for the actual transcript.

The audio file is copied (not re-saved from the FileStorage stream, which would be empty by then) for the Whisper pass; IndicConformer consumes the original tmp file.

### Compliance with the Kaggle Gemma 4 Impact Challenge

The challenge mandates Gemma 4 for **LLM inference** (translation, triage extraction, prescription OCR). ASR is a separate subsystem — Whisper has always been a non-Gemma ASR in this project — and IndicConformer is the same category. All inference still runs locally; no cloud APIs are called. See the README "Models used and why each is compliant" section for the full audit.

## Backward Compatibility

✅ All changes are backward compatible:
- Existing code continues to work
- Preprocessing is automatic (no API changes)
- Script normalization is transparent
- Tests pass without modifications

## Conclusion

The Bridge Record Patient feature now has:
1. ✅ Audio quality analysis before transcription
2. ✅ Automatic preprocessing for poor quality audio
3. ✅ Script detection and normalization for Hindi
4. ✅ Re-transcription logic to force correct script
5. ✅ Comprehensive test coverage

This should **significantly improve** the accuracy of Hindi transcriptions and prevent Urdu script output.
