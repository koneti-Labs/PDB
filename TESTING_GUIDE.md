# Testing Guide - Bridge Record Patient Fixes

## Quick Test Checklist

### ✅ Test 1: Patient Transcript in Devanagari

**Steps**:
1. Open web UI: `http://localhost:5000`
2. Go to "Bridge" tab
3. Select "Patient language: Hindi"
4. Click "● Record Patient"
5. Speak in Hindi (e.g., "मुझे पेट में दर्द है और बुखार है")
6. Click "■ Stop"

**Expected Result**:
- ✅ Transcript shows Devanagari: "मुझे पेट में दर्द है और बुखार है"
- ❌ NOT romanized: "mujhe pate mein dard hai aur bukhar hai"

**Console Output to Check**:
```
Audio quality (patient):
  SNR: X.X dB
  ...
✓ Audio quality acceptable

[If romanized detected:]
Detected romanized script for hi - re-transcribing with language locked to 'hi'...
```

---

### ✅ Test 2: Doctor-to-Patient Translation

**Steps**:
1. After patient speaks (Test 1), note the detected language
2. Scroll to "Doctor → Patient" section
3. Verify "Translate back to:" is auto-synced to patient's language (e.g., Hindi)
4. Click "● Record Doctor"
5. Speak in English (e.g., "I have checked your temperature. Take this medicine twice a day.")
6. Click "■ Stop"

**Expected Result**:
- ✅ Transcript shows English: "I have checked your temperature..."
- ✅ Translation shows Hindi: "मैंने आपका तापमान जांचा है। यह दवा दिन में दो बार लें।"
- ❌ NOT empty

**Console Output to Check**:
```
Translating doctor (en) to patient (hi)...
Doctor transcript: I have checked your temperature...
Translation result: मैंने आपका तापमान जांचा है...
```

---

### ✅ Test 3: TTS (Text-to-Speech)

**Steps**:
1. After patient speaks and translation appears
2. Click 🔊 button next to "Translation for Doctor (English)"
3. Listen to English audio

**Expected Result**:
- ✅ English translation is read aloud
- ✅ Audio plays without errors

**Steps**:
4. After doctor speaks and translation appears
5. Click 🔊 button next to "Translation for Patient"
6. Listen to Hindi audio

**Expected Result**:
- ✅ Hindi translation is read aloud in Devanagari
- ✅ Audio plays without errors

---

## Detailed Testing Scenarios

### Scenario 1: Poor Audio Quality

**Test**: Record patient with background noise or low volume

**Expected Behavior**:
```
Audio quality (patient):
  SNR: 3.8 dB
  Silence: 8.5%
  RMS: 0.0089
  Peak: 0.28
⚠ Low SNR (3.8 dB) - noisy audio may affect accuracy
⚠ Very low RMS level (0.0089) - audio may be too quiet
Applying audio preprocessing to improve quality...
```

**Result**: Audio should be preprocessed before transcription

---

### Scenario 2: Urdu Script Detection

**Test**: If Whisper outputs Urdu script for Hindi

**Expected Behavior**:
```
Detected arabic script for hi - re-transcribing with language locked to 'hi'...
⚠ Detected Urdu script for Hindi language - applying transliteration to Devanagari
```

**Result**: Text should be converted to Devanagari

---

### Scenario 3: Multiple Languages

**Test**: Record patient in different languages

**Languages to Test**:
- Hindi (hi) → Should show Devanagari
- Telugu (te) → Should show Telugu script
- Kannada (kn) → Should show Kannada script
- Tamil (ta) → Should show Tamil script
- English (en) → Should show Latin script

**Expected**: Each language uses its native script, not romanized

---

## Console Commands for Testing

### Run Unit Tests
```bash
cd c:\Users\ADMIN\Documents\NewProject\Claude_proj\PDB
python -m pytest tests/test_audio_preprocessing.py -v
```

**Expected**: 13 tests pass

### Start Web Server
```bash
python -m web.server
```

**Expected**:
```
PatientDoctorBridge Web UI
Open in browser: http://localhost:5000/
Loading Whisper 'base' model on CPU (int8)...
Whisper model ready (CPU).
```

### Check Ollama Status
```bash
ollama list
```

**Expected**: Should show `gemma4:e2b` model

---

## Troubleshooting Commands

### If Romanized Text Persists

**Check script detection**:
```python
from audio.script_normalizer import ScriptNormalizer
normalizer = ScriptNormalizer()

# Test with your actual transcript
text = "mosaic pate mein dhar dhaha hai"
script = normalizer.detect_script(text)
print(f"Detected script: {script}")  # Should be "romanized"

should_retry = normalizer.should_retranscribe(text, "hi", 0.8)
print(f"Should retranscribe: {should_retry}")  # Should be True
```

### If Translation Empty

**Check translation service**:
```python
from core.engine import GemmaEngine
from translation.service import TranslationService

engine = GemmaEngine()
svc = TranslationService(engine)

# Test translation
result = svc.doctor_to_patient(
    "I have checked the temperature and this seems to be okay.",
    "hi"
)
print(f"Translation: {result}")
```

### If TTS Fails

**Check TTS service**:
```python
from translation.tts import TTSService

svc = TTSService()
audio_path = svc.synthesize("नमस्ते", "hi")
print(f"Audio saved to: {audio_path}")
```

---

## Expected Console Logs (Full Flow)

### Patient Recording:
```
Audio quality (patient):
  SNR: 14.2 dB
  Silence: 12.3%
  RMS: 0.0892
  Peak: 0.71
✓ Audio quality acceptable

Transcribing patient audio...
Detected romanized script for hi - re-transcribing with language locked to 'hi'...
Translating patient (hi) → doctor (en)…
```

### Doctor Recording:
```
Audio quality (doctor):
  SNR: 16.8 dB
  Silence: 10.1%
  RMS: 0.1023
  Peak: 0.65
✓ Audio quality acceptable

Transcribing doctor audio...
Translating doctor (en) to patient (hi)...
Doctor transcript: I have checked the temperature and this seems to be okay, but I am writing a medicine. Take for a two days, come back after two days.
Translation result: मैंने तापमान जांचा है और यह ठीक लग रहा है, लेकिन मैं एक दवा लिख रहा हूं। दो दिन के लिए लें, दो दिन बाद वापस आएं।
```

---

## Success Criteria

✅ **Patient transcript in native script** (Devanagari for Hindi)  
✅ **Doctor-to-patient translation appears** (not empty)  
✅ **TTS works for both directions**  
✅ **Console logs show re-transcription** (if romanized detected)  
✅ **Console logs show translation details**  
✅ **All 13 unit tests pass**  

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Transcript still romanized | Check console for "Detected romanized script" message. If missing, add more Hindi indicators to `_is_romanized_indic()` |
| Translation empty | Check Ollama is running: `ollama list`. Check console logs for translation details |
| TTS not working | Check browser console for errors. Verify `/api/tts` endpoint is accessible |
| Audio quality warnings | Normal for noisy environments. Preprocessing should handle it automatically |
| Re-transcription slow | Normal - takes 2-4s. Only happens when wrong script detected |

---

## Performance Benchmarks

| Operation | Expected Time |
|-----------|---------------|
| Audio quality analysis | 50-100ms |
| Preprocessing (if needed) | 200-500ms |
| Initial transcription | 2-4s |
| Re-transcription (if needed) | 2-4s |
| Translation | 1-3s |
| TTS synthesis | 1-2s |
| **Total (worst case)** | **~10-15s** |
| **Total (typical)** | **~5-8s** |

---

## Next Steps After Testing

1. ✅ Verify patient transcript is in Devanagari
2. ✅ Verify doctor translation appears
3. ✅ Test TTS functionality
4. 📝 Report any remaining issues with console logs
5. 🚀 Deploy to production if all tests pass

---

## Contact & Support

If issues persist after testing:
1. Copy full console logs (both browser and server)
2. Take screenshots of the UI
3. Note the exact steps to reproduce
4. Share the audio file (if possible) for debugging

The enhanced logging will provide detailed information for troubleshooting.
