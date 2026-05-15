# Phase 1 Manual Evaluation — 5-Language Language ID

**Acceptance bar:** ≥ 9 / 10 correct language identification  
**Model:** faster-whisper `small` (244 M params, int8, CPU)  
**Evaluator:** _________________  
**Date:** _________________  
**Machine:** _________________

---

## Instructions

1. Run `pdb listen` for each utterance below.
2. Speak the sample sentence naturally into the mic for ≥ 3 s.
3. Record the detected language code and confidence in the table.
4. Mark ✓ (correct) or ✗ (wrong) in the Result column.

---

## Evaluation Log

| # | Language | Expected code | Sample utterance | Detected code | Confidence | Result | Notes |
|---|----------|--------------|-----------------|--------------|-----------|--------|-------|
| 1 | Hindi | `hi` | मुझे बुखार है और सिर में दर्द हो रहा है | | | | |
| 2 | Hindi | `hi` | मेरे पेट में बहुत तेज दर्द है, कल से कुछ नहीं खाया | | | | |
| 3 | Telugu | `te` | నాకు తల నొప్పిగా ఉంది మరియు జ్వరం వస్తోంది | | | | |
| 4 | Telugu | `te` | నా కడుపులో నొప్పి ఉంది, రాత్రి నుంచి వాంతులు అవుతున్నాయి | | | | |
| 5 | Kannada | `kn` | ನನಗೆ ತಲೆ ನೋವು ಮತ್ತು ಜ್ವರ ಇದೆ | | | | |
| 6 | Kannada | `kn` | ನನ್ನ ಎದೆಯಲ್ಲಿ ನೋವು ಇದೆ, ಉಸಿರಾಡಲು ಕಷ್ಟ ಆಗುತ್ತಿದೆ | | | | |
| 7 | English | `en` | I have had a fever and body aches since yesterday | | | | |
| 8 | English | `en` | My chest feels tight and I am having trouble breathing | | | | |
| 9 | Tamil | `ta` | எனக்கு காய்ச்சல் மற்றும் தலைவலி இருக்கிறது | | | | |
| 10 | Tamil | `ta` | வயிற்று வலி இருக்கிறது, நேற்று இரவிலிருந்து சாப்பிடவில்லை | | | | |

---

## Summary

| Metric | Value |
|--------|-------|
| Total utterances | 10 |
| Correct language ID | ___ / 10 |
| Pass (≥ 9/10)? | ☐ Yes  ☐ No |
| Average confidence (correct detections) | |
| Average confidence (wrong detections) | |

---

## Per-language breakdown

| Language | Correct | Notes |
|----------|---------|-------|
| Hindi (hi) | __ / 2 | |
| Telugu (te) | __ / 2 | |
| Kannada (kn) | __ / 2 | |
| English (en) | __ / 2 | |
| Tamil (ta) | __ / 2 | |

---

## Failure analysis (fill in if any ✗)

_For each wrong detection, note what the model detected instead, and any context
(background noise, accent, short clip, code-switching)._

| # | Expected | Got | Possible cause |
|---|----------|-----|---------------|
| | | | |

---

## Escalation criteria

If either Kannada or Telugu scores 0/2, escalate to `medium` model:

```bash
# In config/settings.py, change:
WHISPER_MODEL_SIZE = "medium"
# Then re-run this eval
```

If language ID is consistently wrong for a language (detects as a neighbouring
language like Urdu, Marathi), check `audio/language_id.py` — the
`constrain_and_renormalize()` function should mask out those alternatives.
If the problem persists, consider adding the Phase 2 hybrid pipeline
(faster-whisper for detection, IndicConformer-600M for transcription).

---

## Environment

```
Python version:
faster-whisper version:
WHISPER_MODEL_SIZE:
compute_type:
Device:
```
