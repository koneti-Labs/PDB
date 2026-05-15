# Phase 2 Manual Evaluation — Two-Way Translation

**Acceptance bar:** Patient→Doctor ≥ 8/10 medically accurate · Doctor→Patient ≥ 8/10 patient-comprehensible  
**Model:** gemma4:e2b via Ollama (local, no cloud)  
**Evaluator:** _________________  
**Date:** _________________

---

## Setup

```bash
pdb bridge
```

Run one full exchange per row below. For each exchange:
1. Speak the **Patient utterance** when prompted for patient input.
2. Speak the **Doctor response** when prompted for doctor input.
3. Record what Gemma 4 produced in the output columns.
4. Score each translation 0 (wrong/missing info) or 1 (medically accurate and complete).

---

## Part A — Patient → Doctor (Indic → English)

| # | Language | Patient utterance | Expected English summary | Actual Gemma 4 output | Score | Notes |
|---|----------|------------------|--------------------------|----------------------|-------|-------|
| 1 | Hindi `hi` | मुझे तीन दिन से बुखार है, 102 डिग्री तक | 3-day fever, 102°F | | /1 | |
| 2 | Hindi `hi` | मेरे सीने में दर्द है, सांस लेने में दिक्कत हो रही है | Chest pain, breathing difficulty | | /1 | |
| 3 | Telugu `te` | నాకు రెండు రోజుల నుండి వాంతులు మరియు విరేచనాలు అవుతున్నాయి | 2-day vomiting and diarrhoea | | /1 | |
| 4 | Telugu `te` | నా కాలు మోకాలు నొప్పి ఉంది, నడవడానికి కష్టంగా ఉంది | Knee pain, difficulty walking | | /1 | |
| 5 | Kannada `kn` | ನನಗೆ ತಲೆ ತಿರುಗುತ್ತಿದೆ ಮತ್ತು ವಾಂತಿ ಆಗುತ್ತಿದೆ | Dizziness and vomiting | | /1 | |
| 6 | Kannada `kn` | ನನ್ನ ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮಾಡುವಾಗ ತುಂಬಾ ನೋವು ಇದೆ | Painful urination | | /1 | |
| 7 | English `en` | I have been having severe headaches every morning for a week | Week of morning headaches, severe | | /1 | |
| 8 | English `en` | My blood sugar was 280 this morning and I feel very weak | High blood sugar 280, weakness | | /1 | |
| 9 | Tamil `ta` | எனக்கு இரண்டு நாட்களாக தொண்டை வலி மற்றும் காய்ச்சல் இருக்கிறது | 2-day sore throat and fever | | /1 | |
| 10 | Tamil `ta` | என் குழந்தைக்கு நேற்று இரவிலிருந்து வயிற்று வலி இருக்கிறது | Child's stomach pain since last night | | /1 | |

**Part A score: ___ / 10**

---

## Part B — Doctor → Patient (English → Indic)

For each row, after the Part A exchange above (same session), speak the doctor response and check the back-translation.

| # | Target lang | Doctor response (English) | Key info to preserve | Actual Gemma 4 output | Score | Notes |
|---|------------|--------------------------|---------------------|----------------------|-------|-------|
| 1 | Hindi `hi` | Take paracetamol 500 mg every 8 hours. Drink plenty of fluids and rest. | Paracetamol 500mg, every 8h, fluids, rest | | /1 | |
| 2 | Hindi `hi` | I am referring you to a cardiologist. Please go to the emergency room now. | Referral to cardiologist, go to emergency now | | /1 | |
| 3 | Telugu `te` | You have a stomach infection. Take ORS solution after every loose motion. No spicy food for 3 days. | Stomach infection, ORS after each motion, no spicy food 3 days | | /1 | |
| 4 | Telugu `te` | Your knee X-ray shows mild arthritis. Take ibuprofen 400 mg after meals twice a day. | Mild arthritis, ibuprofen 400mg twice daily after food | | /1 | |
| 5 | Kannada `kn` | You are dehydrated. I am giving you IV fluids. You need to stay for observation for 2 hours. | Dehydration, IV fluids, 2h observation | | /1 | |
| 6 | Kannada `kn` | This looks like a urinary tract infection. Take ciprofloxacin 500 mg twice a day for 5 days. | UTI, ciprofloxacin 500mg twice daily, 5 days | | /1 | |
| 7 | English `en` | *(skip — patient is English-speaking, no back-translation needed)* | — | — | — | |
| 8 | English `en` | *(skip)* | — | — | — | |
| 9 | Tamil `ta` | You have a throat infection. Take amoxicillin 250 mg three times a day for 7 days. Rest your voice. | Throat infection, amoxicillin 250mg 3x/day 7 days, rest voice | | /1 | |
| 10 | Tamil `ta` | Your child has appendicitis. We need to operate tonight. Please sign the consent form. | Appendicitis, surgery tonight, consent form | | /1 | |

**Part B score: ___ / 8** (rows 7–8 skipped for English)

---

## Summary

| Metric | Value | Pass? |
|--------|-------|-------|
| Part A — Patient→Doctor | ___ / 10 | ≥ 8 ✓ / ✗ |
| Part B — Doctor→Patient | ___ / 8 | ≥ 6 ✓ / ✗ |
| Avg Gemma latency (first token) | ___s | < 5s target |
| Any hallucinations detected? | Y / N | N ✓ |

---

## Failure analysis

_For any score of 0, describe what went wrong._

| # | Direction | Problem | Likely cause |
|---|-----------|---------|-------------|
| | | | |

---

## Escalation criteria

**Translation quality low on Kannada/Telugu:**
- Upgrade prompt: add a concrete example pair (few-shot) in `translation/prompts.py`
- Or increase `GEMMA_NUM_CTX` from 4096 → 8192 in `config/settings.py`

**Hallucination (model adds symptoms/diagnoses not in input):**
- Strengthen the "Do not diagnose" rule in `PATIENT_TO_DOCTOR_PROMPT`
- Lower `GEMMA_TEMPERATURE` from 0.2 → 0.0

**Latency > 5s first token:**
- Switch to `gemma4:e2b` if using `4b` accidentally (check with `pdb doctor` in Phase 3)
- Reduce `GEMMA_NUM_CTX` to 2048 for faster KV-cache allocation

---

## Environment

```
Python version:
ollama version:
gemma4:e2b pulled: Y/N
gemma4:4b pulled: Y/N
CPU / RAM:
```
