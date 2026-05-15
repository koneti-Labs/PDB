# Phase 3 Manual Evaluation — Emergency Triage Extraction

## Setup

```bash
pdb triage
```

Gemma 4 model: `gemma4:4b` (REASONING_EXTRACTION, think=True)

## Scoring rubric (per utterance, max 5 points)

| Field | Points | Criteria |
|-------|--------|----------|
| chief_complaint | 1 | Correct body location + nature in English |
| severity | 1 | Matches cues in the speech (mild/moderate/severe/critical) |
| duration | 0.5 | Extracted if mentioned; "not mentioned" if absent |
| symptoms list | 1 | All distinct symptoms captured; no fabrications |
| needs_immediate_attention | 0.5 | Correct boolean for the scenario |
| (bonus) vitals_mentioned | — | Not scored but spot-checked for accuracy |

Acceptance bar: **>=90 points / 100** (i.e. >=18/20 utterances fully correct).

---

## Test utterances

### Hindi (hi) — 5 utterances

| # | Input (Hindi) | Expected severity | Expected needs_immediate |
|---|---------------|-------------------|--------------------------|
| H1 | "Mujhe seene mein bahut tez dard ho raha hai, saath mein saans lene mein bhi takleef hai. Subah se ho raha hai." | severe/critical | true |
| H2 | "Mera sar bahut dard kar raha hai, do din se. Bukhaar bhi hai, 102." | moderate | false |
| H3 | "Pet mein halka dard hai, khaana khane ke baad. Kal se hai." | mild | false |
| H4 | "Mujhe chakkar aa rahe hain aur aankhon ke aage andhera chhaa raha hai, gir gayi ek baar." | severe | true |
| H5 | "Gale mein kharaash hai, thodi si, aaj se." | mild | false |

### Telugu (te) — 4 utterances

| # | Input (Telugu) | Expected severity | Expected needs_immediate |
|---|----------------|-------------------|--------------------------|
| T1 | "Naa chetiki chaala bayam gaa undi, naalugu rojulatho nunchi. Vurulata undi." | moderate | false |
| T2 | "Chaala teevranga okaasariga gundella noppulu vastunnaayi, chaeyyi varaku vastundi." | critical | true |
| T3 | "Naa kalla mundhu anni tirugutunnayi, padipovadam jarigindi." | severe | true |
| T4 | "Kaapu noppulu, nadustu unna time lo. Rendu rojulugaa." | mild/moderate | false |

### Kannada (kn) — 4 utterances

| # | Input (Kannada) | Expected severity | Expected needs_immediate |
|---|-----------------|-------------------|--------------------------|
| K1 | "Nange thumba joraagi talenoovu aagthide, eradu dina aythu." | moderate | false |
| K2 | "Ushna maadikonde, joraagi kashta aagthide ushiraadaaga." | mild | false |
| K3 | "Nange ushira badidaaga thale suththu bandbithu, bididde." | severe | true |
| K4 | "Hengu iruthide, adhu thumba joraagi ede aagthide, beligge nantara." | moderate | false |

### Tamil (ta) — 4 utterances

| # | Input (Tamil) | Expected severity | Expected needs_immediate |
|---|---------------|-------------------|--------------------------|
| A1 | "Enakku manasu kalaiyaaguthu, moochu vaangi varuthu. Kaalai le irundu." | severe | true |
| A2 | "Thalai vali iruku, oru naal achu. Kanjam kanjam." | mild | false |
| A3 | "Vayiru valikuthu, rendum moonum naal achu, kaayal marunthu saapturukken." | moderate | false |
| A4 | "Kaiyil thimiru iruku, pakkathu kaiyilayum. Rendu madam achu." | moderate | false |

### English (en) — 3 utterances

| # | Input (English) | Expected severity | Expected needs_immediate |
|---|-----------------|-------------------|--------------------------|
| E1 | "I have severe chest pain radiating to my left arm since this morning. I am sweating a lot." | severe/critical | true |
| E2 | "My ankle is a bit swollen after I twisted it yesterday. It hurts when I walk." | mild | false |
| E3 | "I can not speak properly and my face feels droopy on one side. Started an hour ago." | critical | true |

---

## Results table

| # | Lang | severity correct | needs_immediate correct | complaint correct | symptoms complete | Score /5 |
|---|------|:----------------:|:-----------------------:|:-----------------:|:-----------------:|:--------:|
| H1 | hi | | | | | |
| H2 | hi | | | | | |
| H3 | hi | | | | | |
| H4 | hi | | | | | |
| H5 | hi | | | | | |
| T1 | te | | | | | |
| T2 | te | | | | | |
| T3 | te | | | | | |
| T4 | te | | | | | |
| K1 | kn | | | | | |
| K2 | kn | | | | | |
| K3 | kn | | | | | |
| K4 | kn | | | | | |
| A1 | ta | | | | | |
| A2 | ta | | | | | |
| A3 | ta | | | | | |
| A4 | ta | | | | | |
| E1 | en | | | | | |
| E2 | en | | | | | |
| E3 | en | | | | | |
| **Total** | | | | | | **/100** |

---

## Escalation criteria

Escalate to prompt revision if:

- Severity is wrong on both critical/severe utterances for a language (H1, T2, A1, E1, E3)
- `needs_immediate_attention=false` for any utterance marked **true** above
- `chief_complaint` is vague ("pain" without body location) on >3 utterances
- Symptoms list contains fabricated symptoms not in the input

## Notes

_Fill in after running._
