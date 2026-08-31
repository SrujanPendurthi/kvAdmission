# Phase 9 — Surprisal Admission, and Screening Signals Without a GPU

**Date:** 2026-08-30
**Runs:** local, MPS. `scripts/run_surprisal.py --diagnose` on both task families.
**Hardware cost: $0.** No GPU was rented; the result is a prospective one.
**Verdict: negative, and it sharpens the central claim.** Information content is not query relevance.

---

## The candidate

Every scoring rule that has failed here is a function of position. Recency keeps the newest, accumulated attention keeps the oldest, observation-window voting keeps whatever the last few queries attended to. None asks whether a token carries information.

Surprisal does. A token the model already predicts from its prefix is reconstructible — the surrounding context regenerates it — so storing it buys nothing. A token it did not predict carries information the rest of the cache does not hold. Score each resident by `−log p(token | prefix)`, keep the top `B`.

Three things made it the right candidate to try first:

- **Genuinely write-time.** The logits already exist during prefill, and surprisal at position `t` depends only on tokens before `t`. No lookahead, no query, no second pass. It is computed from the *compressed* forward pass, so it is a policy and not an oracle.
- **Half the specification for free.** §7.2 requires the flagged set to fit the budget. Top-`B` makes `|flagged| = B` by construction, so the crowding law never binds. Only coverage is at risk.
- **Not positional.** The first candidate tested here that is not a function of distance.

## Screening it without running it

§7.2's partial-oracle ablation established that **coverage is unforgiving**: at 0.75 coverage over answer tokens, middle-band recall collapsed from 7/7 to 1/7. That makes coverage a *screening test*. If a candidate signal does not put essentially every answer token inside the top `B`, the end-to-end policy is predicted to fail and does not need to be run.

The diagnostic measures exactly that, generating nothing.

## Result: it fails on both task families

| task | coverage @ B=2048 | chance | spans fully retained | mean percentile of answer tokens |
|---|---|---|---|---|
| synthetic needle | **0.107** | 0.250 | 0/16 | **0.410** |
| real QA (answer phrase) | **0.248** | 0.250 | 2/16 | **0.484** |

(0.5 is a typical token; surviving a top-25% cut needs >0.75.)

**On needles, surprisal selects *against* the answer.** 0.107 is well below the 0.250 chance rate.

**On real QA it is exactly indifferent.** 0.248 against 0.250, and a percentile of 0.484 against 0.500. Answer tokens are statistically indistinguishable from ordinary prose.

A first pass measured coverage over the whole gold *paragraph*, which is the wrong denominator — a SQuAD span is ~160 tokens and the answer is a short phrase inside it. Locating the actual answer phrase (16/16 found) moved the number from 0.240 to 0.248. The conclusion did not change, but the earlier number should not have been trusted.

## Why needles are worse than chance

The token-level trace explains it exactly. From one needle:

```
' secret'   5.06
' code'     0.03      template - free
' for'      0.02
' jun'      9.02      the TOPIC is the surprising part
'iper'      0.01
' is'       0.01
'4'         3.00      the actual answer
'5'         2.08
'1'         1.81
'9'         2.48
'5'         2.41
'8'         2.81
```

Corpus median 2.79, p75 5.57, p90 8.57. **The digits score 1.8–3.0 — at or below the median.**

A uniform random digit is `−log(1/10)` = **2.30 nats**. That is the information-theoretic *ceiling* for a digit, not a floor. An unexpected English word runs 6–9 nats, because it is drawn from a 150k-token vocabulary rather than a 10-symbol alphabet.

So the prediction that motivated this experiment — that random digits would be maximally surprising — was wrong, and wrong in a specific way worth recording: **random over a small alphabet is barely surprising at all.** Worse, note what surprisal *would* keep: `' jun'` at 9.02, the topic word, while discarding the digits. Exactly backwards for answering "what is the code for juniper?"

## What this establishes

**Information content is not query relevance.** This is the sharpest statement of §7's observability thesis so far, because it is a failure of a *principled* signal rather than an ad-hoc one. Surprisal is the correct answer to "which tokens can the context not reconstruct?" That question turns out not to be the same as "which tokens will someone ask about," and the gap between them is the whole difficulty.

Two independent reasons the two come apart, one per task:

- **Small-alphabet answers** are low-entropy by construction, so any information criterion ranks them below ordinary prose. This affects the entire needle-in-haystack genre, not just this harness.
- **Natural-language answers** are simply typical. A name or a date is no harder to predict than the sentence around it.

## Method note: the specification is a screening tool

This result cost nothing because §7.2's coverage law let us reject the candidate before implementing the end-to-end run. Any proposed write-time signal can be screened the same way: measure what fraction of answer spans fall inside its top `B`, and compare against the partial-oracle curve. Signals that do not clear ~1.00 fully-retained are predicted to fail.

That is worth more than this particular negative result. It turns §7.2 from a description of the oracle's degradation into a cheap test any future candidate must pass first.

## Caveats

- **The end-to-end policy was not run.** The prediction rests on the partial-oracle coverage curve, which was measured on the same harness but on the synthetic task only. Confirming it end-to-end would cost about $0.10 and would also validate the screening method itself.
- One model (Qwen3-1.7B) for the diagnostic, two seeds per task.
- `recency_weight` is implemented but untested; a surprisal/recency blend might beat either alone without addressing the coverage problem that kills it here.
- Surprisal is measured under the compressed cache during a real run but under an uncompressed one in the diagnostic. They differ once compression bites; the diagnostic is therefore the *optimistic* case, which strengthens a negative result.
