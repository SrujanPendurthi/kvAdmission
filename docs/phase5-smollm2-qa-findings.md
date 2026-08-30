# Phase 5 — Real QA on SmolLM2: Inconclusive, and Why

**Date:** 2026-08-30
**Runs:** `results/smollm2-benchmark.json` (264 rows), `results/smollm2-closed-book.json` (24 rows)
**Hardware:** RTX 3090 via vast.ai, ~20 min, **$0.085**
**Verdict: inconclusive.** The benchmark does not discriminate on this model. Recording it as a null-information result rather than a replication or a refutation.

---

## The question

`docs/phase3-batch-findings.md` §3 found the oracle reaching 40/40 on Qwen3-1.7B synthetic needles but only 12/24 on SmolLM2 — the one result that did not replicate. `docs/phase4-benchmark-findings.md` then showed everything replicating on real QA, but on Qwen only. This run asks whether the SmolLM2 oracle shortfall also appears on real QA.

## The problem: there is almost no dynamic range

| | SmolLM2 | Qwen3 |
|---|---|---|
| closed-book floor (no haystack) | **3/24 (0.125)** | 0/24 (0.000) |
| uncompressed ceiling (`full`) | **11/24 (0.458)** | 19/24 (0.792) |
| **usable band** | **8 trials of 24** | 19 trials of 24 |

SmolLM2 answers only 11 of 24 planted-evidence SQuAD questions **with the entire context present and uncompressed**, and guesses 3 of them with no context at all. Every policy is being scored inside an 8-trial window, at n=24.

## Result

Entries are `hits/trials`; "vs ceiling" is the policy's total as a fraction of `full`.

| policy | B | near | middle | far | total | vs ceiling |
|---|---|---|---|---|---|---|
| `full` | 8192 | 6/9 | 4/8 | 1/7 | 11/24 | 100% |
| **oracle** | 2048 | 6/9 | 3/8 | 0/7 | **9/24** | **82%** |
| oracle | 512 | 6/9 | 2/8 | 0/7 | 8/24 | 73% |
| `snapkv` | 2048 | 6/9 | 1/8 | 0/7 | 7/24 | 64% |
| `random` | 2048 | 7/9 | 0/8 | 0/7 | 7/24 | 64% |
| `recent` | 2048 | 6/9 | 0/8 | 0/7 | 6/24 | 55% |
| `h2o` | 2048 | 2/9 | 1/8 | 0/7 | 3/24 | 27% |

Heuristic middle band: **4/64**, 95% upper **13.7%**, against a ceiling of 4/8.

## What can and cannot be concluded

**Cannot:** whether the oracle helps on SmolLM2. It scores 9/24 against a best heuristic of 7/24 — a two-trial gap at n=24, which is noise. On Qwen the same comparison was 18/24 against 7/24. The absence of separation here is fully explained by the ceiling being 11/24 instead of 19/24, so this run does not distinguish "the oracle does not help SmolLM2" from "SmolLM2 cannot do this task."

**Cannot:** the far band. `full` scores 1/7, so the band is uninformative on this model by construction.

**Weakly can:** the dead zone points the same way. Middle band 4/64 for heuristics against a 4/8 ceiling, and both pure-recency policies (`recent`, `random`) are at 0/8. But 95% upper of 13.7% is much weaker than the 7.2% on Qwen and the 1.1% on synthetic needles.

**One thing worth noting:** relative to its own ceiling, the SmolLM2 oracle recovers 82% here, against 52% (12/24 of 23/24) on synthetic needles. If anything the real-QA shortfall is *milder* than the needle shortfall, which is the opposite of what the contiguity hypothesis would predict if contiguity were the whole story. This is one data point on a noisy measurement and should not be leaned on.

## The process error

I ran a local uncompressed-ceiling check on Qwen before renting anything for phase 4, precisely because a floor effect would make the benchmark worthless — and it caught a broken metric. **I did not run that check for SmolLM2 before renting.** Doing so takes a few minutes on the laptop and would have shown an 11/24 ceiling against a 3/24 floor, at which point the right call is to fix the task or drop the model, not to buy a GPU.

The cost was only $0.085, so the money is not the point. The point is that the check exists, it had already paid for itself once, and I skipped it on the second run of the same experiment.

## What would actually answer the question

1. **A stronger second model.** Qwen2.5-3B or Llama-3.1-8B has the headroom this measurement needs. The dead-zone claim is about compression policies, and it cannot be tested on a model that fails the task uncompressed.
2. **Or an easier real task** — more redundant answers, shorter contexts — trading realism for dynamic range.
3. **The contiguity control remains the direct test** of §7.1's hypothesis, and unlike this run it does not depend on the second model being good at QA: it compares two oracles against each other on whichever model shows the shortfall.

Recommendation: do (3) before repeating (1). It is cheaper and it tests the actual hypothesis.
