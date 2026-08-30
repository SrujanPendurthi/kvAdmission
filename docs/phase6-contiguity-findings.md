# Phase 6 — The Contiguity Control: Hypothesis Half-Refuted

**Date:** 2026-08-30
**Runs:** `results/contiguity-qwen.json`, `results/contiguity-smollm2.json` (240 rows)
**Hardware:** RTX 3090 via vast.ai, ~18 min, **$0.044**
**Verdict:** the *fragment-length* form of the contiguity hypothesis is refuted. The *positional-gap* form is untested and now has more support than before, from a band breakdown rather than from the sweep.

---

## The hypothesis under test

`docs/phase3-batch-findings.md` §3 found the oracle at 40/40 on Qwen3-1.7B and 12/24 on SmolLM2-1.7B. The proposed explanation was **contiguity**: the oracle's retained set is a handful of short answer-span islands spliced between recency fill, with gaps in the position indices, and the two models may differ in whether they can read that.

That is really two hypotheses:

- **(a) fragment length** — a bare answer span carries no surrounding context, and a model may need the sentence around a fact to use it
- **(b) fragment count / positional gaps** — the retained set is discontinuous, and the model cannot bridge jumps in position index

This run tests (a). It **cannot** test (b), for a reason found before renting: with 8 needles ~1024 tokens apart, dilating spans does not merge them until the flagged set is several times the budget (r=384 gives 6 runs but 5958 tokens against B=2048). Run count is pinned at 8 across the entire usable range.

## Method

Each answer span is dilated by radius `r` and overlapping runs merge. Same needles, same budget (2048), same evaluation. `flagged` is reported alongside, because past B the crowding law takes over and a fall in recall there is not a fragmentation effect.

## Result

**Qwen3-1.7B** — the control, already at ceiling:

| r | flagged | overall | near | middle | far |
|---|---|---|---|---|---|
| 0 | 125 | **24/24** | 9/9 | 7/7 | 8/8 |
| 16 | 381 | 24/24 | 9/9 | 7/7 | 8/8 |
| 48 | 893 | 24/24 | 9/9 | 7/7 | 8/8 |
| 96 | 1661 | 23/24 | 9/9 | 6/7 | 8/8 |
| 192 | 3197 *(over)* | 15/24 | 9/9 | 6/7 | **0/8** |

**SmolLM2-1.7B-Instruct** — the model with the shortfall:

| r | flagged | overall | near | middle | far |
|---|---|---|---|---|---|
| 0 | 125 | 12/24 | **9/9** | 2/7 | 1/8 |
| 16 | 381 | 12/24 | **9/9** | 2/7 | 1/8 |
| 48 | 893 | 14/24 | **9/9** | 3/7 | 2/8 |
| 96 | 1661 | 14/24 | **9/9** | 3/7 | 2/8 |
| 192 | 3197 *(over)* | 14/24 | 8/9 | 6/7 | **0/8** |

## 1. Fragment length is not the cause

Under budget, SmolLM2 goes from 12/24 at r=0 to 14/24 at r=96 — **+2/24, roughly 0.8 standard errors at n=24.** Middle band 2/7 to 3/7. Widening the retained window from a bare 16-token answer span to a ~200-token coherent block of prose does essentially nothing.

If a missing sentence-context were the reason the oracle fails on SmolLM2, this curve should climb steeply. It is flat. **Hypothesis (a) is refuted.**

## 2. The band breakdown is the real finding, and it points at (b)

**SmolLM2's near band is 9/9 at every under-budget radius**, including r=0 where each needle is a bare 16-token island.

This is decisive about one thing: **SmolLM2 reads spliced answer spans perfectly well.** The retained set is exactly as fragmented in the near band as in the middle band — same oracle, same splicing, same short islands. Only the *distant* ones fail. So the failure is not "this model cannot use a fragmented cache."

What distinguishes near from middle here is that near-band needles fall inside the recency block, so they sit in a contiguous run that runs unbroken up to the query. Middle- and far-band needles are islands separated from the query by gaps in the position index. **That is exactly the pattern hypothesis (b) predicts**, and it now has support it did not have before — from the band structure, not from the sweep, which was designed to test (a).

Note also that SmolLM2 with a **full uncompressed cache** scores 7/7 in the middle band. The same needle, at the same distance, is answerable when the intervening tokens are present and unanswerable when they are not. Distance alone is not the barrier; the gap is.

## 3. Crowding law re-validated

At r=192 the flagged set is 3197 against B=2048, so the predicted retention is 2048/3197 = **0.64**:

| model | r=0 recall | × 0.64 | observed r=192 |
|---|---|---|---|
| Qwen3 | 1.00 | **0.64** | **0.62** |
| SmolLM2 | 0.50 | 0.32 | 0.58 |

Qwen matches to 0.02. SmolLM2 does not, and the reason is visible in its table: at r=192 its middle band *rises* to 6/7 while the far band collapses to 0/8. Once the flagged set overflows, the oracle keeps the most recent flagged tokens, which discards far-band windows entirely and spends the budget on complete nearer ones. On a model already at ceiling that is pure loss; on one that was failing the middle band it is a favourable reallocation. The law describes token retention, not recall on a model whose per-band recall is non-uniform.

## 4. What to run next

The decisive experiment for (b) holds *fragment count* as the variable at matched flagged tokens — for example 8 needles with 16-token windows versus 2 needles with 64-token windows, same total, same distances. That changes the number of questions per stream, so it needs more seeds, but it isolates the gap count directly.

A cheaper partial probe: an oracle that retains everything from the first needle to the query (one contiguous run, no gaps, budget permitting) versus the standard scattered oracle at matched token count.

## Scope and honesty notes

- This refutes one form of the hypothesis, not the hypothesis. The writeup previously stated contiguity as the leading explanation without qualification; it now has to distinguish (a) from (b) and report (a) as tested and false.
- n=24 per cell, 3 seeds, 8 needles. A +2/24 shift is not distinguishable from zero, and neither would a −2/24 shift be.
- Qwen is at ceiling for every under-budget radius, so it cannot show an effect in either direction. It functions as a harness check and a crowding-law check, not as a contrast.
