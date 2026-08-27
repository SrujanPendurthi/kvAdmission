# Phase 1 Findings — Mid-Context Dead Zone Replication

**Date:** 2026-08-27
**Run:** `results/phase1.json` — Qwen3-1.7B, 8192-token streams, 8 needles, 5 seeds, 85 cells, 680 trials
**Plan:** `docs/superpowers/plans/2026-08-27-kv-admission-harness.md` Task 7
**Verdict:** The dead zone replicated. Phase 2 proceeds as specified.

---

## Headline

**0 hits in 192 middle-band trials**, across every compressed policy at every budget. By the rule of three the 95% upper bound on the true middle-band recall rate is **1.6%**.

| region | hits / trials | rate |
|---|---|---|
| near | 31 / 224 | 0.138 |
| **middle** | **0 / 192** | **0.000** (≤0.016) |
| far | 9 / 224 | 0.040 |

The full cache scores 1.00 in all three bands, so every zero is attributable to compression rather than to task difficulty.

## Recall by band

Mean, with 95% upper bound in parentheses. Zero-hit cells use the rule of three; others the normal approximation.

| policy | B | near | middle | far |
|---|---|---|---|---|
| full | 8192 | 1.00 | 1.00 | 1.00 |
| recent | 256 | 0.00 (≤0.21) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| recent | 512 | 0.14 (≤0.33) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| recent | 1024 | 0.36 (≤0.61) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| recent | 2048 | 0.71 (≤0.95) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| h2o | 1024 | 0.00 (≤0.21) | 0.00 (≤0.25) | 0.07 (≤0.21) |
| h2o | 2048 | 0.00 (≤0.21) | 0.00 (≤0.25) | 0.57 (≤0.83) |
| snapkv | 1024 | 0.07 (≤0.21) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| snapkv | 2048 | 0.29 (≤0.52) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| random | 1024 | 0.14 (≤0.33) | 0.00 (≤0.25) | 0.00 (≤0.21) |
| random | 2048 | 0.50 (≤0.76) | 0.00 (≤0.25) | 0.00 (≤0.21) |

Cells at B=256 and B=512 are 0.00 in every band for every policy and are omitted above; they are present in `results/phase1.json`.

---

## The five questions from the plan

### 1. Does `full` recall stay at or near 1.00 across all bands?

**Yes — exactly 1.00 in all three bands, on all five seeds.** 8/8 needles recalled every time. The task is solvable at 8192 tokens, so nothing below is confounded by the model simply failing at this length.

### 2. Each baseline's middle-band recall

**All four are 0.00, at every budget** (n=12 per cell, 95% upper bound 0.25 per cell; 0/192 pooled, upper bound 0.016).

### 3. Is any baseline's middle-band recall distinguishable from zero?

**No.** Not one hit in 192 trials. The per-cell bounds are weak individually — 12 trials cannot exclude a 25% rate — but the pooled result is strong, and every cell points the same way.

This is the question the plan originally phrased as "distinguishable from zero at 2 stderr", which is ill-posed: an all-zero cell has zero sample variance, so 2×stderr = 0 for every one of them. `summarize` now reports a rule-of-three bound instead.

### 4. Does near/far complementarity hold?

**Yes, and more sharply than in the spike.** At B=2048:

| policy | near | middle | far |
|---|---|---|---|
| recent | **10/14** | 0/12 | 0/14 |
| random | 7/14 | 0/12 | 0/14 |
| snapkv | 4/14 | 0/12 | 0/14 |
| h2o | 0/14 | 0/12 | **8/14** |

`recent` scores 10/14 in the near band and **exactly zero** in the far. `h2o` scores 8/14 in the far band and **exactly zero** in the near. They are perfect complements, and neither reaches the middle.

`snapkv` behaves like a weaker `recent`, consistent with its observation window scoring against the final query positions.

### 5. Does geometric survival still track `random`'s cutoff?

**Directionally yes; the exact functional form needs refining.** For `random` at B=2048, per-token survival is `(2048/2176)^(d/128)`:

| band | d range | predicted (per-token) | predicted (4-token span) | observed |
|---|---|---|---|---|
| near | 303–2615 | 0.527 | 0.153 | **0.50** |
| middle | 2778–5290 | 0.157 | 0.001 | **0.00** |
| far | 5719–7873 | 0.043 | 0.000 | **0.00** |

The single-token model nails the near band (0.527 predicted, 0.50 observed) but over-predicts the middle (0.157 vs 0.00). Treating the needle as a 4-token span whose tokens must all survive independently flips the error: it under-predicts near and matches middle/far.

The truth sits between, and the discrepancy is informative rather than a failure. Two effects the simple model omits:

- **How many tokens the code actually occupies.** A 6-digit number may tokenize to fewer than four pieces, and partial survival plus surrounding context ("The secret code for X is") may be enough for recall.
- **Correlated vs independent dropping.** `recent`'s FIFO removes contiguous blocks, so a surviving span survives whole. `random` drops independently, fragmenting spans — which is why `random` (0.50 near) now sits clearly below `recent` (0.71 near) where the spike had them tied.

The qualitative claim — geometric decay with distance, and a cutoff set by the compression schedule rather than the budget — holds. Refining to a span-survival model with a measured token count is a small, worthwhile follow-up.

---

## Changes from the spike

The spike's numbers should not be compared directly to these. Three things changed:

1. **Corrected `random`.** The spike drew an independent admission mask per layer, giving each token ~28 survival draws per chunk and inflating its recall until it tied `recent`. Fixed to one draw per chunk shared across layers (see spec §4.2 on the layer-alignment constraint). `random` now sits clearly below `recent`, as it should.
2. **Jittered needle placement.** Seeds now vary where needles land, so error bars describe placement sensitivity rather than only generation noise.
3. **Proper bounds.** Zero-hit cells report a rule-of-three upper bound instead of a misleading `±0.00`.

## What this still does not show

The caveats from spec §8 are undischarged and phase 2 does not discharge them either:

- **One model.** Qwen3-1.7B only. The dead zone may be a small-model artifact.
- **One task.** Synthetic verbatim needle retrieval — the easiest possible case for a cache and the least like real long-context work. Recall is exact substring match, so a model that holds the fact but paraphrases counts as a miss.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads, not the reference implementations. Their absolute numbers should not be compared to published ones.
- **Terciles are coarse.** "Middle" spans d≈2731–5461. A finer sweep would locate the boundary rather than assume a third.

The strong claim this licenses is narrow: *on this model and this task, no tested heuristic recovers mid-context information at any budget below full.* That is enough to justify phase 2, and not enough for a paper on its own.

## Decision

**Phase 2 proceeds.** The target is concrete: any nonzero middle-band recall at matched budget beats every baseline measured here. The go/no-go against a WG-KV-style differentiable gate at phase 3 is unchanged.

Before phase 2 starts, spec §4.2's layer-alignment resolution should be re-read — the learned policy emits one decision per token shared across layers, not the per-layer factored Bernoulli originally specified.
