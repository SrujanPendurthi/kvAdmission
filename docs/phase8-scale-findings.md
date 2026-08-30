# Phase 8 — Scale to 8B, and the Horizon Law's Last Prediction

**Date:** 2026-08-30
**Runs:** `results/scale8b-sweep.json` (216 rows), `results/scale8b-oracle.json` (24), `results/horizon-p05.json` (64)
**Hardware:** RTX 3090 via vast.ai, ~50 min, **$0.42**
**Verdict:** the dead zone and the oracle both replicate at 4.7× scale. The horizon law's horizon is confirmed; its plateau prediction was mine and it was wrong.

---

## 1. The dead zone is not a small-model artifact

Every prior result used ~1.7B models, which was the single largest threat to this work. Qwen3-8B is the same family as the primary model, so scale is the only variable.

| policy | B | near | middle | far |
|---|---|---|---|---|
| `full` | 8192 | 9/9 | **7/7** | 8/8 |
| **oracle** | 2048 | **9/9** | **7/7** | **8/8** |
| `recent` | 2048 | 6/9 | **0/7** | **0/8** |
| `h2o` | 2048 | **0/9** | **0/7** | **4/8** |
| `snapkv` | 2048 | 3/9 | **0/7** | 0/8 |
| `random` | 2048 | 4/9 | **0/7** | 0/8 |

**Middle band across all four heuristics and both budgets: 0/56** (95% upper 5.4%), against an uncompressed cache that is *perfect* in all three bands.

**The near/far complementarity holds for a third time.** `recent` scores 6/9 near and 0/8 far; `h2o` scores 0/9 near and 4/8 far. Each family owns one end of the context, scores exactly zero at the other, and neither reaches the middle. This is now observed on Qwen3-1.7B, SmolLM2-1.7B and Qwen3-8B.

## 2. The oracle also replicates at scale

**24/24 — perfect recall in every band at B=2048**, matching the uncompressed cache exactly on a quarter of the context. Identical to the 40/40 on Qwen3-1.7B.

This matters as much as the dead zone itself. The observability conclusion — that the budget is not the binding constraint and the failure is that nothing at write time identifies which tokens matter — was previously established on one 1.7B model and had *failed* to replicate on SmolLM2. It now holds at 8B. Combined with phase 7, the picture is coherent: the capacity exists on every model tested, and SmolLM2's shortfall was fragmentation sensitivity rather than a limit on what a compressed cache can carry.

## 3. The horizon law: half confirmed, half my error

§4 predicted that under admission at rate `p` with FIFO eviction the cache spans the most recent `B/p` positions, so halving the rate should double the horizon and halve the plateau. At `p`=0.5, B=2048, the horizon should move to 4096.

| | observed |
|---|---|
| overall | 2/64 = 0.031 |
| **d < 4096** (inside predicted horizon) | **2/32 = 0.062** |
| **d ≥ 4096** | **0/32 = 0.000** |

**The horizon is confirmed.** Every hit falls inside 4096 and there are none beyond it. At `p`=0.941 the horizon sat at 2048–2304; at `p`=0.5 it is inside 4096 and empty past it. Retention tracks `B/p`.

**The plateau prediction was wrong, and it was my error rather than the law's.** I wrote "halve the plateau", i.e. plateau = `p` = 0.50. Observed inside the horizon is 0.062, eight times lower.

The reason is that a needle is a **multi-token irredundant span**: a 6-digit code is one token per digit, and recall requires *every* one to survive. The plateau is therefore `p^L`, not `p`:

| rate | plateau = p | p⁴ | p⁵ | p⁶ | observed |
|---|---|---|---|---|---|
| 0.941 | 0.94 | 0.78 | 0.74 | 0.69 | **0.81** |
| 0.500 | 0.50 | 0.063 | 0.031 | 0.016 | **0.062** |

The span-survival model is directionally right and far better than `plateau = p`, but it is not a clean quantitative law: the implied `L` is about 2.7 at `p`=0.941 and 4.0 at `p`=0.5. Tokenisation varies across codes, and partial credit does not exist under exact match, so a single effective span length does not fit both rates.

**What §4 should say.** The residency law — `P(token resident at d) = p if d < B/p, else 0` — is about *token* residency and is confirmed. What does not follow from it is that *needle recall* equals `p` inside the horizon; recall is the survival of a whole span, which falls much faster. The earlier draft already hedged this for `p`=0.941 by quoting both 0.94 and 0.69; the "halve the plateau" line went further than the evidence and is now corrected.

## Cost note

$0.42 against a $0.15 estimate. The overrun is the 8B sweep: `h2o` and `snapkv` need `output_attentions=True`, which forces eager attention, and at 36 layers × 32 heads that is far slower than the 1.7B case. Memory was fine — 24 GB held the 16 GB of weights, a 1.2 GB cache and the attention tensors without trouble.

## What this changes

The paper's first limitation was "two models, one size class, may be a small-model artifact." That is now answered for the dead zone, the complementarity and the oracle. What remains untested at scale: the real-QA benchmark, the fragmentation threshold, and everything about SmolLM2.
