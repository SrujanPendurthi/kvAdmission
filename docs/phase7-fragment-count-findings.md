# Phase 7 — Fragment Count Is the Variable

**Date:** 2026-08-30
**Runs:** `results/fragcount-qwen.json`, `results/fragcount-smollm2.json` (96 rows each)
**Hardware:** RTX 3090 via vast.ai, ~15 min × 2 (one run lost, see Process), **$0.20**
**Verdict: positive.** The number of gaps in the position index is what breaks SmolLM2's oracle. Fragment count matters; fragment length does not.

---

## Design

The stream, its 8 needles and their distances are identical in every condition. The oracle flags `k` of them, with each window widened so the **total flagged tokens stays at exactly 1600**. Only flagged needles are queried, and seeds scale as 24/k so every condition gets 24 trials. Recency fill is therefore also identical across conditions (2048 − 1600 = 448 tokens).

The only thing that varies is how many contiguous runs the retained evidence occupies — that is, how many jumps in position index sit between the query and the facts it needs.

## Result

**SmolLM2-1.7B-Instruct** — the model with the shortfall:

| k | runs | flagged | window | overall | 95% CI | near | middle | far |
|---|---|---|---|---|---|---|---|---|
| **8** | 8 | 1600 | 200 | **13/24 = 0.54** | [0.35, 0.72] | 9/9 | 3/7 | 1/8 |
| **4** | 4 | 1600 | 400 | **22/24 = 0.92** | [0.74, 0.98] | 11/11 | 7/8 | 4/5 |
| 2 | 2 | 1600 | 800 | 20/24 = 0.83 | [0.64, 0.93] | 8/8 | 6/7 | 6/9 |
| 1 | 1 | 1600 | 1600 | 19/24 = 0.79 | [0.60, 0.91] | 13/13 | 2/3 | 4/8 |

**Qwen3-1.7B** — the control, no headroom:

| k | runs | overall | near | middle | far |
|---|---|---|---|---|---|
| 8 | 8 | 23/24 = 0.96 | 9/9 | 6/7 | 8/8 |
| 4 | 4 | 24/24 = 1.00 | 11/11 | 8/8 | 5/5 |
| 2 | 2 | 24/24 = 1.00 | 8/8 | 7/7 | 9/9 |
| 1 | 1 | 24/24 = 1.00 | 13/13 | 3/3 | 8/8 |

**SmolLM2 k=8 (13/24) versus k=4 (22/24): Fisher exact two-sided p = 0.0078.**

Recall jumps between 8 fragments and 4, then is flat within noise across k=4, 2, 1 — their confidence intervals all overlap heavily. The honest statement is a step between k=8 and k≤4, not a smooth monotone trend.

## The effect is entirely in the middle and far bands

SmolLM2's near band is **9/9, 11/11, 8/8, 13/13** — perfect at every k. Near-band needles sit inside the recency block, contiguous with the query, so no gap separates them from it and there is nothing for fragment count to affect.

What moves is exactly the dead zone:

| band | k=8 | k=4 |
|---|---|---|
| middle | 3/7 | **7/8** |
| far | 1/8 | **4/5** |

This is the mid-context dead zone being escaped, on the model where the oracle had previously failed to escape it.

## The confound, and why it does not explain the result

`k` and window width co-vary by construction: holding flagged tokens at 1600 means k=8 gives 200-token windows and k=4 gives 400-token ones. Within this experiment alone, "fewer gaps" and "wider windows" cannot be separated.

Two things separate them anyway:

1. **Phase 6 varied width at fixed count and found nothing.** At 8 runs, widening each window from 16 to 208 tokens moved SmolLM2 from 0.50 to 0.58 — about 0.8 standard errors. The k=8 point here (200-token windows, 0.54) reproduces that. If width were the driver, the phase 6 curve would have climbed.
2. **The trend is not monotone in width.** k=1 has the widest windows by far (1600 tokens) and scores *worst* of the three low-k conditions (0.79 against k=4's 0.92). A width effect would make k=1 best.

Together: width does nothing, count does something, and the something is a step down when the retained evidence is split into 8 pieces rather than 4.

## What this settles, and what it does not

**Settles.** The contiguity hypothesis from `docs/phase3-batch-findings.md` §3 is correct in its second form. The oracle's SmolLM2 shortfall is not that the evidence is absent, nor that answer spans lack local context — it is that the retained set is broken into too many pieces, and SmolLM2 cannot bridge that many jumps in position index while Qwen3 can.

**Does not settle.** Why 8 is too many and 4 is not. Nothing here identifies the mechanism — whether it is RoPE extrapolation across gaps, attention dispersal over many discontinuities, or something specific to SmolLM2's training. The threshold is also measured at one budget, one stream length and one needle count, so "4 is fine, 8 is not" should not be read as a general constant.

**Consequence for the wider result.** The dead zone is escapable on both models after all — but on SmolLM2 only if the retained set stays coarse. That is a real constraint on method design that none of the published heuristics respect: H2O, SnapKV and their relatives select tokens individually and produce maximally fragmented caches. On a model with SmolLM2's sensitivity, a policy that retains the right tokens in too many pieces buys much less than the same token budget spent on fewer, larger blocks.

## Process: a run destroyed with its results on it

The first execution of this experiment completed and was then **destroyed by my own watchdog before its results were pulled.** The watchdog's `scp` glob was hardcoded to `results/gpu-*` from the first batch weeks earlier; this run wrote `fragcount-*`, the glob matched nothing, `scp` reported success, and the box was torn down. Qwen's output and both raw JSONs were lost. Only the SmolLM2 aggregates survived, in terminal scrollback.

The rerun reproduced SmolLM2 **exactly** at every k (13/22/20/19 of 24), which is at least a free determinism check.

The watchdog now pulls the entire `results/` directory and, on any pull failure, leaves the instance **alive** and exits non-zero rather than destroying it. A few cents of idle billing is strictly better than losing a run.

Second note: the invariant test for this experiment failed on first write and it was a real bug, not a bad fixture. Symmetric dilation clamps at the stream edges, so at k=1 the flagged set reached 908 of a 1600 target — low-k conditions would have received less evidence and more recency fill, and the result would have looked like a fragment-count effect that was really an evidence effect. `expand_spans_to_quota` fixes it; flagged is now exactly 1600 at every k.
