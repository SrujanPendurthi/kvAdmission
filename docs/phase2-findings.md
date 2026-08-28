# Phase 2 Findings — The Learned Policy Reproduced Recency

**Date:** 2026-08-28
**Run:** `results/policy/` (300 steps), `results/phase2-eval.json` (5 seeds × 8 needles)
**Hardware:** RTX 3090 via vast.ai, ~4.5 h wall, $2.04
**Plan:** `docs/superpowers/plans/2026-08-27-kv-admission-phase2-policy.md` Tasks 6–7
**Verdict: negative result.** The reward was mis-specified. The fix is identified and cheap.

---

## Result

| policy | near | middle | far |
|---|---|---|---|
| **learned** | **0.71** (≤0.95) | **0.00** (≤0.25) | **0.00** (≤0.21) |
| `recent` (phase 1) | 0.71 | 0.00 | 0.00 |
| `h2o` (phase 1) | 0.00 | 0.00 | 0.57 |
| `full` (phase 1) | 1.00 | 1.00 | 1.00 |

The learned policy is **indistinguishable from pure recency in all three bands**. Middle-band recall remains zero. 40 eval rows, 5 seeds, phase 1's harness unchanged — same haystack, same seeds, same banding.

## Training did what it was asked

| steps | mean KL | occupancy | λ | r̄ | value loss |
|---|---|---|---|---|---|
| 0–50 | 0.2302 | 1634 | 0.038 | −0.3060 | 0.081 |
| 50–150 | 0.1279 | 1760 | 0.046 | −0.2731 | 0.992 |
| 150–250 | 0.1873 | 1711 | 0.043 | −0.2777 | 0.430 |
| 250–300 | 0.1182 | 1766 | 0.046 | −0.2687 | 0.937 |

KL fell from 0.23 to 0.12 and average reward improved from −0.306 to −0.269. **Training worked.** The optimizer found a better policy under the reward it was given.

The tell is in the second column: **occupancy rose from 1634 to 1766 as KL fell.** The policy learned to admit more, and under FIFO eviction "admit more" *is* recency. It never had to learn what to retain.

## Why: the reward averages away the signal it needs

This is a correction to a claim made in `docs/phase2-signal-check.md`, and it is the whole finding.

The signal check measured the **KL ratio at** long-dependency positions and found it strong — up to 12.5× baseline at B=2048. On that basis it concluded that reweighting the reward toward high-KL positions was unnecessary. That conclusion was wrong, because the ratio at a position and the position's share of the reward are different quantities.

The training reward is the **mean KL over all positions in a chunk**. Long-dependency positions are rare:

| stream | n_long | long-range KL mass | baseline KL mass | long-range share |
|---|---|---|---|---|
| seed 0 | 68 | 45.8 | 328.2 | 12.3% |
| seed 1 | 174 | 349.9 | 1096.9 | 24.2% |
| seed 2 | 121 | 49.5 | 545.6 | 8.3% |
| | | | | **mean 14.9%** |

At ~121 long-dependency positions in 8192 tokens — **1.5% of positions** — a 12.5× KL contrast still yields only ~15% of the total reward mass. **The remaining 85% is local next-token prediction, which recency already maximizes.**

So the policy converged to the reward's actual optimum. Recency is not a failure of optimization here; it is the correct answer to the question the reward asked.

## The fix

Stop averaging. Weight each position's contribution by its own divergence, so the 15% is not drowned by the 85%:

```
r_k  =  − Σ_t w_t · KL_t / Σ_t w_t     with  w_t = KL_t^β  (β ≈ 1)
```

Equivalently, reward the *worst* positions rather than the mean — a CVaR-style objective over the KL distribution within a chunk. Positions where compression already changed nothing contribute no gradient regardless; averaging over them only dilutes.

This costs nothing extra to compute. `dual_prefill` already returns per-position KL; `train.rollout` throws that detail away by calling `.mean()` on it.

**Prediction to test:** under the reweighted reward, occupancy should stop being the lever the policy pulls. If KL falls while occupancy stays flat or drops, the policy is selecting rather than hoarding.

## Two suspects the plan flagged, now largely cleared

The plan's self-review named chunk-level credit assignment and greedy-vs-sampled evaluation as the first things to suspect. Neither is the primary cause:

- **Chunk-level credit** is coarse, but the policy did learn — reward improved monotonically. A credit-assignment failure would look like a flat or noisy reward curve, not a clean descent to the wrong optimum.
- **Greedy eval** produced exactly the near-band number `recent` achieves (0.71), which is what a recency-equivalent policy should score. If sampling-vs-greedy were the issue the eval would be *worse* than training suggested, not precisely consistent with it.

Both remain worth ablating, but neither explains the result. The reward specification does.

## Cost accounting

$2.04 against a $0.30 estimate. The overrun was setup, not compute:

- Training itself: ~50 minutes on a 3090 (versus ~12 hours projected on the M4)
- The rest went to four failures on a stock PyTorch CUDA image — transformers 5 requiring torch ≥2.5, torchvision ABI breakage after the upgrade, a conda `readline` that segfaults pytest, and triton needing a C compiler the image lacks
- Plus idle time while tasks 6–7 were written *after* the box was already running, which was avoidable sequencing

All four image failures are now fixed and documented inline in `scripts/provision.sh`. A rerun should be close to the original estimate.

## What this does not show

- **One reward variant, one seed of training.** No hyperparameter search. It is possible a different learning rate or λ target changes the picture, though the reward-mass arithmetic suggests otherwise.
- **The 85/15 split is corpus-dependent.** Denser long-range dependencies would shift it. Whether prose or agent traces behave differently is untested.
- All phase 1 caveats stand: one model, one synthetic verbatim-recall task, simplified baseline reimplementations.

## Decision

**Do not proceed to phase 3 (the WG-KV head-to-head).** Comparing a recency-equivalent policy against a differentiable gate would measure nothing.

> **FOLLOW-UP 2026-08-28 — see `docs/phase2b-findings.md`.** Phase 2b applied the
> reweighting proposed below. It reached this run's KL and occupancy by step 150
> (0.119 / 1775 versus 0.121 / 1773 here), then collapsed to admitting nothing
> and scored 0.00 in every band. The reward diagnosis below still looks correct;
> the proposed fix was insufficient on its own because plain policy gradient
> destabilised under it, the one-sided occupancy penalty made an empty cache
> free, and only the last checkpoint was retained.

Next step is a phase 2b: reweight the reward as above and retrain. It is a small change to one line of `rollout`, and the existing harness, checkpointing, and evaluation all carry over unchanged. If mid-band recall is still zero under a reward that actually emphasises long-range positions, *that* is the interesting negative result — and it would point at the policy or the credit assignment rather than the objective.
