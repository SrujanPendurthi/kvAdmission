# Phase 2b Findings — Reweighted Reward, Training Collapse

**Date:** 2026-08-28
**Run:** `results/policy-b1/` (300 steps, β=1), `results/phase2b-eval.json` (5 seeds × 8 needles)
**Hardware:** RTX 3090 via vast.ai, ~2.2 h, $0.38
**Verdict: worse than phase 2.** Not a reward exploit — an optimization collapse, with three separable causes.

---

## Result

| | near | middle | far | final occupancy |
|---|---|---|---|---|
| `recent` (phase 1) | 0.71 | 0.00 | 0.00 | 2048 |
| phase 2 (β=0) | 0.71 | 0.00 | 0.00 | 1776 |
| **phase 2b (β=1)** | **0.00** | **0.00** | **0.00** | **4** |
| `full` (phase 1) | 1.00 | 1.00 | 1.00 | 8192 |

0/8 on every seed, all three bands. Direct measurement of the trained policy confirms why: **it admits nothing.** Occupancy holds at 4 — the pinned sink tokens — for an entire 8192-token stream. Admit fraction 0.0000.

## What was changed

Phase 2's diagnosis was that the reward's plain chunk mean drowns long-range signal: long-dependency positions are ~1.5% of a stream and carry ~15% of KL mass, so 85% of the gradient goes to local prediction, which recency maximises.

The fix tried here weights each position by its own divergence:

```
r_k = − Σ_t w_t·KL_t / Σ_t w_t ,   w_t = KL_t^β ,   β = 1
```

The reweighting did work mechanically. Observed weighted KL ran ~14× the unweighted mean, so the rare high-divergence positions genuinely dominated the reward.

## It was learning, then it collapsed

| steps | KL weighted | KL raw | occupancy | λ | r̄ |
|---|---|---|---|---|---|
| 0–25 | 3.681 | 0.277 | 1600 | 0.036 | −3.905 |
| 25–75 | 3.213 | 0.165 | 1718 | 0.043 | −3.747 |
| **75–150** | **2.576** | **0.119** | **1775** | 0.047 | **−3.397** |
| 150–250 | 3.891 | 0.400 | 1271 | 0.020 | −3.473 |
| 250–300 | 4.335 | 0.572 | 790 | 0.001 | −3.906 |

**At step ~150 this run was working.** `kl_raw` 0.119 and occupancy 1775 are statistically indistinguishable from phase 2's final 0.121 / 1773 — reached in half the steps, under a reward that actually emphasised long-range positions. Then it fell apart.

**r̄ degraded too**, −3.397 back to −3.906, ending where it started. That is the decisive detail. A policy exploiting a loophole in its objective shows reward *rising* while the true metric falls. Here reward and metric fell together. **This is optimization failure, not reward hacking**, and an earlier read of mine that called it a "genuine optimum of the objective" was wrong.

## Three separable causes

**1. Plain policy gradient was insufficient.** The plan specified "no PPO until simple PG demonstrably fails." It demonstrably failed. Under β=1 the reward is a ratio of moments dominated by rare large values, so gradient variance is high. With no trust region and no entropy bonus, the policy walked into a deterministic reject-everything state and could not walk back — once every logit is confidently negative, sampled actions are all rejections and the gradient only reinforces them.

**2. The Lagrangian penalises over-use only.** `λ = max(0, λ + η(occupancy − target))` means occupancy *below* target costs nothing; λ decays to zero (observed: 0.047 → 0.001) and an empty cache is free. This flaw was present in the spec from the start and survived phase 2 only because plain-mean KL punished dropping tokens hard enough to keep occupancy pinned high. Phase 2b removed that pressure and exposed it.

**3. Checkpointing overwrote the good policy.** `train()` saves to the same `policy.pt` every 20 steps. The step-150 checkpoint was as good as phase 2 and was overwritten by the collapsed one. A recoverable run became unrecoverable through pure engineering carelessness — no best-checkpoint retention, no early stopping on a validation metric.

## Fixes, in priority order

1. **Keep the best checkpoint**, scored on held-out unweighted KL, not the last one. Cheapest fix, largest regret avoided.
2. **Two-sided occupancy penalty.** Penalise |occupancy − target| rather than max(0, ·), so an empty cache is never free.
3. **Constrain the policy update** — PPO clipping or a KL trust region on the policy itself, plus an entropy bonus to prevent premature determinism.
4. **Reconsider β.** β=1 makes the objective a ratio of moments, which is scale-free in an unhelpful way. A fixed positional weighting derived offline from the signal check, or a top-k-worst objective with k fixed, emphasises long-range positions without being reducible by raising the floor.

Do 1–3 before revisiting 4. The step-150 evidence says the reward was not the binding problem in this run; the optimizer was.

## Honest read across phase 2 and 2b

Neither result is evidence the approach cannot work. Both are specification and engineering failures with known fixes:

- Phase 2: reward averaged away the signal → converged to recency
- Phase 2b: reward emphasised the signal and was learning → optimizer collapsed before the run ended

The uncomfortable summary is that **two GPU runs and $2.42 have produced no evidence about the actual research question.** Every policy tested, heuristic or learned, still scores exactly zero in the middle band. What has been established is that the phase-1 dead zone is robust, and that this particular training setup has three concrete bugs.

## Cost

$0.38 for phase 2b versus $2.04 for phase 2. The difference is entirely setup: the hardened `provision.sh` brought remote bring-up from over an hour to ~4 minutes, and all 81 tests passed on CUDA first try.

One process-level failure worth recording: the first eval launch silently never started, because `nohup` was issued inside an SSH session that was itself inside a local command that got backgrounded. `setsid nohup ... < /dev/null & disown` is the form that survives.

## Decision

**Phase 3 remains on hold.** A head-to-head against WG-KV needs a policy that does something; a reject-everything policy measures nothing.

Next step is phase 2c: fixes 1–3 above, rerun at β=1 and β=0 for comparison. If the step-150 behaviour holds to 300 steps under a stable optimizer, mid-band recall becomes testable for the first time.
